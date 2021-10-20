"""
Base ETL classes.

Each ETL class **must** be extended for different data sets.
The `_BaseBatchIterator` can just be imported under a different name and used
with the extended ETL classes.

For audio, the different classes can:
- load an input set of speech examples (standard)
- apply pre-processing (i.e. trimming) to the speech examples
- generate silent examples that have the same length as the input samples
- generate white noise examples that have the same length as the input samples

For transcriptions, the base class loads candidates from a CSV file then
determines how to pair an audio example with a transcription based on some
rules.

Each example will have a transcription that is:
- completely unique (no other adversarial example will have that transcription)
- at most M / 4 in length
- at least 4 characters
- not equal to the ground truth transcription of the speech example

"""


import os
import json
import librosa
import numpy as np

from abc import ABC, abstractmethod

from cleverspeech.data.utils import wav_file
from cleverspeech.data.egress.load import convert_types_for_json
from cleverspeech.utils.Utils import (
    np_arr, np_zero, np_one, l_map, l_filter, l_sort, log
)

DEFAULT_TOKENS = " abcdefghijklmnopqrstuvwxyz'-"


class _IterableETL(ABC):

    def __init__(self, *args, **kwargs):
        self.pool = None
        self._extract(*args, **kwargs)

    @abstractmethod
    def _extract(self, *args, **kwargs):
        pass

    def __len__(self):
        return len(self.pool)

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
        if len(self) == 0:
            raise StopIteration

        return self.__next__(*args, **kwargs)

    def __next__(self, *args, **kwargs):
        return self.create_batch(*args, **kwargs)

    @abstractmethod
    def _popper(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_batch(self, *args, **kwargs):
        pass

    @abstractmethod
    def _transform_and_load(self, *args, **kwargs):
        pass


class AbstractAudioType(ABC):
    @property
    @abstractmethod
    def bit_depth(self):
        pass

    @property
    @abstractmethod
    def max_sample_value(self):
        pass

    @property
    @abstractmethod
    def min_sample_value(self):
        pass

    @property
    @abstractmethod
    def smallest_nonzero_value(self):
        pass

    @property
    @abstractmethod
    def signed(self):
        pass

    @property
    @abstractmethod
    def np_dtype(self):
        pass

    @property
    @abstractmethod
    def soundfile_dtype(self):
        pass


class Wav16BitSignedInt(AbstractAudioType):
    @property
    def bit_depth(self):
        return 16

    @property
    def max_sample_value(self):
        return 2**15-1

    @property
    def min_sample_value(self):
        return -2**15

    @property
    def smallest_nonzero_value(self):
        return 1

    @property
    def signed(self):
        return True

    @property
    def np_dtype(self):
        return np.int16

    @property
    def soundfile_dtype(self):
        return "int16"


class Wav32BitSignedFloat(AbstractAudioType):
    @property
    def bit_depth(self):
        return 32

    @property
    def max_sample_value(self):
        return 1.0

    @property
    def min_sample_value(self):
        return -1.0

    @property
    def smallest_nonzero_value(self):
        # NOTE: this is close to smallest possible value ~8e-46 for np.float32
        # and tf.float32
        return 1e-45

    @property
    def signed(self):
        return True

    @property
    def np_dtype(self):
        return np.float32

    @property
    def soundfile_dtype(self):
        return "float32"


class OneQuarterScaleWav32BitSignedFloat(Wav32BitSignedFloat):
    @property
    def max_sample_value(self):
        return 1.0 * 0.25

    @property
    def min_sample_value(self):
        return -1.0 * 0.25


class OneHalfScaleWav32BitSignedFloat(Wav32BitSignedFloat):
    @property
    def max_sample_value(self):
        return 1.0 * 0.5

    @property
    def min_sample_value(self):
        return -1.0 * 0.5


class ThreeQuarterScaleWav32BitSignedFloat(Wav32BitSignedFloat):
    @property
    def max_sample_value(self):
        return 1.0 * 0.75

    @property
    def min_sample_value(self):
        return -1.0 * 0.75


class _BaseAudiosBatchETL(_IterableETL):
    """
    NOTE: Change the output_dtype class to scale the amplitude of input data.
    For example, for 50% amplitude set the output data type during
    object init to `MyETLClass(output_dtype=OneHalfScaleWav32BitSignedFloat())`
    """

    def __init__(
            self, *args,
            input_dtype=Wav16BitSignedInt(),
            output_dtype=Wav32BitSignedFloat(),
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    @staticmethod
    @abstractmethod
    def get_batch_wav_file_paths(batch_data):
        pass

    @staticmethod
    @abstractmethod
    def get_batch_wav_basenames(batch_data):
        pass

    @staticmethod
    @abstractmethod
    def get_batch_ground_truth_file_paths(batch_data):
        pass

    @staticmethod
    @abstractmethod
    def get_single_ground_truth_phrase(file_path):
        pass

    def get_batch_ground_truth_phrases(self, file_paths):
        return l_map(self.get_single_ground_truth_phrase, file_paths)

    @staticmethod
    def get_batch_ground_truth_phrase_lengths(phrases):
        return l_map(len, phrases)

    def convert_audio_type(self, audio):
        return audio.astype(self.output_dtype.np_dtype)

    def get_batch_audio(self, fps):

        return l_map(
            lambda x: self.convert_audio_type(
                wav_file.load(x, self.input_dtype.soundfile_dtype)
            ),
            fps
        )

    def get_single_pseudo_silence(self, audio):
        level = self.output_dtype.smallest_nonzero_value
        np_dtype = self.output_dtype.np_dtype
        return np_one(audio.shape, np_dtype) * level

    def get_batch_pseudo_silence(self, audios):
        return l_map(self.get_single_pseudo_silence, audios)

    def get_single_maximum_amplitude(self, audio):
        level = self.output_dtype.max_sample_value
        np_dtype = self.output_dtype.np_dtype
        return np_one(audio.shape, np_dtype) * level

    def get_batch_maximum_amplitude(self, audios):
        return l_map(self.get_single_maximum_amplitude, audios)

    def get_single_white_noise(self, audio):
        max_level = self.output_dtype.max_sample_value
        min_level = self.output_dtype.min_sample_value
        white_noise = np.random.uniform(max_level, min_level, audio.shape)
        return white_noise.astype(self.output_dtype.np_dtype)

    def get_batch_white_noise(self, audios):
        return l_map(self.get_single_white_noise, audios)

    @staticmethod
    def rms_to_dbfs(rms):
        return 20.0 * np.log10(max(1e-16, rms)) + 3.0103

    def max_dbfs(self, sample_data):
        # Peak dBFS based on the maximum energy sample.
        # Will prevent overdrive if used for normalization.
        return self.rms_to_dbfs(
            max(abs(np.min(sample_data)), abs(np.max(sample_data)))
        )

    @staticmethod
    def gain_db_to_ratio(gain_db):
        return np.power(10.0, gain_db / 20.0)

    def normalize(self, sample_data, dbfs=3.0103):
        return np.maximum(
            np.minimum(
                sample_data * self.gain_db_to_ratio(
                    dbfs - self.max_dbfs(sample_data)
                ), 1.0
            ), -1.0
        )

    def get_single_scaled(self, audio):

        input_max = self.input_dtype.max_sample_value
        input_min = self.input_dtype.min_sample_value

        output_max = self.output_dtype.max_sample_value
        output_min = self.output_dtype.min_sample_value

        audio[audio > 0] = audio[audio > 0] * output_max / input_max
        audio[audio < 0] = audio[audio < 0] * output_min / input_min

        return audio

    def get_batch_scaled(self, audios):
        return l_map(self.get_single_scaled, audios)

    def get_batch_normalised_audio(self, audios):
        return l_map(self.normalize, audios)

    @staticmethod
    def get_single_trimmed(audio, ref_fn, top_db):
        return librosa.effects.trim(audio, ref=ref_fn, top_db=top_db)[0]

    def get_batch_trimmed(self, audio, ref_fn=np.max, top_db=24.0):
        return l_map(
            lambda x: self.get_single_trimmed(x, ref_fn, top_db), audio
        )

    def apply_single_dither(self, audio):
        level = self.output_dtype.smallest_nonzero_value * 2
        return audio + np.random.uniform(-level, level)

    def get_batch_dithered(self, audios):
        return l_map(self.apply_single_dither, audios)

    def apply_single_nonzero_values(self, audio):
        audio[audio == 0] = self.output_dtype.smallest_nonzero_value
        return audio

    def get_batch_nonzero_values(self, audios):
        return l_map(self.apply_single_nonzero_values, audios)

    @staticmethod
    def get_batch_max_samples(audios):
        return max(l_map(lambda x: x.size, audios))

    def padding(self, max_len):
        """
        Frame length of 512, split by frame step size 320.
        Recursively calculates padding again when pad length is > 512.
        """
        extra = max_len - (((max_len - 320) // 320) * 320)
        if extra > 512:
            return self.padding(max_len + ((extra // 512) * 512))
        else:
            return 512 - extra

    def get_batch_max_padded_length(self, audios):
        max_samples = self.get_batch_max_samples(audios)
        return max_samples + self.padding(max_samples)

    @staticmethod
    def get_single_padded_audio(audio, max_padded_length):
        return np.concatenate(
            [audio, np_zero(max_padded_length - audio.size, np.float32)]
        )

    def get_batch_padded_audio(self, audios, max_padded_length):
        batch_padded = l_map(
            lambda x: self.get_single_padded_audio(x, max_padded_length),
            audios
        )
        return np_arr(batch_padded, np.float32)

    @staticmethod
    def get_single_n_samples(unpadded):
        return unpadded.size

    def get_batch_n_samples(self, unpadded):
        return np_arr(l_map(self.get_single_n_samples, unpadded), np.int32)

    @staticmethod
    def get_single_max_n_feats(maximum_padded_length):
        return np.round((maximum_padded_length - 320) / 320)

    def get_batch_max_n_feats(self, batch_size, maximum_padded_length):
        max_feats = l_map(
            lambda _: self.get_single_max_n_feats(maximum_padded_length),
            range(batch_size)
        )
        return np_arr(max_feats, np.int32)

    def get_single_actual_n_feats(self, audio):
        n_samples = self.get_single_n_samples(audio)
        return np.round((n_samples - 320) / 320)

    def get_batch_actual_n_feats(self, audios):
        return np_arr(l_map(self.get_single_actual_n_feats, audios), np.int32)


class _BaseStandardAudioBatchETL(_BaseAudiosBatchETL):
    def _transform_and_load(self, batch_data):

        # meta
        file_paths = self.get_batch_wav_file_paths(batch_data)
        basenames = self.get_batch_wav_basenames(batch_data)
        audios = self.get_batch_audio(file_paths)
        n_samples = self.get_batch_n_samples(audios)

        ground_truth_file_paths = self.get_batch_ground_truth_file_paths(
            batch_data
        )
        ground_truth_phrase = self.get_batch_ground_truth_phrases(
            ground_truth_file_paths
        )
        ground_truth_length = self.get_batch_ground_truth_phrase_lengths(
            ground_truth_phrase
        )

        # audio processing
        scaled = self.get_batch_scaled(audios)
        normalised = self.get_batch_normalised_audio(scaled)
        non_zero = self.get_batch_nonzero_values(normalised)

        # post processing meta
        maximum_padded_length = self.get_batch_max_padded_length(non_zero)
        padded = self.get_batch_padded_audio(non_zero, maximum_padded_length)
        max_n_features = self.get_batch_max_n_feats(
            len(batch_data), maximum_padded_length
        )
        actual_n_features = self.get_batch_actual_n_feats(audios)

        return {
            "file_paths": file_paths,
            "max_samples": maximum_padded_length,
            "max_feats": max_n_features[0],
            "audio": audios,
            "padded_audio": padded,
            "basenames": basenames,
            "n_samples": n_samples,
            "ds_feats": max_n_features,
            "real_feats": actual_n_features,
            "ground_truth": {
                "phrase": ground_truth_phrase,
                "phrase_length": ground_truth_length,
            },
        }


class _BaseTrimmedAudioBatchETL(_BaseAudiosBatchETL):

    def _transform_and_load(self, batch_data):

        # meta
        file_paths = self.get_batch_wav_file_paths(batch_data)
        basenames = self.get_batch_wav_basenames(batch_data)
        audios = self.get_batch_audio(file_paths)
        n_samples = self.get_batch_n_samples(audios)

        ground_truth_file_paths = self.get_batch_ground_truth_file_paths(
            batch_data
        )
        ground_truth_phrase = self.get_batch_ground_truth_phrases(
            ground_truth_file_paths
        )
        ground_truth_length = self.get_batch_ground_truth_phrase_lengths(
            ground_truth_phrase
        )

        # audio processing
        scaled = self.get_batch_scaled(audios)
        normalised = self.get_batch_normalised_audio(scaled)
        non_zero = self.get_batch_nonzero_values(normalised)
        trimmed = self.get_batch_nonzero_values(non_zero)

        # post processing meta
        maximum_padded_length = self.get_batch_max_padded_length(trimmed)
        padded = self.get_batch_padded_audio(trimmed, maximum_padded_length)
        max_n_features = self.get_batch_max_n_feats(
            len(batch_data), maximum_padded_length
        )
        actual_n_features = self.get_batch_actual_n_feats(audios)

        return {
            "file_paths": file_paths,
            "max_samples": maximum_padded_length,
            "max_feats": max_n_features[0],
            "audio": audios,
            "padded_audio": padded,
            "basenames": basenames,
            "n_samples": n_samples,
            "ds_feats": max_n_features,
            "real_feats": actual_n_features,
            "ground_truth": {
                "phrase": ground_truth_phrase,
                "phrase_length": ground_truth_length,
            },
        }


class _BaseSilenceAudioBatchETL(_BaseAudiosBatchETL):

    @staticmethod
    def get_single_ground_truth_phrase(file_path):
        return ""

    def _transform_and_load(self, batch_data):

        # meta
        file_paths = self.get_batch_wav_file_paths(batch_data)
        basenames = self.get_batch_wav_basenames(batch_data)
        audios = self.get_batch_audio(file_paths)
        n_samples = self.get_batch_n_samples(audios)

        ground_truth_phrase = self.get_batch_ground_truth_phrases(
            range(len(batch_data))
        )
        ground_truth_length = self.get_batch_ground_truth_phrase_lengths(
            ground_truth_phrase
        )

        # audio processing
        silenced = self.get_batch_pseudo_silence(audios)
        non_zero = self.get_batch_nonzero_values(silenced)

        # post processing meta
        maximum_padded_length = self.get_batch_max_padded_length(non_zero)
        padded = self.get_batch_padded_audio(silenced, maximum_padded_length)
        max_n_features = self.get_batch_max_n_feats(
            len(batch_data), maximum_padded_length
        )
        actual_n_features = self.get_batch_actual_n_feats(audios)

        return {
            "file_paths": file_paths,
            "max_samples": maximum_padded_length,
            "max_feats": max_n_features[0],
            "audio": audios,
            "padded_audio": padded,
            "basenames": basenames,
            "n_samples": n_samples,
            "ds_feats": max_n_features,
            "real_feats": actual_n_features,
            "ground_truth": {
                "phrase": ground_truth_phrase,
                "phrase_length": ground_truth_length,
            },
        }


class _BaseConstantAmplitudeAudioBatchETL(_BaseAudiosBatchETL):

    @staticmethod
    def get_single_ground_truth_phrase(file_path):
        return ""

    def _transform_and_load(self, batch_data):

        # meta
        file_paths = self.get_batch_wav_file_paths(batch_data)
        basenames = self.get_batch_wav_basenames(batch_data)
        audios = self.get_batch_audio(file_paths)
        n_samples = self.get_batch_n_samples(audios)

        ground_truth_phrase = self.get_batch_ground_truth_phrases(
            range(len(batch_data))
        )
        ground_truth_length = self.get_batch_ground_truth_phrase_lengths(
            ground_truth_phrase
        )

        # audio processing
        silenced = self.get_batch_pseudo_silence(audios)
        non_zero = self.get_batch_nonzero_values(silenced)

        # post processing meta
        maximum_padded_length = self.get_batch_max_padded_length(non_zero)
        padded = self.get_batch_padded_audio(silenced, maximum_padded_length)
        max_n_features = self.get_batch_max_n_feats(
            len(batch_data), maximum_padded_length
        )
        actual_n_features = self.get_batch_actual_n_feats(audios)

        return {
            "file_paths": file_paths,
            "max_samples": maximum_padded_length,
            "max_feats": max_n_features[0],
            "audio": audios,
            "padded_audio": padded,
            "basenames": basenames,
            "n_samples": n_samples,
            "ds_feats": max_n_features,
            "real_feats": actual_n_features,
            "ground_truth": {
                "phrase": ground_truth_phrase,
                "phrase_length": ground_truth_length,
            },
        }


class _BaseWhiteNoiseAudioBatchETL(_BaseAudiosBatchETL):

    @staticmethod
    def get_single_ground_truth_phrase(file_path):
        return ""

    def _transform_and_load(self, batch_data):

        # meta
        file_paths = self.get_batch_wav_file_paths(batch_data)
        basenames = self.get_batch_wav_basenames(batch_data)
        audios = self.get_batch_audio(file_paths)
        n_samples = self.get_batch_n_samples(audios)

        ground_truth_phrase = self.get_batch_ground_truth_phrases(
            range(len(batch_data))
        )
        ground_truth_length = self.get_batch_ground_truth_phrase_lengths(
            ground_truth_phrase
        )

        # audio processing
        noise = self.get_batch_white_noise(audios)

        # post processing meta
        maximum_padded_length = self.get_batch_max_padded_length(noise)
        padded = self.get_batch_padded_audio(noise, maximum_padded_length)
        max_n_features = self.get_batch_max_n_feats(
            len(batch_data), maximum_padded_length
        )
        actual_n_features = self.get_batch_actual_n_feats(audios)

        return {
            "file_paths": file_paths,
            "max_samples": maximum_padded_length,
            "max_feats": max_n_features[0],
            "audio": audios,
            "padded_audio": padded,
            "basenames": basenames,
            "n_samples": n_samples,
            "ds_feats": max_n_features,
            "real_feats": actual_n_features,
            "ground_truth": {
                "phrase": ground_truth_phrase,
                "phrase_length": ground_truth_length,
            },
        }


class NoSuitableTranscriptionFoundException(Exception):
    pass


class _BaseTranscriptionsBatchETL(_IterableETL, ABC):

    def __init__(self, *args, tokens=DEFAULT_TOKENS, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens = tokens

    def _extract(self, *args, **kwargs):
        pass

    def create_batch(self, batch_size, audios_batch):
        selections = self._popper(batch_size, audios_batch)
        return self._transform_and_load(selections)

    @staticmethod
    def _selection_rules(phrase, n_feats, ground_truth, selections):
        upper_bound = len(phrase) <= n_feats // 4
        lower_bound = len(phrase) >= 4
        not_ground_truth = phrase != ground_truth
        existing_phrases = l_map(lambda x: x[0], selections)
        not_selected = phrase not in existing_phrases

        return upper_bound and lower_bound and not_ground_truth and not_selected

    @staticmethod
    def get_batch_transcriptions_text(ts):
        return l_map(lambda x: x[0], ts)

    @staticmethod
    def get_single_transcription_as_indices(t, tokens):
        indices = np_arr([tokens.index(i) for i in t], np.int32)
        return indices

    def get_batch_transcription_as_indices(self, ts):
        return l_map(
            lambda x: self.get_single_transcription_as_indices(x, self.tokens),
            ts
        )

    @staticmethod
    def pad_single_transcription_indices(t, max_len):
        return np.concatenate(
            [t, np.zeros(max_len - len(t))]
        )

    def pad_batch_of_transcription_indices(self, ts, max_len):
        padded = l_map(
            lambda x: self.pad_single_transcription_indices(x, max_len),
            ts
        )
        return np_arr(padded, np.int32)

    @staticmethod
    def get_batch_transcription_lengths(ts):
        return np_arr(l_map(lambda x: len(x), ts), np.int32)

    @staticmethod
    def get_batch_maximum_transcription_length(ts):
        return max(map(len, ts))

    @staticmethod
    def get_batch_transcriptions_unique_id(ts):
        return list(map(lambda x: x[1], ts))


class Batch:
    """
    A batch of data to use in an attack.

    :param size: the number of audio examples
    :param audios: the audio data (and metadata)
    :param targets: the targeting data (and metadata)
    """
    def __init__(self, idx, size, audios, targets):

        self.idx = idx
        self.size = size
        self.audios = audios
        self.targets = targets


class _BaseBatchIterator:
    def __init__(self, settings, audios, targets):

        self.current_idx = 0
        self.audios = audios
        self.targets = targets
        self.n_examples = len(self.audios)
        self.batch_size = settings["batch_size"]

        base_out_dir = os.path.join(
            settings["outdir"], str(settings["unique_run_id"])
        )
        self.out_file_path = os.path.join(base_out_dir, "batches.json")

        n_batches = self.n_examples / self.batch_size

        if n_batches > int(n_batches):
            self.n_batches = int(n_batches) + 1
        else:
            self.n_batches = int(n_batches)

        log(
            "New Run",
            "Number of test examples: {}".format(self.n_examples),
            ''.join(
                ["{k}: {v}\n".format(k=k, v=v) for k, v in settings.items()]),
        )

    def _write_batch_to_json_file(self, batch):

        d = [
            {
                batch.idx: {
                    "size": batch.size,
                    "audios": convert_types_for_json(batch.audios),
                    "targets": convert_types_for_json(batch.targets),
                }
            }
        ]

        if os.path.exists(self.out_file_path):
            with open(self.out_file_path, "r") as f:
                batch_data = json.load(f)
            d += batch_data

        with open(self.out_file_path, "w+") as f:
            json.dump(
                d, f,
                indent=2,
                sort_keys=True,
                ensure_ascii=True
            )

    def __next__(self):

        self.current_idx += 1

        if self.current_idx > self.n_batches:
            raise StopIteration

        # Handle remainders: number of examples // desired batch size != 0
        if len(self.audios) < self.batch_size:
            batch_size = len(self.audios)
        else:
            batch_size = self.batch_size

        audios_batch = self.audios.next(batch_size)
        targets_batch = self.targets.next(batch_size, audios_batch)

        batch = Batch(
            self.current_idx,
            batch_size,
            audios_batch,
            targets_batch,
        )

        log("Writing batch data to {}".format(self.out_file_path))
        self._write_batch_to_json_file(batch)

        return batch

    def __iter__(self):
        return self
