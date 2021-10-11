import os
import json
import numpy as np

from cleverspeech.utils.Utils import l_map, log

from cleverspeech.data.ingress.bases import (
    _BaseAudiosBatchETL,
    _BaseStandardAudioBatchETL,
    _BaseTranscriptionsBatchETL,
    _BaseBatchIterator as TwoStageIterableBatches,
)


class _BaseFromTwoStageAudios(_BaseAudiosBatchETL):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_file_paths(x):
        for fp in os.listdir(x):
            absolute_file_path = os.path.join(x, fp)
            basename = os.path.basename(fp)
            file_size = os.path.getsize(absolute_file_path)
            if "audio.wav" in absolute_file_path:
                yield file_size, absolute_file_path, basename

    def _extract(self, indir, numb_examples, filter_term=None):

        if not os.path.exists(indir):
            raise Exception("Path does not exist: {}".format(indir))

        fps = [x for x in self.get_file_paths(indir)]

        if filter_term:
            fps = list(
                filter(lambda x: filter_term in x[1], fps)
            )

        self.pool = fps

    def create_batch(self, batch_size):
        batch_data = self._popper(self.pool, batch_size)
        return self._transform_and_load(batch_data)

    @staticmethod
    def _popper(data, size):
        return l_map(
            lambda x: data.pop(x-1), range(size, 0, -1)
        )

    @staticmethod
    def get_batch_wav_file_paths(batch_data):
        return l_map(lambda x: x[1], batch_data)

    @staticmethod
    def get_batch_wav_basenames(batch_data):
        return l_map(lambda x: x[2], batch_data)

    @staticmethod
    def get_batch_ground_truth_file_paths(batch_size):
        return l_map(lambda _: "", range(batch_size))


    @staticmethod
    def get_single_ground_truth_phrase(file_path):
        return ""

    @staticmethod
    def get_batch_ground_truth_phrase_lengths(batch_size):
        return l_map(lambda _: 0, range(batch_size))

    def _transform_and_load(self, batch_data):
        # FIXME
        # meta
        file_paths = self.get_batch_wav_file_paths(batch_data)
        basenames = self.get_batch_wav_basenames(batch_data)
        audios = self.get_batch_audio(batch_data)
        n_samples = self.get_batch_n_samples(audios)

        ground_truth_phrase = self.get_batch_ground_truth_phrases(
            len(batch_data)
        )
        ground_truth_length = self.get_batch_ground_truth_phrase_lengths(
            len(batch_data)
        )

        # audio processing
        # no audio processing should occur in two stage

        # post processing meta
        maximum_padded_length = self.get_batch_max_padded_length(audios)
        padded = self.get_batch_padded_audio(audios, maximum_padded_length)
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


class TwoStageStandardAudioBatchETL(
    _BaseStandardAudioBatchETL, _BaseFromTwoStageAudios
):
    pass


class IncorrectAlignmentForTranscription(Exception):
    pass


class TwoStageTranscriptions(_BaseTranscriptionsBatchETL):

    def __init__(self, indir):
        super().__init__(indir)

    def _extract(self, indir):

        self.pool = list(
            filter(
                lambda x: ".json" in x and "setting" not in x,
                l_map(lambda i: os.path.join(indir, i), os.listdir(indir))
           )
        )
        self.indir = indir

    def create_batch(self, audios_batch, *args):
        selections = self._popper(audios_batch)
        return self._transform_and_load(selections)

    def _popper(self, audios_batch):

        jsonified_basenames = l_map(
            lambda x: x.replace("_audio.wav", ".json"),
            audios_batch["basenames"]
        )

        ordered_target_file_paths = l_map(
            lambda x: self.pool.pop(
                self.pool.index(os.path.join(self.indir, x))
            ),
            jsonified_basenames
        )

        return ordered_target_file_paths

    @staticmethod
    def validate(argmaxed, original):
        merged_argmax = []
        prev = None

        for entry in argmaxed:

            if prev is None and entry != 28:
                merged_argmax.append(entry)

            elif prev is None:
                merged_argmax.append(entry)

            elif prev == entry:
                pass

            else:
                merged_argmax.append(entry)

            prev = entry

        reduced_argmax = list(filter(lambda x: x != 28, merged_argmax))

        return all(x == y for x, y in zip(reduced_argmax, original))

    @staticmethod
    def get_single_json_data(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)[0]
        return data

    def get_batch_json_data(self, file_paths):
        return l_map(self.get_single_json_data, file_paths)

    @staticmethod
    def get_batch_transcriptions_text(ts):
        return l_map(lambda x: x["phrases"][0].replace("=", " "), ts)

    @staticmethod
    def get_batch_transcriptions_unique_id(ts):
        return l_map(lambda x: x["row_ids"][0], ts)

    @staticmethod
    def get_batch_transcription_lengths(ts):
        return l_map(lambda x: int(x["lengths"][0]), ts)

    @staticmethod
    def get_batch_transcription_as_original_indices(ts):
        return l_map(lambda x: x["original_indices"], ts)

    def get_batch_transcription_as_indices(self, ts):
        return l_map(lambda x: np.argmax(x["softmax_logits"], axis=-1), ts)

    def validate_argmax_indices(self, align_indices, orig_indices):
        return l_map(
            lambda x: self.validate(*x), zip(align_indices, orig_indices)
        )

    def _transform_and_load(self, batch_data):

        batched_json_data = self.get_batch_json_data(batch_data)

        target_phrases = self.get_batch_transcriptions_text(batched_json_data)
        row_ids = self.get_batch_transcriptions_unique_id(batched_json_data)
        lengths = self.get_batch_transcription_lengths(batched_json_data)
        original_indices = self.get_batch_transcription_as_original_indices(
            batched_json_data
        )
        indices = self.get_batch_transcription_as_indices(batched_json_data)

        maxlen = batched_json_data["max_feats"]

        padded_indices = self.pad_batch_of_transcription_indices(
            indices, maxlen
        )

        validate_merge_reduce = self.validate_argmax_indices(
            indices, original_indices
        )

        try:
            assert all(x is True for x in validate_merge_reduce)

        except AssertionError as e:

            s = "\033[1;31mERROR:\033[1;0m "
            s += "Not all target alignments match the intended target phrase!"
            s += "\n\nCheck your input data located in {}".format(self.indir)
            s += "\n==> Make sure that argmax(softmax, axis=-1) "
            s += "matches the target phrases"

            errors = l_map(
                lambda x: (x[0], x[1]),
                enumerate(validate_merge_reduce)
            )

            filtered_errors = l_map(
                lambda x: x[0],
                list(filter(lambda x: x[1] is False, errors))
            )

            error_basenames = l_map(
                lambda i: batched_json_data[i],
                filtered_errors
            )

            s += "\n\nThe following files are causing errors:\n"
            s += "\n".join(error_basenames)

            raise IncorrectAlignmentForTranscription(s)

        return {
            "tokens": self.tokens,
            "phrases": target_phrases,
            "row_ids": row_ids,
            "indices": padded_indices,
            "original_indices": original_indices,  # may modify for alignments
            "lengths": lengths,
        }