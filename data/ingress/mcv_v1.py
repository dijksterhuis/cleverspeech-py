import os
import json
import random
import numpy as np

from cleverspeech.utils.Utils import np_arr, np_zero, lcomp, l_map, log
from cleverspeech.data.utils import WavFile

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"


class IterableETL:
    pool = None

    def __len__(self):
        return len(self.pool)

    def __iter__(self):
        return self

    def next(self, *args):

        if len(self) == 0:
            raise StopIteration

        return self.__next__(*args)


class Audios(IterableETL):

    def __init__(self, indir, numb_examples, file_size_sort=True, filter_term=None, max_file_size=None, min_file_size=None):

        if not os.path.exists(indir):
            raise Exception("Path does not exist: {}".format(indir))

        fps = [x for x in self.get_file_paths(indir)]

        if file_size_sort is not None:
            if file_size_sort == 'desc':
                fps = self.get_size_sorted_file_paths(fps, reverse=True)
            elif file_size_sort == 'asc':
                fps = self.get_size_sorted_file_paths(fps, reverse=False)
            else:
                # otherwise we'll sort anyway as it can affect optimisation
                fps = self.get_size_sorted_file_paths(fps, reverse=True)

        if filter_term:
            fps = list(
                filter(lambda x: filter_term in x[1], fps)
            )

        # bigger examples require more gpu memory and potentially smaller batch
        if max_file_size:
            fps = list(
                filter(lambda x: x[0] <= max_file_size, fps)
            )

        if min_file_size:
            fps = list(
                filter(lambda x: x[0] >= min_file_size, fps)
            )

        self.pool = fps[:numb_examples]

    @staticmethod
    def get_file_paths(x):
        for fp in os.listdir(x):
            absolute_file_path = os.path.join(x, fp)
            basename = os.path.basename(fp)
            file_size = os.path.getsize(absolute_file_path)
            yield (file_size, absolute_file_path, basename)

    @staticmethod
    def get_size_sorted_file_paths(fps, reverse=False):
        fps = [x for x in fps]
        fps.sort(key=lambda x: x[0], reverse=reverse)
        return fps

    def __next__(self, batch_size):
        return self.create_batch(self.popper(self.pool, batch_size))

    @staticmethod
    def popper(data, size):
        return l_map(
            lambda x: data.pop(x-1), range(size, 0, -1)
        )

    def create_batch(self, batched_file_path_data, dtype="int16"):

        audio_fps = l_map(lambda x: x[1], batched_file_path_data)
        basenames = l_map(lambda x: x[2], batched_file_path_data)

        audios = lcomp([WavFile.load(f, dtype) for f in audio_fps])

        # N.B. ==> If audios is 0 at any point then the perturbation will always
        # be zero for that sample due to a zero gradient. so add 1 to zero
        # samples make  backpropogation work for all samples (1/2**15 is small
        # so side-effects should be minimal).
        for audio in audios:
            audio[audio == 0] = 1.0

        maxlen = max(map(len, audios))
        maximum_length = maxlen + self.padding(maxlen)

        padded_audio = np_arr(
            lcomp(self.gen_padded_audio(audios, maximum_length)),
            np.float32
        )
        actual_lengths = np_arr(
            l_map(lambda x: x.size, audios),
            np.int32
        )

        # N.B. Remember to use round instead of integer division here!
        maximum_feature_lengths = np_arr(
            l_map(
                lambda _: np.round((maximum_length - 320) / 320),
                audios
            ),
            np.int32
        )

        actual_feature_lengths = np_arr(
            l_map(
                lambda x: np.round((x.size - 320) / 320),
                audios
            ),
            np.int32
        )

        return {
            "file_paths": audio_fps,
            "max_samples": maximum_length,
            "max_feats": maximum_feature_lengths[0],
            "audio": audios,
            "padded_audio": padded_audio,
            "basenames": basenames,
            "n_samples": actual_lengths,
            "ds_feats": maximum_feature_lengths,
            "real_feats": actual_feature_lengths,
        }

    def padding(self, max_len):
        """
        Pad the audio samples to ensure that the CW mfcc framing doesn't break.
        Frame length of 512, split by frame step size 320.
        Recursively calculates padding again when pad length is > 512.

        :param max_len: maximum length of all samples in a batch
        :yield: size of additional pad
        """

        extra = max_len - (((max_len - 320) // 320) * 320)
        if extra > 512:
            return self.padding(max_len + ((extra // 512) * 512))
        else:
            return 512 - extra

    @staticmethod
    def gen_padded_audio(audios, max_len):
        """
        Add the required padding to each example.
        :param audios: the batch of audio examples
        :param max_len: the length of the longest audio example in the batch
        :return: audio array padded to length max_len
        """
        for audio in audios:
            b = np_zero(max_len - audio.size, np.float32)
            yield np.concatenate([audio, b])


class Targets(IterableETL):

    def __init__(self, indir, numb):

        with open(indir, 'r') as f:
            data = f.readlines()

        targets = [
            (row.split(',')[1], idx) for idx, row in enumerate(data) if
            idx > 0
        ]
        targets.sort(key=lambda x: len(x[0]), reverse=False)

        self.pool = targets[:numb]

    def __next__(self, *args):
        return self.create_batch(*args)

    def create_batch(self, batch_size, trues_batch, audios_batch, tokens=TOKENS):

        target_data = []
        for i in range(batch_size):
            p = self.pop_target_phrase(
                trues_batch["true_targets"],
                min(audios_batch["real_feats"]) - 4
            )
            target_data.append(p)

        target_phrases = l_map(lambda x: x[0], target_data)

        lengths = np_arr(
            l_map(lambda x: len(x), target_phrases),
            np.int32
        )

        row_ids = list(map(lambda x: x[1], target_data))
        maxlen = max(l_map(lambda x: len(x[0]), target_data))

        original_indices = l_map(
            lambda x: self.get_indices(x, tokens),
            target_phrases
        )

        padded_indices = np_arr(
            l_map(
                lambda x: np.concatenate(
                    [x, np.zeros(maxlen - len(x))]
                ),
                original_indices
            ),
            np.int32
        )

        return {
            "tokens": tokens,
            "phrases": target_phrases,
            "row_ids": row_ids,
            "indices": padded_indices,
            "original_indices": original_indices ,  # we may modify for alignments
            "lengths": lengths,
        }

    def pop_target_phrase(self, true_targets, min_feats, idx=0):
        candidate_target = random.choice(self.pool)

        length_test = len(candidate_target[0]) > min_feats
        matches_true_test = candidate_target[0] in true_targets

        if length_test or matches_true_test:
            return self.pop_target_phrase(
                true_targets, min_feats, idx=idx + 1
            )
        else:
            return candidate_target

    @staticmethod
    def get_indices(phrase, tokens):
        """
        Generate the target indices for CTC and alignments

        :return: array of target indices from tokens [phrase length]
        """
        indices = np_arr([tokens.index(i) for i in phrase], np.int32)
        return indices


def create_true_batch(audio_batch, tokens=TOKENS):
    metadata_fps = l_map(
        lambda fp: fp.replace(".wav", ".json"), audio_batch["file_paths"]
    )

    metadatas = l_map(
        lambda fp: json.load(open(fp, 'r'))[0], metadata_fps
    )

    true_transcriptions = l_map(
        lambda m: m["correct_transcription"], metadatas
    )

    true_transcriptions_indices = l_map(
        lambda t: Targets.get_indices(t, tokens),
        true_transcriptions
    )

    true_transcriptions_lengths = l_map(len, true_transcriptions)
    max_len = max(true_transcriptions_lengths)

    def pad_trues(xs):
        for x in xs:
            b = np.zeros(max_len - x.size, np.int32)
            yield np.concatenate([x, b])

    padded_true_trans_indices = [
        t for t in pad_trues(true_transcriptions_indices)
    ]
    return {
        "true_targets": true_transcriptions,
        "padded_indices": padded_true_trans_indices,
        "indices": true_transcriptions_indices,
        "lengths": true_transcriptions_lengths,
    }


class Batch:
    """
    A batch of data to use in an attack.

    TODO: Should Batch be a class, or a dictionary?!

    :param size: the number of audio examples
    :param audios: the audio data (and metadata)
    :param targets: the targeting data (and metadata)
    :param trues: the true transcriptions for each example
    """
    def __init__(self, size, audios, targets, trues):

        self.size = size
        self.audios = audios
        self.targets = targets
        self.trues = trues


class BatchIterator:
    def __init__(self, settings):

        # get N samples of all the data. alsp make sure to limit example length,
        # otherwise we'd have to do adaptive batch sizes.

        self.current_idx = 0

        self.audios = Audios(
            settings["audio_indir"],
            settings["max_examples"],
            filter_term=".wav",
            max_file_size=settings["max_audio_file_bytes"]
        )

        self.targets = Targets(
            settings["targets_path"], settings["max_targets"],
        )

        # Generate the batches in turn, rather than all in one go ...
        # to save resources by only running the final ETLs on a batch of data

        self.n_examples = len(self.audios)
        self.batch_size = settings["batch_size"]

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

    def __next__(self):

        self.current_idx += 1

        if self.current_idx > self.n_batches:
            raise StopIteration

        # Handle remainders: number of examples // desired batch size != 0
        if len(self.audios) < self.batch_size:
            batch_size = len(self.audios)
        else:
            batch_size = self.batch_size

        # get n files paths and create the audio batch data
        audios_batch = self.audios.next(batch_size)
        # load a valid target transcription for the audio files (must not match
        # original true transcription)
        trues_batch = create_true_batch(audios_batch)
        targets_batch = self.targets.next(batch_size, trues_batch, audios_batch)

        batch = Batch(
            batch_size,
            audios_batch,
            targets_batch,
            trues_batch,
        )

        return self.current_idx, batch

    def __iter__(self):
        return self

