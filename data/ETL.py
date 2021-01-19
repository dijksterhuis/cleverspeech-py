import os
import json
import numpy as np

from abc import ABC, abstractmethod

from cleverspeech.data.Batches import TargetsBatch, AudiosBatch
from cleverspeech.utils.Utils import np_arr, np_zero, lcomp, load_wavs, l_map


class ETL(ABC):
    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def load(self):
        pass

    def run(self):
        self.extract().transform().load()

    @staticmethod
    def get_file_paths(x):
        for fp in os.listdir(x):
            absolute_file_path = os.path.join(x, fp)
            basename = os.path.basename(fp)
            file_size = os.path.getsize(absolute_file_path)
            yield (file_size, absolute_file_path, basename)


class AllAudioFilePaths(ETL):
    def __init__(self, indir, n, sort_by_file_size=True, filter_term=None, max_samples=None):

        if not os.path.exists(indir):
            raise Exception("Path does not exist: {}".format(indir))
        self.__data = indir

        # private vars
        self.__sort_by_file_size = sort_by_file_size
        self.__filter_term = filter_term
        self.__max_samples = max_samples
        self.__extracted = None
        self.__transformed = None

        # public vars
        self.numb_examples = n

    def extract(self):

        self.__extracted = [x for x in self.get_file_paths(self.__data)]

        return self

    def transform(self):

        def get_size_sorted_file_paths(fps, reverse=False):
            fps = [x for x in fps]
            fps.sort(key=lambda x: x[0], reverse=reverse)
            return fps

        fps = self.__extracted

        if self.__sort_by_file_size is not None:
            if self.__sort_by_file_size == 'desc':
                fps = get_size_sorted_file_paths(fps, reverse=True)
            elif self.__sort_by_file_size == 'asc':
                fps = get_size_sorted_file_paths(fps, reverse=False)
            else:
                # otherwise we'll sort anyway as it can affect optimisation
                fps = get_size_sorted_file_paths(fps, reverse=True)

        if self.__filter_term:
            fps = list(
                filter(lambda x: self.__filter_term in x[1], fps)
            )

        # bigger examples require more gpu memory / smaller batches
        if self.__max_samples:
            fps = list(
                filter(lambda x: x[0] < self.__max_samples, fps)
            )

        self.__transformed = fps[:self.numb_examples]

        return self

    def load(self):
        return self.__transformed


class AllTargetPhrases(ETL):
    def __init__(self, data, n):
        # private vars
        self.__data = data
        self.__extracted = None
        self.__transformed = None

        # public vars
        self.numb = n

    def extract(self):
        with open(self.__data, 'r') as f:
            data = f.readlines()

        self.__extracted = [
            (row.split(',')[1], idx) for idx, row in enumerate(data) if idx > 0
        ]

        return self

    def transform(self):
        targets = self.__extracted
        targets.sort(key=lambda x: len(x[0]), reverse=True)
        self.__transformed = targets[-self.numb:]
        return self

    def load(self):
        return self.__transformed


class AudioExamples(ETL):
    def __init__(self, data, dtype="int16"):

        # private vars
        self.__dtype = dtype
        self.__data = data
        self.__extracted = None
        self.__transformed = None

        # public vars
        self.maximum_length = None

    def extract(self):

        audio_fps = l_map(lambda x: x[1], self.__data)
        basenames = l_map(lambda x: x[2], self.__data)

        audios = lcomp(load_wavs(audio_fps, self.__dtype))

        self.__extracted = [audios, basenames]

        return self

    def transform(self):

        def padding(max_len):
            """
            Pad the audio samples to ensure that the CW mfcc framing doesn't break.
            Frame length of 512, split by frame step size 320.
            Recursively calculates padding again when pad length is > 512.

            :param max_len: maximum length of all samples in a batch
            :yield: size of additional pad
            """

            extra = max_len - (((max_len - 320) // 320) * 320)
            if extra > 512:
                return padding(max_len + ((extra // 512) * 512))
            else:
                return 512 - extra

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

        [audios, basenames] = self.__extracted

        maxlen = max(map(len, audios))
        self.maximum_length = maxlen + padding(maxlen)

        padded_audio = np_arr(
            lcomp(gen_padded_audio(audios, self.maximum_length)),
            np.float32
        )
        actual_lengths = np_arr(
            l_map(lambda x: x.size, audios),
            np.int32
        )

        maximum_feature_lengths = np_arr(
            l_map(
                lambda _: (self.maximum_length - 320) // 320,
                audios
            ),
            np.int32
        )
        actual_feature_lengths = np_arr(
            l_map(
                lambda x: (x.size - 320) // 320,
                audios
            ),
            np.int32
        )

        self.__transformed = [
            audios,
            basenames,
            padded_audio,
            actual_lengths,
            maximum_feature_lengths,
            actual_feature_lengths
        ]

        return self

    def load(self):
        return AudiosBatch(self.__transformed, self.maximum_length)


class TargetPhrases(ETL):
    def __init__(self, data, tokens=" abcdefghijklmnopqrstuvwxyz'-"):

        # private_vars
        self.__data = data
        self.__extracted = None
        self.__transformed = None

        # public vars
        self.tokens = tokens

    def extract(self):
        target_phrases = list(map(lambda x: x[0], self.__data))
        target_ids = list(map(lambda x: x[1], self.__data))
        self.__extracted = [target_phrases, target_ids]

        return self

    def transform(self):
        def get_indices(phrase):
            """
            Generate the target indices for CTC and alignments

            :return: array of target indices from tokens [phrase length]
            """
            indices = np_arr([self.tokens.index(i) for i in phrase], np.int32)
            return indices

        [target_phrases, target_ids] = self.__extracted

        indices = np_arr(
            l_map(lambda x: get_indices(x), target_phrases),
            np.int32
        )
        lengths = np_arr(
            l_map(lambda x: len(x), target_phrases),
            np.int32
        )

        self.__transformed = [
            target_phrases,
            target_ids,
            indices,
            lengths
        ]

        return self

    def load(self):
        return TargetsBatch(self.__transformed, self.tokens)

