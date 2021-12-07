import os
import random
import json

from cleverspeech.utils.Utils import l_map, l_filter, l_sort

from cleverspeech.data.ingress.bases import (
    _BaseAudiosBatchETL,
    _BaseStandardAudioBatchETL,
    _BaseTrimmedAudioBatchETL,
    _BaseSilenceAudioBatchETL,
    _BaseWhiteNoiseAudioBatchETL,
    _BaseTranscriptionsBatchETL,
    _BaseConstantAmplitudeAudioBatchETL,
    _BaseBatchIterator as IterableBatches,
    NoSuitableTranscriptionFoundException,
)

from cleverspeech.data.ingress import downloader


class _BaseFromAudios(_BaseAudiosBatchETL):

    def __init__(self, s3_key, *args, **kwargs):
        downloader.download(s3_key)
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_file_paths(x):
        for fp in os.listdir(x):
            if "json" not in fp:
                absolute_file_path = os.path.join(x, fp)
                basename = os.path.basename(fp)
                file_size = os.path.getsize(absolute_file_path)
                ground_truth_file_path = absolute_file_path.rstrip(".wav") + ".json"
                yield file_size, absolute_file_path, basename, ground_truth_file_path

    @staticmethod
    def get_size_sorted_file_paths(fps, reverse=False):
        assert type(fps) is list
        fps.sort(key=lambda x: x[0], reverse=reverse)
        return fps

    def _extract(self, indir, numb_examples, file_size_sort=None, filter_term=None, max_file_size=None, min_file_size=None):

        if not os.path.exists(indir):
            raise Exception("Path does not exist: {}".format(indir))

        fps = l_map(lambda x: x, self.get_file_paths(indir))

        if file_size_sort is not None:

            if file_size_sort == 'desc':
                fps = l_sort(lambda x: x[0], fps, reverse=True)

            elif file_size_sort == 'asc':
                fps = l_sort(lambda x: x[0], fps, reverse=False)

            elif file_size_sort == 'shuffle':

                # os.listdir returns filenames in an **arbitrary** ordering
                # determined by the *OS*. Sort first *then* shuffle otherwise
                # you will load different examples on different machines/OSs.

                fps = sorted(fps)
                random.shuffle(fps)
            else:
                # otherwise we'll sort by ascending file sizes for memory
                fps = l_sort(lambda x: x[0], fps, reverse=False)

        if filter_term:
            fps = l_filter(lambda x: filter_term in x[1], fps)

        # bigger examples require more gpu memory and potentially smaller batch
        if max_file_size:
            fps = l_filter(lambda x: x[0] <= max_file_size, fps)

        if min_file_size:
            fps = l_filter(lambda x: x[0] >= min_file_size, fps)

        self.pool = fps[:numb_examples]

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
    def get_batch_ground_truth_file_paths(batch_data):
        return l_map(lambda x: x[3], batch_data)

    @staticmethod
    def get_single_ground_truth_phrase(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)[0]
        return data["correct_transcription"]

    @staticmethod
    def get_batch_ground_truth_phrase_lengths(phrase):
        return l_map(len, phrase)


class StandardAudioBatchETL(
    _BaseStandardAudioBatchETL, _BaseFromAudios
):
    pass


class TrimmedAudioBatchETL(
    _BaseTrimmedAudioBatchETL, _BaseFromAudios
):
    pass


class SilenceAudioBatchETL(
    _BaseSilenceAudioBatchETL, _BaseFromAudios
):
    pass


class ConstantAmplitudeAudioBatchETL(
    _BaseConstantAmplitudeAudioBatchETL, _BaseFromAudios
):
    pass


class WhiteNoiseAudioBatchETL(
    _BaseWhiteNoiseAudioBatchETL, _BaseFromAudios
):
    pass


class TranscriptionsFromCSVFile(_BaseTranscriptionsBatchETL):

    def __init__(self, csv_file_path, numb):
        super().__init__(csv_file_path, numb)

    def _extract(self, csv_file_path, numb):

        with open(csv_file_path, 'r') as f:
            data = f.readlines()

        targets = [
            (row.split(',')[1], idx) for idx, row in enumerate(data) if idx > 0
        ]
        targets = l_sort(lambda x: len(x[0]), targets, reverse=False)

        self.pool = targets[:numb]

    def create_batch(self, batch_size, audios_batch):
        selections = self._popper(batch_size, audios_batch)
        return self._transform_and_load(selections)

    def _popper(self, batch_size, audios_batch):

        selections = []

        candidates = self.pool

        for i in range(batch_size):

            n_feats = audios_batch["real_feats"][i]
            ground_truth = audios_batch["ground_truth"]["phrase"][i]

            selection = None

            # first perform a pass over the shuffled candidates
            # grab a candidate that suits the rules

            for candidate in candidates:

                phrase, _ = candidate

                validate = self._selection_rules(
                    *candidate, n_feats, ground_truth, selections
                )

                if validate:
                    selection = candidate

            # if a candidate could not be found, search again
            # this time splitting the candidates by space characters
            # to find a single suitable single word

            if selection is None:
                for candidate in candidates:

                    phrase, row_id = candidate

                    word = random.choice(phrase.split(" "))

                    validate = self._selection_rules(
                        word, row_id, n_feats, ground_truth, selections
                    )

                    if validate:
                        selection = (word, candidate[1])

            # if nothing suitable has been found the raise an exception
            # ==> something is dodgy in your data and you need to check things

            if selection is None:
                s = "No suitable transcription could be found for {}".format(
                        audios_batch["basenames"][i]
                    )
                raise NoSuitableTranscriptionFoundException(s)

            selections.append(selection)

        # now pop transcriptions from the pool to avoid assigning duplicate
        # transcriptions over different batch

        for selection in selections:
            if selection in self.pool:
                self.pool.remove(selection)

        return selections

    def _transform_and_load(self, batch_data):

        phrases = self.get_batch_transcriptions_text(batch_data)
        lengths = self.get_batch_transcription_lengths(phrases)
        max_len = self.get_batch_maximum_transcription_length(phrases)
        ragged_indices = self.get_batch_transcription_as_indices(phrases)
        padded_indices = self.pad_batch_of_transcription_indices(
            ragged_indices, max_len
        )

        row_ids = self.get_batch_transcriptions_unique_id(batch_data)

        return {
            "tokens": self.tokens,
            "phrases": phrases,
            "row_ids": row_ids,
            "indices": padded_indices,
            "original_indices": ragged_indices,
            # we may modify for alignments
            "lengths": lengths,
        }

