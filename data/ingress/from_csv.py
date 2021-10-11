import os
import pandas as pd

from cleverspeech.utils.Utils import (
    l_map
)
from cleverspeech.data.ingress.bases import (
    _BaseAudiosBatchETL,
    _BaseStandardAudioBatchETL,
    _BaseTrimmedAudioBatchETL,
    _BaseSilenceAudioBatchETL,
    _BaseTranscriptionsBatchETL,
    _BaseBatchIterator as CSVIterableBatches,
)


class _BaseFromCSVAudios(_BaseAudiosBatchETL):

    def _extract(self, csv_file_path, file_size_sort=None, filter_term=None):
        if not os.path.exists(csv_file_path):
            raise Exception("Path does not exist: {}".format(csv_file_path))

        df = pd.read_csv(csv_file_path)
        cols = df.columns.tolist()

        assert "file_path" in cols
        assert "ground_truth_file_path" in cols
        assert "run" in cols

        if filter_term:
            df = df.where(df["run"] == filter_term).dropna()

        fps = df["file_path"].dropna().tolist()
        gts = df["ground_truth_file_path"].dropna().tolist()

        self.pool = l_map(
            lambda x: x, zip(fps, gts)
        )

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
        return l_map(lambda x: x[0], batch_data)

    @staticmethod
    def get_batch_wav_basenames(batch_data):
        return l_map(lambda x: os.path.basename(x), batch_data)

    @staticmethod
    def get_batch_ground_truth_file_paths(batch_data):
        return l_map(lambda x: x[1], batch_data)

    @staticmethod
    def get_batch_ground_truth_phrase_lengths(phrase):
        return l_map(len, phrase)


class FromCSVFileStandardAudioBatchETL(
    _BaseStandardAudioBatchETL, _BaseFromCSVAudios
):
    pass


class FromCSVFileTrimmedAudioBatchETL(
    _BaseTrimmedAudioBatchETL, _BaseFromCSVAudios
):
    pass


class FromCSVFileSilenceAudioBatchETL(
    _BaseSilenceAudioBatchETL, _BaseFromCSVAudios
):
    pass


class TranscriptionsFromCSVFile(_BaseTranscriptionsBatchETL):
    def _extract(self, csv_file_path):
        if not os.path.exists(csv_file_path):
            raise Exception("Path does not exist: {}".format(csv_file_path))

        df = pd.read_csv(csv_file_path)

        self.pool = df
        self.columns = self.pool.columns.tolist()
        self.csv_file_path = csv_file_path

        assert "file_path" in self.columns
        assert "phrase" in self.columns
        assert "row_id" in self.columns

    def _popper(self, batch_size, **kwargs):
        selections = self.pool.tail(batch_size)
        self.pool = self.pool.drop(self.pool.tail(batch_size).index)
        return selections

    def create_batch(self, batch_size, audios_batch):
        selections = self._popper(batch_size)
        return self._transform_and_load(selections)

    @staticmethod
    def _selection_rules(phrase, n_feats, ground_truth, selections):
        upper_bound = len(phrase) <= n_feats // 4
        lower_bound = len(phrase) >= 4
        not_ground_truth = phrase != ground_truth
        not_selected = phrase not in selections

        return upper_bound and lower_bound and not_ground_truth and not_selected

    def get_single_transcription_entry(self, batch_df, audio_file_path, key="phrase"):
        df = batch_df.where(self.pool["file_path"] == audio_file_path)
        return df.dropna()[key].tolist()[0]

    def get_batch_transcriptions_text(self, ts):
        # FIXME
        batch_df, audio_file_paths = ts
        return l_map(
            lambda x: self.get_single_transcription_entry(
                batch_df, x, "phrase"
            ),
            audio_file_paths
        )

    def get_batch_transcriptions_row_ids(self, ts):
        # FIXME
        batch_df, audio_file_paths = ts
        return l_map(
            lambda x: self.get_single_transcription_entry(batch_df, x, "row_id"),
            audio_file_paths
        )

    def _transform_and_load(self, batch_df, audio_file_paths):

        phrases = self.get_batch_transcriptions_text(
            (batch_df, audio_file_paths)
        )
        lengths = self.get_batch_transcription_lengths(phrases)
        max_len = self.get_batch_maximum_transcription_length(phrases)
        ragged_indices = self.get_batch_transcription_as_indices(phrases)
        padded_indices = self.pad_batch_of_transcription_indices(
            ragged_indices, max_len
        )

        row_ids = self.get_batch_transcriptions_unique_id(
            (batch_df, audio_file_paths)
        )

        return {
            "tokens": self.tokens,
            "phrases": phrases,
            "row_ids": row_ids,
            "indices": padded_indices,
            "original_indices": ragged_indices,
            # we may modify for alignments
            "lengths": lengths,
        }
