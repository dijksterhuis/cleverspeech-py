from cleverspeech.utils.Utils import l_sort

from cleverspeech.data.ingress.bases import (
    _BaseStandardAudioBatchETL,
    _BaseTrimmedAudioBatchETL,
    _BaseSilenceAudioBatchETL,
    _BaseWhiteNoiseAudioBatchETL,
    _BaseConstantAmplitudeAudioBatchETL,
    _BaseBatchIterator as MCV7IterableBatches,
    Wav32BitSignedFloat,
)

from cleverspeech.data.ingress.mcv_v1 import (
    _BaseFromMCV1Audios, MCV1TranscriptionsFromCSVFile
)


class _BaseFromMCV7Audios(_BaseFromMCV1Audios):
    """
    WAV files are just wav files.
    """
    pass


class MCV7StandardAudioBatchETL(
    _BaseStandardAudioBatchETL, _BaseFromMCV7Audios
):
    pass


class MCV7TrimmedAudioBatchETL(
    _BaseTrimmedAudioBatchETL, _BaseFromMCV7Audios
):
    pass


class MCV7SilenceAudioBatchETL(
    _BaseSilenceAudioBatchETL, _BaseFromMCV7Audios
):
    pass


class MCV7ConstantAmplitudeAudioBatchETL(
    _BaseConstantAmplitudeAudioBatchETL, _BaseFromMCV7Audios
):
    pass


class MCV7WhiteNoiseAudioBatchETL(
    _BaseWhiteNoiseAudioBatchETL, _BaseFromMCV7Audios
):
    pass


class MCV7TranscriptionsFromCSVFile(MCV1TranscriptionsFromCSVFile):

    """
    Just need to change the column index of the transcription field from 1 -> 2
    """

    def _extract(self, csv_file_path, numb):

        with open(csv_file_path, 'r') as f:
            data = f.readlines()

        targets = [
            (row.split(',')[2], idx) for idx, row in enumerate(data) if idx > 0
        ]
        targets = l_sort(lambda x: len(x[0]), targets, reverse=False)

        self.pool = targets[:numb]
