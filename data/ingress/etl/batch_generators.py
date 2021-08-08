"""
A batch generator creates a batch (funnily enough) with generators (obviously).

This means we can run multiple attacks at once by starting new server
subprocesses for each attack.

--------------------------------------------------------------------------------
"""

from cleverspeech.data.ingress.etl import etls
from cleverspeech.data.ingress.etl.utils import BatchGen
from cleverspeech.utils.Utils import log


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


def standard(settings):

    # get N samples of all the data. alsp make sure to limit example length,
    # otherwise we'd have to do adaptive batch sizes.

    audio_file_paths_pool = etls.get_audio_file_path_pool(
        settings["audio_indir"],
        settings["max_examples"],
        filter_term=".wav",
        max_file_size=settings["max_audio_file_bytes"]
    )

    targets_pool = etls.get_target_phrase_pool(
        settings["targets_path"], settings["max_targets"],
    )

    # Generate the batches in turn, rather than all in one go ...
    # ... To save resources by only running the final ETLs on a batch of data

    total_numb_examples = len(audio_file_paths_pool)
    batch_size = settings["batch_size"]

    log(
        "New Run",
        "Number of test examples: {}".format(total_numb_examples),
        ''.join(
            ["{k}: {v}\n".format(k=k, v=v) for k, v in settings.items()]),
    )

    for idx in range(0, total_numb_examples, batch_size):

        # Handle remainders: number of examples // desired batch size != 0
        if len(audio_file_paths_pool) < settings["batch_size"]:
            batch_size = len(audio_file_paths_pool)

        # get n files paths and create the audio batch data
        audio_batch_data = BatchGen.popper(audio_file_paths_pool, batch_size)
        audios_batch = etls.create_audio_batch_from_wav_files(audio_batch_data)

        # load the correct transcription for the audio files
        trues_batch = etls.create_true_batch(audios_batch)

        targets_batch = etls.create_standard_target_batch(
            targets_pool,
            batch_size,
            trues_batch,
            audios_batch,
        )

        batch = Batch(
            batch_size,
            audios_batch,
            targets_batch,
            trues_batch,
        )

        yield idx, batch


def dense(settings):

    batch_size = settings["batch_size"]

    for idx, batch in standard(settings):

        targets_batch = etls.create_dense_target_batch_from_standard(
            batch.targets, batch.audios["real_feats"], batch.audios["ds_feats"],
        )

        batch = Batch(
            batch_size,
            batch.audios,
            targets_batch,
            batch.trues
        )

        yield idx, batch


def sparse(settings):

    batch_size = settings["batch_size"]

    for idx, batch in standard(settings):

        targets_batch = etls.create_sparse_target_batch_from_standard(
            batch.targets, batch.audios["real_feats"], batch.audios["ds_feats"],
        )

        batch = Batch(
            batch_size,
            batch.audios,
            targets_batch,
            batch.trues
        )

        yield idx, batch


def custom_repeats(settings):

    batch_size = settings["batch_size"]

    for idx, batch in standard(settings):

        targets_batch = etls.create_custom_repeats_target_batch_from_standard(
            batch.targets,
            batch.audios["real_feats"],
            batch.audios["ds_feats"],
            settings["align_repeat_factor"]
        )

        batch = Batch(
            batch_size,
            batch.audios,
            targets_batch,
            batch.trues
        )

        yield idx, batch


def midish(settings):

    batch_size = settings["batch_size"]

    for idx, batch in standard(settings):

        targets_batch = etls.create_midish_target_batch_from_standard(
            batch.targets, batch.audios["real_feats"], batch.audios["ds_feats"],
        )

        batch = Batch(
            batch_size,
            batch.audios,
            targets_batch,
            batch.trues
        )

        yield idx, batch


def ctcalign(settings):

    batch_size = settings["batch_size"]

    for idx, batch in standard(settings):

        targets_batch = etls.create_ctcalign_target_batch_from_standard(
            batch
        )

        batch = Batch(
            batch_size,
            batch.audios,
            targets_batch,
            batch.trues
        )

        yield idx, batch


def monotonic_random_sparse(settings):

    batch_size = settings["batch_size"]

    for idx, batch in standard(settings):

        targets_batch = etls.create_monotonic_random_sparse_from_standard(
            batch.targets, batch.audios["real_feats"], batch.audios["ds_feats"],
        )

        batch = Batch(
            batch_size,
            batch.audios,
            targets_batch,
            batch.trues
        )

        yield idx, batch


def transcription_patch_start(settings):

    batch_size = settings["batch_size"]

    for idx, batch in standard(settings):

        targets_batch = etls.create_transcription_path_at_beginning_from_standard(
            batch.targets, batch.audios["real_feats"], batch.audios["ds_feats"],
        )

        batch = Batch(
            batch_size,
            batch.audios,
            targets_batch,
            batch.trues
        )

        yield idx, batch


def transcription_patch_mid(settings):

    batch_size = settings["batch_size"]

    for idx, batch in standard(settings):

        targets_batch = etls.create_transcription_path_at_midpoint_from_standard(
            batch.targets, batch.audios["real_feats"], batch.audios["ds_feats"],
        )

        batch = Batch(
            batch_size,
            batch.audios,
            targets_batch,
            batch.trues
        )

        yield idx, batch


def transcription_patch_end(settings):

    batch_size = settings["batch_size"]

    for idx, batch in standard(settings):

        targets_batch = etls.create_transcription_path_at_end_from_standard(
            batch.targets, batch.audios["real_feats"], batch.audios["ds_feats"],
        )

        batch = Batch(
            batch_size,
            batch.audios,
            targets_batch,
            batch.trues
        )

        yield idx, batch


ALL_GENERATORS = {
    "standard": standard,
    "mid": midish,
    "dense": dense,
    "sparse": sparse,
    "ctc": ctcalign,
    "mono-sparse": monotonic_random_sparse,
    "patch-start": transcription_patch_start,
    "patch-mid": transcription_patch_mid,
    "patch-end": transcription_patch_end,
}

PATH_GENERATORS = {
    "mid": midish,
    "dense": dense,
    "sparse": sparse,
    "ctc": ctcalign,
    "mono-sparse": monotonic_random_sparse,
    "patch-start": transcription_patch_start,
    "patch-mid": transcription_patch_mid,
    "patch-end": transcription_patch_end,
}
