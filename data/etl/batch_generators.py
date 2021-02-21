from cleverspeech.data.etl import etls
from cleverspeech.data.etl.utils import BatchGen
from cleverspeech.utils.Utils import log, l_map


class Batch:
    def __init__(self, size, audios, targets):

        self.size = size
        self.audios = audios
        self.targets = targets


def get_standard_batch_generator(settings):

    # get N samples of all the data. alsp make sure to limit example length,
    # otherwise we'd have to do adaptive batch sizes.

    audio_file_paths_pool = etls.get_audio_file_path_pool(
        settings["audio_indir"],
        settings["max_examples"],
        filter_term=".wav",
        max_samples=settings["max_audio_length"]
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
        audios_batch = etls.create_audio_batch(audio_batch_data)

        #  we need to make sure target phrase length < n audio feats.
        # also, the first batch should also have the longest target phrase
        # and longest audio examples so we can easily manage GPU Memory
        # resources with the AttackSpawner context manager.

        target_phrase = BatchGen.pop_target_phrase(
            targets_pool, min(audios_batch["real_feats"])
        )

        # each target must be the same length else numpy throws a hissyfit
        # because it can't understand skewed matrices
        target_batch_data = l_map(
            lambda _: target_phrase, range(batch_size)
        )

        # actually load the n phrases as a batch of target data
        targets_batch = etls.create_standard_target_batch(target_batch_data)

        batch = Batch(
            batch_size,
            audios_batch,
            targets_batch,
        )

        id = idx

        yield id, batch


def get_dense_batch_factory(settings):

    # get N samples of all the data. alsp make sure to limit example length,
    # otherwise we'd have to do adaptive batch sizes.

    audio_file_paths_pool = etls.get_audio_file_path_pool(
        settings["audio_indir"],
        settings["max_examples"],
        filter_term=".wav",
        max_samples=settings["max_audio_length"]
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
        audios_batch = etls.create_audio_batch(audio_batch_data)

        #  we need to make sure target phrase length < n audio feats.
        # also, the first batch should also have the longest target phrase
        # and longest audio examples so we can easily manage GPU Memory
        # resources with the AttackSpawner context manager.

        # ensure we get *at least* duplicate characters by dividing the
        # minimum possible length by 3

        target_phrase = BatchGen.pop_target_phrase(
            targets_pool, min(audios_batch["real_feats"]) // 3
        )

        # each target must be the same length else numpy throws a hissyfit
        # because it can't understand skewed matrices
        target_batch_data = l_map(
            lambda _: target_phrase, range(batch_size)
        )

        # actually load the n phrases as a batch of target data
        targets_batch = etls.create_standard_target_batch(
            target_batch_data
        )
        targets_batch = etls.create_dense_target_batch_from_standard(
            targets_batch, audios_batch["real_feats"], audios_batch["ds_feats"],
        )

        batch = Batch(
            batch_size,
            audios_batch,
            targets_batch,
        )

        id = idx

        yield id, batch


def get_validation_batch_generator(settings):

    # get N samples of all the data. alsp make sure to limit example length,
    # otherwise we'd have to do adaptive batch sizes.

    audio_file_paths_pool = etls.get_audio_file_path_pool(
        settings["audio_indir"],
        settings["max_examples"],
        filter_term=".wav",
        max_samples=settings["max_audio_length"]
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
        audios_batch = etls.create_audio_batch(audio_batch_data)

        #  we need to make sure target phrase length < n audio feats.
        # also, the first batch should also have the longest target phrase
        # and longest audio examples so we can easily manage GPU Memory
        # resources with the AttackSpawner context manager.

        target_phrase = BatchGen.pop_target_phrase(
            targets_pool, min(audios_batch["real_feats"])
        )

        # each target must be the same length else numpy throws a hissyfit
        # because it can't understand skewed matrices
        target_batch_data = l_map(
            lambda _: target_phrase, range(batch_size)
        )

        # actually load the n phrases as a batch of target data
        targets_batch = etls.create_standard_target_batch(target_batch_data)

        batch = Batch(
            batch_size,
            audios_batch,
            targets_batch,
        )

        id = idx

        yield id, batch

