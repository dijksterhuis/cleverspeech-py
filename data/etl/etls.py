import os
import numpy as np

from cleverspeech.data.etl import utils
from cleverspeech.utils.Utils import np_arr, lcomp, load_wavs, l_map


def get_audio_file_path_pool(indir, numb_examples, file_size_sort=True, filter_term=None, max_samples=None, min_samples=None):

    if not os.path.exists(indir):
        raise Exception("Path does not exist: {}".format(indir))

    fps = [x for x in utils.Files.get_file_paths(indir)]

    if file_size_sort is not None:
        if file_size_sort == 'desc':
            fps = utils.Files.get_size_sorted_file_paths(fps, reverse=True)
        elif file_size_sort == 'asc':
            fps = utils.Files.get_size_sorted_file_paths(fps, reverse=False)
        else:
            # otherwise we'll sort anyway as it can affect optimisation
            fps = utils.Files.get_size_sorted_file_paths(fps, reverse=True)

    if filter_term:
        fps = list(
            filter(lambda x: filter_term in x[1], fps)
        )

    # bigger examples require more gpu memory / smaller batches
    if max_samples:
        fps = list(
            filter(lambda x: x[0] <= max_samples, fps)
        )

    if min_samples:
        fps = list(
            filter(lambda x: x[0] >= min_samples, fps)
        )

    file_path_data = fps[:numb_examples]
    return file_path_data


def get_target_phrase_pool(indir, numb):
    with open(indir, 'r') as f:
        data = f.readlines()

    targets = [
        (row.split(',')[1], idx) for idx, row in enumerate(data) if idx > 0
    ]
    targets.sort(key=lambda x: len(x[0]), reverse=False)

    return targets[:numb]


def create_audio_batch(batched_file_path_data, dtype="int16"):

    audio_fps = l_map(lambda x: x[1], batched_file_path_data)
    basenames = l_map(lambda x: x[2], batched_file_path_data)

    audios = lcomp(load_wavs(audio_fps, dtype=dtype))

    maxlen = max(map(len, audios))
    maximum_length = maxlen + utils.Audios.padding(maxlen)

    padded_audio = np_arr(
        lcomp(utils.Audios.gen_padded_audio(audios, maximum_length)),
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
        "max_samples": maximum_length,
        "max_feats": maximum_feature_lengths[0],
        "audio": audios,
        "padded_audio": padded_audio,
        "basenames": basenames,
        "n_samples": actual_lengths,
        "ds_feats": maximum_feature_lengths,
        "real_feats": actual_feature_lengths,
    }


def create_standard_target_batch(data, tokens=" abcdefghijklmnopqrstuvwxyz'-"):
    target_phrases = list(map(lambda x: x[0], data))
    target_ids = list(map(lambda x: x[1], data))

    indices = np_arr(
        l_map(lambda x: utils.Targets.get_indices(x, tokens), target_phrases),
        np.int32
    )
    lengths = np_arr(
        l_map(lambda x: len(x), target_phrases),
        np.int32
    )

    return {
        "tokens": tokens,
        "phrases": target_phrases,
        "row_ids": target_ids,
        "indices": indices,
        "lengths": lengths,
    }


def create_dense_target_batch_from_standard(data, actual_feats, max_feats):
    target_phrases = data["phrases"]
    target_ids = data["row_ids"]
    orig_indices = data["indices"]
    lengths = data["lengths"]
    tokens = data["tokens"]

    new_indices = np_arr(
        l_map(
            lambda x: utils.DenseTargets.insert_target_blanks(x),
            orig_indices
        ),
        np.int32
    )

    # calculate the actual number of repeats
    z = zip(actual_feats, lengths)
    n_repeats = [
        utils.DenseTargets.calculate_possible_repeats(x, y) for x, y in z
    ]
    # do linear expansion only on the existing indices (target phrases
    # are still valid as they are).
    z = zip(new_indices, actual_feats, n_repeats)

    new_indices = np_arr(
        [
            l_map(
                lambda x: x,
                utils.DenseTargets.create_new_target_indices(x, y, z, )
            ) for x, y, z in z
        ],
        np.int32
    )

    # do padding for nopn-ctc loss
    z = zip(new_indices, max_feats)
    new_indices = np_arr(
        [
            l_map(
                lambda x: x,
                utils.DenseTargets.pad_indices(x, y)
            ) for x, y in z
        ],
        np.int32
    )

    # update the target sequence lengths
    lengths = l_map(
        lambda x: x.size,
        new_indices
    )

    return {
        "tokens": tokens,
        "phrases": target_phrases,
        "row_ids": target_ids,
        "indices": new_indices,
        "original_indices": orig_indices,
        "lengths": lengths,
    }


