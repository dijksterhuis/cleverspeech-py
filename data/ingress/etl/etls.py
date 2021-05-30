import os
import json
import numpy as np

from cleverspeech.data.ingress.etl import utils
from cleverspeech.utils.Utils import np_arr, lcomp, l_map
from cleverspeech.utils import WavFile


TOKENS = " abcdefghijklmnopqrstuvwxyz'-"


def get_audio_file_path_pool(indir, numb_examples, file_size_sort=True, filter_term=None, max_file_size=None, min_file_size=None):

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

    # bigger examples require more gpu memory and potentially smaller batches
    if max_file_size:
        fps = list(
            filter(lambda x: x[0] <= max_file_size, fps)
        )

    if min_file_size:
        fps = list(
            filter(lambda x: x[0] >= min_file_size, fps)
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


def create_audio_batch_from_wav_files(batched_file_path_data, dtype="int16"):

    audio_fps = l_map(lambda x: x[1], batched_file_path_data)
    basenames = l_map(lambda x: x[2], batched_file_path_data)

    audios = lcomp([WavFile.load(f, dtype) for f in audio_fps])

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
        lambda t: utils.Targets.get_indices(t, tokens),
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


def create_standard_target_batch(data, tokens=TOKENS):
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
        "original_indices": indices,  # we may modify for alignments
        "lengths": lengths,
    }


def create_dense_target_batch_from_standard(data, actual_feats, max_feats):
    target_phrases = data["phrases"]
    target_ids = data["row_ids"]
    orig_indices = data["indices"]
    lengths = data["lengths"]
    tokens = data["tokens"]

    new_transcription_indices = np_arr(
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
    z = zip(new_transcription_indices, actual_feats)

    new_alignment_indices = [
        l_map(
            lambda x: x,
            utils.DenseTargets.create_new_target_indices(x, y, n_repeats)
        ) for x, y in z
    ]

    # do padding for non-ctc loss functions
    z = zip(new_alignment_indices, max_feats)
    padded_alignment_indices = np_arr(
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
        padded_alignment_indices
    )

    return {
        "tokens": tokens,
        "phrases": target_phrases,
        "row_ids": target_ids,
        "indices": padded_alignment_indices,
        "original_indices": orig_indices,
        "lengths": lengths,
    }


def create_sparse_target_batch_from_standard(data, actual_feats, max_feats):
    target_phrases = data["phrases"]
    target_ids = data["row_ids"]
    orig_indices = data["indices"]
    tokens = data["tokens"]

    new_transcription_indices = np_arr(
        l_map(
            lambda x: utils.SparseTargets.insert_target_blanks(x),
            orig_indices
        ),
        np.int32
    )

    # do linear expansion only on the existing indices (target phrases
    # are still valid as they are).
    z = zip(new_transcription_indices, actual_feats)

    new_alignment_indices = [
        l_map(
            lambda x: x,
            utils.SparseTargets.create_new_target_indices(x, y)
        ) for x, y in z
    ]

    # do padding for non-ctc loss functions
    z = zip(new_alignment_indices, max_feats)
    padded_alignment_indices = np_arr(
        [
            l_map(
                lambda x: x,
                utils.SparseTargets.pad_indices(x, y)
            ) for x, y in z
        ],
        np.int32
    )

    # update the target sequence lengths
    lengths = l_map(
        lambda x: x.size,
        padded_alignment_indices
    )

    return {
        "tokens": tokens,
        "phrases": target_phrases,
        "row_ids": target_ids,
        "indices": padded_alignment_indices,
        "original_indices": orig_indices,
        "lengths": lengths,
    }

