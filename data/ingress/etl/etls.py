import os
import json
import numpy as np


from cleverspeech.data.ingress.etl import utils
from cleverspeech.utils.Utils import np_arr, lcomp, l_map, log
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

    # N.B. ==> If audios is 0 at any point then the perturbation will always be
    # zero for that sample due to a zero gradient. so add 1 to zero samples make
    # backpropogation work for all samples (1/2**15 is small so side-effects
    # should be minimal).
    for audio in audios:
        audio[audio == 0] = 1.0

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


def create_standard_target_batch(targets_pool, batch_size, trues_batch, audios_batch, tokens=TOKENS):

    target_data = []
    for i in range(batch_size):
        p = utils.BatchGen.pop_target_phrase(
            targets_pool,
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
        lambda x: utils.Targets.get_indices(x, tokens),
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


def create_dense_target_batch_from_standard(data, actual_feats, max_feats):
    target_phrases = data["phrases"]
    target_ids = data["row_ids"]
    orig_indices = data["indices"]
    lengths = data["lengths"]
    tokens = data["tokens"]

    z = zip(orig_indices, lengths)

    new_transcription_indices = l_map(
        lambda x: utils.AlignmentTargets.insert_target_blanks(*x),
        z
    )

    # calculate the actual number of repeats
    z = zip(actual_feats, lengths)

    n_repeats = [
        utils.AlignmentTargets.calculate_densest_repeats(x, y) for x, y in z
    ]

    # do linear expansion only on the existing indices (target phrases
    # are still valid as they are).
    z = zip(
        new_transcription_indices,
        actual_feats,
        l_map(len, new_transcription_indices),
        n_repeats
    )

    new_alignment_indices = [
        l_map(
            lambda x: x,
            utils.AlignmentTargets.create_new_dense_indices(x, y, l, n)
        ) for x, y, l, n in z
    ]

    # do padding for non-ctc loss functions
    z = zip(new_alignment_indices, max_feats)
    padded_alignment_indices = np_arr(
        [
            l_map(
                lambda x: x,
                utils.AlignmentTargets.pad_indices(x, y)
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
    lengths = data["lengths"]
    orig_indices = data["indices"]
    tokens = data["tokens"]

    z = zip(orig_indices, lengths)

    new_transcription_indices = l_map(
        lambda x: utils.AlignmentTargets.insert_target_blanks(*x),
        z
    )

    # do linear expansion only on the existing indices (target phrases
    # are still valid as they are).
    z = zip(new_transcription_indices, actual_feats)

    new_alignment_indices = [
        l_map(
            lambda x: x,
            utils.AlignmentTargets.create_new_sparse_indices(x, y)
        ) for x, y in z
    ]

    # do padding for non-ctc loss functions
    z = zip(new_alignment_indices, max_feats)
    padded_alignment_indices = np_arr(
        [
            l_map(
                lambda x: x,
                utils.AlignmentTargets.pad_indices(x, y)
            ) for x, y in z
        ],
        np.int32
    )

    # update the target sequence lengths
    lengths = l_map(
        len,
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


def create_midish_target_batch_from_standard(data, actual_feats, max_feats):
    target_phrases = data["phrases"]
    target_ids = data["row_ids"]
    orig_indices = data["indices"]
    lengths = data["lengths"]
    tokens = data["tokens"]

    z = zip(orig_indices, lengths)

    new_transcription_indices = l_map(
        lambda x: utils.AlignmentTargets.insert_target_blanks(*x),
        z
    )

    # calculate the actual number of repeats
    z = zip(actual_feats, lengths)

    n_repeats = [
        utils.AlignmentTargets.calculate_midpoint_repeats(x, y) for x, y in z
    ]

    # do linear expansion only on the existing indices (target phrases
    # are still valid as they are).
    z = zip(
        new_transcription_indices,
        actual_feats,
        l_map(len, new_transcription_indices),
        n_repeats
    )

    new_alignment_indices = [
        l_map(
            lambda x: x,
            utils.AlignmentTargets.create_new_dense_indices(x, y, l, n)
        ) for x, y, l, n in z
    ]

    # do padding for non-ctc loss functions
    z = zip(new_alignment_indices, max_feats)
    padded_alignment_indices = np_arr(
        [
            l_map(
                lambda x: x,
                utils.AlignmentTargets.pad_indices(x, y)
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


def create_custom_repeats_target_batch_from_standard(data, actual_feats, max_feats, path_repeat_factor):
    target_phrases = data["phrases"]
    target_ids = data["row_ids"]
    orig_indices = data["indices"]
    lengths = data["lengths"]
    tokens = data["tokens"]

    z = zip(orig_indices, lengths)

    new_transcription_indices = l_map(
        lambda x: utils.AlignmentTargets.insert_target_blanks(*x),
        z
    )

    # calculate the actual number of repeats
    z = zip(actual_feats, lengths)

    n_repeats = [
        utils.AlignmentTargets.calculate_custom_repeats(x, y, r=path_repeat_factor) for x, y in z
    ]
    print(actual_feats, lengths, n_repeats)

    # do linear expansion only on the existing indices (target phrases
    # are still valid as they are).
    z = zip(
        new_transcription_indices,
        actual_feats,
        l_map(len, new_transcription_indices),
        n_repeats
    )

    new_alignment_indices = [
        l_map(
            lambda x: x,
            utils.AlignmentTargets.create_new_dense_indices(x, y, l, n)
        ) for x, y, l, n in z
    ]

    # do padding for non-ctc loss functions
    z = zip(new_alignment_indices, max_feats)
    padded_alignment_indices = np_arr(
        [
            l_map(
                lambda x: x,
                utils.AlignmentTargets.pad_indices(x, y)
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


class NoValidCTCAlignmentException(Exception):
    pass


def create_ctcalign_target_batch_from_standard(data):
    """

    :param data: a full starter batch generated by batch_gen.standard
    :return: a new batch of target data
    """

    target_phrases = data.targets["phrases"]
    target_ids = data.targets["row_ids"]
    orig_indices = data.targets["indices"]
    tokens = data.targets["tokens"]

    log("Searching for high likelihood CTC alignments...", wrap=False)
    results = utils.subprocess_ctcalign_search(data)

    if results == "dead":
        raise NoValidCTCAlignmentException(
            "Could not find any optimal CTC alignments for you..."
        )

    else:
        log(
            "Found CTC alignments, continuing to initialise the attack...",
            wrap=True
        )
        target_alignments = np.asarray(results, dtype=np.int32)

    lengths = l_map(
        lambda x: x.size,
        target_alignments
    )

    return {
        "tokens": tokens,
        "phrases": target_phrases,
        "row_ids": target_ids,
        "indices": target_alignments,
        "original_indices": orig_indices,
        "lengths": lengths,
    }


def create_tightly_packed_target_batch_from_standard(data, actual_feats, max_feats):
    # TODO: Place chars next to each other in an otherwise blank path
    pass


def create_shuffled_target_batch_from_standard(data, actual_feats, max_feats):
    # TODO: Create a randomly shuffled sparse (and/or midpoint/dense)
    pass

