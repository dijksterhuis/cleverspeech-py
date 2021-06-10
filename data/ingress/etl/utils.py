import os
import random
import numpy as np
from cleverspeech.utils.Utils import np_arr, np_zero, l_map


class Files(object):

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


class Audios(object):

    @staticmethod
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
            return Audios.padding(max_len + ((extra // 512) * 512))
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


class Targets(object):

    @staticmethod
    def get_indices(phrase, tokens):
        """
        Generate the target indices for CTC and alignments

        :return: array of target indices from tokens [phrase length]
        """
        indices = np_arr([tokens.index(i) for i in phrase], np.int32)
        return indices


class AlignmentTargets(object):

    @staticmethod
    def calculate_densest_repeats(actual_n_feats, target_phrase_length):
        return actual_n_feats // target_phrase_length

    @staticmethod
    def calculate_midpoint_repeats(actual_n_feats, target_phrase_length):
        return ((actual_n_feats / 2) // target_phrase_length) - 1

    @staticmethod
    def insert_target_blanks(target_indices, length):
        # get a shifted list so we can compare back one step in the phrase

        previous_indices = target_indices.tolist()
        previous_indices.insert(0, None)

        # insert blank tokens where ctc would expect them - i.e. `do-or`
        # also insert a blank at the start to gives time for the RNN hidden
        # states to "warm up"
        with_repeats = [28]

        z = zip(target_indices, previous_indices)

        for idx, (current, previous) in enumerate(z):
            if idx >= length:
                break
            if not previous:
                with_repeats.append(current)
            elif current == previous:
                with_repeats.append(28)
                with_repeats.append(current)
            else:
                with_repeats.append(current)
        return with_repeats

    @staticmethod
    def create_new_dense_indices(new_target, n_feats, length, repeats):

        """
        Taking into account the space we have available, find out the new argmax
        indices for each frame of audio which relate to our target phrase

        :param new_target: the new target phrase included additional blank tokens
        :param n_feats: the number of features in the logits (time steps)
        :param length: the actual length of the transcription with blanks inserted
        :param repeats: the number of repeats for each token

        :return: the index for each frame in turn
        """

        spacing = n_feats // length

        for t in new_target:
            for i in range(spacing):
                if i > repeats:
                    yield 28
                else:
                    yield t

    @staticmethod
    def create_new_sparse_indices(new_target, n_feats):

        """
        Taking into account the space we have available, find out the new argmax
        indices for each frame of audio which relate to our target phrase

        :param new_target: the new target phrase included additional blank tokens
        :param n_feats: the number of features in the logits (time steps)

        :return: the index for each frame in turn
        """

        spacing = n_feats // len(new_target)

        for t in new_target:
            for i in range(spacing):
                if i > 0:
                    yield 28
                else:
                    yield t

    @staticmethod
    def pad_indices(indices, act_len):
        n_paddings = act_len - len(indices)
        padded = np.concatenate([indices, np.ones(n_paddings) * 28])
        return padded


class BatchGen(object):

    @staticmethod
    def popper(data, size):
        return l_map(
            lambda x: data.pop(x-1), range(size, 0, -1)
        )

    @staticmethod
    def pop_target_phrase(all_targets, true_targets, min_feats, idx=0):
        candidate_target = random.choice(all_targets)

        length_test = len(candidate_target[0]) > min_feats
        matches_true_test = candidate_target[0] in true_targets

        if length_test or matches_true_test:
            return BatchGen.pop_target_phrase(
                all_targets, true_targets, min_feats, idx=idx + 1
            )
        else:
            return candidate_target
