import os
import random
import numpy as np
import tensorflow as tf

from cleverspeech.utils.Utils import np_arr, np_zero, l_map, log, lcomp


TOKENS = " abcdefghijklmnopqrstuvwxyz'-"


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
        return (actual_n_feats / 3) // target_phrase_length

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


def subprocess_ctcalign_search(batch):
    import multiprocessing as mp
    q = mp.Queue()

    p = mp.Process(
        target=create_tf_ctc_alignment_search_graph,
        args=(batch, q)
    )
    p.start()
    p.join()
    p.terminate()

    target_aligns = q.get()

    return target_aligns


def create_tf_ctc_alignment_search_graph(batch, q):
    with tf.Session() as sess:

        targets = tf.placeholder(
            tf.int32, [batch.size, None], name='qq_alignment_targets'
        )
        target_lengths = tf.placeholder(
            tf.int32, [batch.size], name='qq_alignment_targets_lengths'
        )

        shape = [
            batch.size, batch.audios["max_feats"], len(batch.targets["tokens"])
        ]

        initial_alignments = tf.Variable(
            tf.zeros(shape),
            dtype=tf.float32,
            trainable=True,
            name='qq_alignment'
        )

        # mask is *added* to force decoder to see the logits for those frames as
        # repeat characters. CTC-Loss outputs zero valued vectors for those
        # character classes (as they're beyond the actual alignment length)
        # This messes with decoder output.

        # --> N.B. This is legacy problem with tensorflow/numpy not being able
        # to handle ragged inputs for tf.Variables etc.

        mask = tf.Variable(
            tf.ones(shape),
            dtype=tf.float32,
            trainable=False,
            name='qq_alignment_mask'
        )

        logits_alignments = initial_alignments + mask
        raw_alignments = tf.transpose(logits_alignments, [1, 0, 2])
        softmax_alignments = tf.nn.softmax(logits_alignments)
        target_alignments = tf.argmax(softmax_alignments, axis=2)

        per_logit_lengths = batch.audios["real_feats"]
        maxlen = shape[1]

        def gen_mask(per_logit_len, maxlen):
            # per actual frame
            for l in per_logit_len:
                # per possible frame
                masks = []
                for f in range(maxlen):
                    if l > f:
                        # if should be optimised
                        mask = np.zeros([29])
                    else:
                        # shouldn't be optimised
                        mask = np.zeros([29])
                        mask[28] = 30.0
                    masks.append(mask)
                yield np.asarray(masks)

        initial_masks = np.asarray(
            [m for m in gen_mask(per_logit_lengths, maxlen)],
            dtype=np.float32
        )

        sess.run(mask.assign(initial_masks))

        seq_lens = batch.audios["real_feats"]

        ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            targets,
            target_lengths
        )

        loss_fn = tf.nn.ctc_loss(
            labels=ctc_target,
            inputs=raw_alignments,
            sequence_length=seq_lens,
        )

        optimizer = tf.train.AdamOptimizer(1)

        grad_var = optimizer.compute_gradients(
            loss_fn,
            initial_alignments
        )
        assert None not in lcomp(grad_var, i=0)

        train_alignment = optimizer.apply_gradients(grad_var)
        variables = optimizer.variables()

        def tf_beam_decode(sess, logits, features_lengths, tokens):

            tf_decode, log_probs = tf.nn.ctc_beam_search_decoder(
                logits,
                features_lengths,
                merge_repeated=False,
                beam_width=500
            )
            dense = tf.sparse.to_dense(tf_decode[0])
            tf_dense = sess.run([dense])
            tf_outputs = [''.join([
                tokens[int(x)] for x in tf_dense[0][i]
            ]).rstrip() for i in range(tf_dense[0].shape[0])]

            return tf_outputs

        while True:

            train_ops = [
                loss_fn,
                softmax_alignments,
                logits_alignments,
                mask,
                train_alignment
            ]

            feed = {
                targets: batch.targets["indices"],
                target_lengths: batch.targets["lengths"],
            }

            ctc_limit, softmax, raw, m, _ = sess.run(
                train_ops,
                feed_dict=feed
            )

            decodings, probs = tf_beam_decode(
                sess, softmax, batch.audios["n_feats"], TOKENS
            )

            target_phrases = batch.targets["phrases"]

            decoding_check = all(
                [d == t for d, t in zip(decodings, target_phrases)]
            )
            ctc_check = all(
                c < 0.1 for c in ctc_limit
            )

            if decoding_check and ctc_check:
                s = "Found an alignment for each example:"
                for d, p, t in zip(decodings, probs, target_phrases):
                    s += "\nTarget: {t} | Decoding: {d} | Probs: {p:.3f}".format(
                        t=t,
                        d=d,
                        p=p,
                    )
                log(s, wrap=True)
                break

        q.put(sess.run(target_alignments).tolist())
