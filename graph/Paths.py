import sys
import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from cleverspeech.utils.Utils import np_arr, np_zero, l_map, log, lcomp


TOKENS = " abcdefghijklmnopqrstuvwxyz'-"


class Path(ABC):

    def __init__(self, batch, updateable=False):

        self.updateable = updateable
        self.batch = batch

        self.target_phrases = batch.targets["phrases"]
        self.target_ids = batch.targets["row_ids"]
        self.orig_indices = batch.targets["indices"]
        self.lengths = batch.targets["lengths"]
        self.tokens = batch.targets["tokens"]
        self.actual_feats = batch.audios["real_feats"]
        self.max_feats = batch.audios["ds_feats"]

    @staticmethod
    def create_modified_transcription(target_indices, length):
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

        # add a blank on the end for symmetry
        with_repeats.append(28)

        return with_repeats

    def pad(self, new_indices):
        padded = np_arr(
            [
                l_map(lambda x: x, self._apply_padding(x, y))
                for x, y in zip(new_indices, self.max_feats)
            ],
            np.int32
        )
        return padded

    @staticmethod
    def _apply_padding(indices, act_len):
        n_paddings = act_len - len(indices)
        padded = np.concatenate([indices, np.ones(n_paddings) * 28])
        return padded

    def create(self):

        modified_transcriptions = l_map(
            lambda x: self.create_modified_transcription(*x),
            zip(self.orig_indices, self.lengths)
        )

        new_indices = self.path_calculation(modified_transcriptions)

        # do padding for non-ctc loss functions
        padded = self.pad(new_indices)

        # update the target sequence lengths
        lengths = l_map(lambda x: x.size, padded)

        logging_alignments = l_map(
            lambda x: "".join([self.tokens[i] for i in x]), padded
        )
        s = "Modified targets batch to use the following paths:\n"
        s += "\n".join(logging_alignments)

        # log(s, wrap=True)

        self.batch.targets = {
            "tokens": self.tokens,
            "phrases": self.target_phrases,
            "row_ids": self.target_ids,
            "indices": padded,
            "original_indices": self.orig_indices,
            "lengths": lengths,
        }

    @abstractmethod
    def path_calculation(self, modified_transcriptions):
        pass

    def mutate(self):
        pass


class _LinearlyDistributed(Path):

    density = None

    def calculate_repeats(self, actual_n_feats, target_phrase_length):
        return (actual_n_feats * self.density) // target_phrase_length

    @staticmethod
    def _create_path_indices(new_target, n_feats, repeats):

        """
        Taking into account the space we have available, find out the new argmax
        indices for each frame of audio which relate to our target phrase

        :param new_target: the new target phrase included additional blank tokens
        :param n_feats: the number of features in the logits (time steps)
        :param length: the actual length of the transcription with blanks inserted
        :param repeats: the number of repeats for each token

        :return: the index for each frame in turn
        """

        spacing = n_feats // len(new_target)

        # guarantee non-minus number of characters
        n_chars = repeats - 1 if repeats > 0 else 0

        for t in new_target:
            for i in range(spacing):
                if i <= n_chars:
                    yield t
                else:
                    yield 28

    def path_calculation(self, modified_transcriptions):

        new_lengths = l_map(len, modified_transcriptions)

        # calculate the actual number of repeats
        n_repeats = l_map(
            lambda x: self.calculate_repeats(*x),
            zip(self.actual_feats, new_lengths)
        )

        # do linear expansion only on the existing indices (target phrases
        # are still valid as they are).
        z = zip(
            modified_transcriptions,
            self.actual_feats,
            n_repeats
        )

        return l_map(
            lambda x: list(self._create_path_indices(*x)), z
        )


class Dense(_LinearlyDistributed):

    density = 1.0


class Mid(_LinearlyDistributed):

    density = 0.5


class Sparse(_LinearlyDistributed):

    density = 0.0


class CustomDensity(_LinearlyDistributed):

    def __init__(self, batch, updateable=False, density=0.5):

        super().__init__(batch, updateable=updateable)

        assert 0 <= density <= 1.0
        self.density = density


class _TranscriptionPatch(Path):

    @staticmethod
    @abstractmethod
    def _calculate_positioning(new_t, act_feat):
        pass

    def path_calculation(self, modified_transcriptions):
        return l_map(
            lambda x: self._calculate_positioning(*x),
            zip(modified_transcriptions, self.actual_feats)
        )


class StartPatch(_TranscriptionPatch):

    @staticmethod
    def _calculate_positioning(new_t, act_feat):
        path = np.ones(act_feat, dtype=np.int32) * 28
        path[:len(new_t)] = new_t
        return path


class EndPatch(_TranscriptionPatch):

    @staticmethod
    def _calculate_positioning(new_t, act_feat):
        path = np.ones(act_feat, dtype=np.int32) * 28
        path[act_feat-len(new_t):act_feat] = new_t
        return path


class MidPatch(_TranscriptionPatch):

    @staticmethod
    def _calculate_positioning(new_t, act_feat):

        feats_mid = act_feat // 2

        trans_start = feats_mid - (len(new_t) // 2)
        trans_end = trans_start + len(new_t)

        path = np.ones(act_feat, dtype=np.int32) * 28
        path[trans_start:trans_end] = new_t
        return path


class CustomShiftPatch(_TranscriptionPatch):

    # TODO: This is very hacky right now, but could save having to deal with
    #  multiple class definitions (start, end, mid etc.)

    def __init__(self, batch, updateable=False, shift=0.5):
        super().__init__(batch, updateable=updateable)

        assert 0 <= shift <= 1
        self.shift = shift

    def _calculate_positioning(self, new_t, act_feat):

        if self.shift == 0:
            feats_centroid = int(len(new_t) // 2)
        else:
            feats_centroid = int(act_feat // (1 / self.shift))

        trans_start = feats_centroid - (len(new_t) // 2)
        trans_end = trans_start + len(new_t)

        if trans_start < 0:
            trans_start, trans_end = 0, len(new_t)

        if trans_end > act_feat:
            trans_start, trans_end = act_feat - len(new_t), act_feat

        path = np.ones(act_feat, dtype=np.int32) * 28
        path[trans_start:trans_end] = new_t
        return path


class RandomMonotonicSparse(Path):

    @staticmethod
    def monotonic_random_sparse_path(t_mod, af):
        total_available_positions = af - len(t_mod)

        path = np.ones(af, dtype=np.int32) * 28

        positions = np.random.choice(
            lcomp(range(total_available_positions)),
            size=len(t_mod),
            replace=False
        )
        positions = sorted(positions)

        for t, p in zip(t_mod, positions):
            path[p] = t

        return path

    def path_calculation(self, modified_transcriptions):
        return l_map(
            lambda x: self.monotonic_random_sparse_path(*x),
            zip(modified_transcriptions, self.actual_feats)
        )


class NoValidCTCAlignmentException(Exception):
    pass


class CTC(Path):

    @staticmethod
    def __ctc_search(batch, use_beam_search_decoder=False):

        g = tf.Graph()
        with tf.Session(graph=g) as sess:

            targets = tf.placeholder(
                tf.int32, [batch.size, None], name='qq_alignment_targets'
            )
            target_lengths = tf.placeholder(
                tf.int32, [batch.size], name='qq_alignment_targets_lengths'
            )

            shape = [
                batch.size, batch.audios["max_feats"],
                len(batch.targets["tokens"])
            ]

            initial_alignments = tf.Variable(
                tf.zeros(shape),
                dtype=tf.float32,
                trainable=True,
                name='qq_alignment'
            )

            mask = tf.Variable(
                tf.ones(shape),
                dtype=tf.float32,
                trainable=False,
                name='qq_alignment_mask'
            )

            logits_alignments = initial_alignments * mask
            raw_alignments = tf.transpose(logits_alignments, [1, 0, 2])
            softmax_alignments = tf.nn.softmax(logits_alignments, axis=-1)
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
                            mask = np.ones([29])
                        else:
                            # shouldn't be optimised
                            mask = np.zeros([29])
                            # mask[28] = 30.0
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
                ]) for i in range(tf_dense[0].shape[0])]

                tf_outputs = [o.rstrip(" ") for o in tf_outputs]

                probs = sess.run(log_probs)
                probs = [prob[0] for prob in probs]
                return tf_outputs, probs

            def tf_greedy_decode(sess, logits, features_lengths, tokens,
                    merge_repeated=True):

                tf_decode, log_probs = tf.nn.ctc_greedy_decoder(
                    logits,
                    features_lengths,
                    merge_repeated=merge_repeated,
                )
                dense = tf.sparse.to_dense(tf_decode[0])
                tf_dense = sess.run([dense])
                tf_outputs = [''.join([
                    tokens[int(x)] for x in tf_dense[0][i]
                ]) for i in range(tf_dense[0].shape[0])]

                tf_outputs = [o.rstrip(" ") for o in tf_outputs]

                neg_sum_logits = sess.run(log_probs)
                neg_sum_logits = [prob[0] for prob in neg_sum_logits]
                return tf_outputs, neg_sum_logits

            variables.append(initial_alignments)

            sess.run(tf.variables_initializer(variables))

            still_have_work = True
            max_iters = 1000
            c = 0

            while still_have_work:

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

                if use_beam_search_decoder is True:
                    decodings, probs = tf_beam_decode(
                        sess, raw_alignments, batch.audios["real_feats"], TOKENS
                    )
                else:
                    decodings, probs = tf_greedy_decode(
                        sess, raw_alignments, batch.audios["real_feats"], TOKENS
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
                    still_have_work = False

                elif c >= max_iters:
                    log("Could not find any CTC optimal alignments for you...")
                    sys.exit(5)
                else:
                    c += 1

            results = sess.run(target_alignments).tolist()

        # remove ops from the graph manually as I don't trust python to garbage
        # collect the sub-graph's ops from the GPU
        for op in g.get_operations():
            g.clear_collection(op.name)

        return results

    def path_calculation(self, modified_transcriptions):
        pass

    # override create() instead of using abstract path_calculation()
    def create(self):
        """

        :param data: a full starter batch generated by batch_gen.standard
        :return: a new batch of target data
        """

        log("Searching for high likelihood CTC alignments...", wrap=False)
        results = self.__ctc_search(self.batch)

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
        self.batch.targets = {
            "tokens": self.tokens,
            "phrases": self.target_phrases,
            "row_ids": self.target_ids,
            "indices": target_alignments,
            "original_indices": self.orig_indices,
            "lengths": lengths,
        }


ALL_PATHS = {
    "mid": Mid,
    "dense": Dense,
    "sparse": Sparse,
    "ctc": CTC,
    "mono-sparse": RandomMonotonicSparse,
    "patch-start": StartPatch,
    "patch-mid": MidPatch,
    "patch-end": EndPatch,
}
