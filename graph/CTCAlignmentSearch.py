import tensorflow as tf
import numpy as np

from cleverspeech.utils.Utils import log, lcomp
from cleverspeech.graph.GraphConstructor import Constructor


def create_tf_ctc_alignment_search_graph(attack, batch, feeds):
    alignment_graph = Constructor(attack.sess, batch, feeds)
    alignment_graph.add_graph(Graph, attack)
    alignment_graph.add_loss(Loss)
    alignment_graph.create_loss_fn()
    alignment_graph.add_optimiser(Procedure)
    return alignment_graph


class Loss(object):
    def __init__(self, alignment_graph):
        seq_lens = alignment_graph.batch.audios["real_feats"]

        self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            alignment_graph.graph.targets,
            alignment_graph.graph.target_lengths
        )

        self.loss_fn = tf.nn.ctc_loss(
            labels=self.ctc_target,
            inputs=alignment_graph.graph.raw_alignments,
            sequence_length=seq_lens,
        )


class Graph:
    def __init__(self, sess, batch, attack_graph):
        batched_alignment_shape = attack_graph.victim.logits.shape.as_list()

        self.initial_alignments = tf.Variable(
            tf.zeros(batched_alignment_shape),
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

        self.mask = tf.Variable(
            tf.ones(batched_alignment_shape),
            dtype=tf.float32,
            trainable=False,
            name='qq_alignment_mask'
        )

        self.logits_alignments = self.initial_alignments + self.mask
        self.raw_alignments = tf.transpose(self.logits_alignments, [1, 0, 2])
        self.softmax_alignments = tf.nn.softmax(self.logits_alignments)
        self.target_alignments = tf.argmax(self.softmax_alignments, axis=2)

        # TODO - this should be loaded from feeds later on
        self.targets = attack_graph.graph.placeholders.targets
        self.target_lengths = attack_graph.graph.placeholders.target_lengths

        per_logit_lengths = batch.audios["real_feats"]
        maxlen = batched_alignment_shape[1]

        initial_masks = np.asarray(
            [m for m in self.gen_mask(per_logit_lengths, maxlen)],
            dtype=np.float32
        )

        sess.run(self.mask.assign(initial_masks))

    @staticmethod
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


class Procedure:
    def __init__(self, graph):

        self.graph = graph
        self.loss = self.graph.adversarial_loss

        self.train_alignment = None
        self.variables = None

    def create_optimiser(self):

        optimizer = tf.train.AdamOptimizer(1)

        grad_var = optimizer.compute_gradients(
            self.graph.loss_fn,
            self.graph.graph.initial_alignments
        )
        assert None not in lcomp(grad_var, i=0)

        self.train_alignment = optimizer.apply_gradients(grad_var)
        self.variables = optimizer.variables()

    def optimise(self, batch, victim):

        g, v, b = self.graph, victim, batch

        logits = v.get_logits(v.raw_logits, g.feeds.examples)
        assert logits.shape == g.graph.raw_alignments.shape

        while True:

            train_ops = [
                self.graph.loss_fn,
                g.graph.softmax_alignments,
                g.graph.logits_alignments,
                g.graph.mask,
                self.train_alignment
            ]

            ctc_limit, softmax, raw, m, _ = g.sess.run(
                train_ops,
                feed_dict=g.feeds.alignments
            )

            decodings, probs = victim.inference(
                b,
                logits=softmax,
                decoder="batch",
                top_five=False
            )

            target_phrases = b.targets["phrases"]

            if all([d == target_phrases[0] for d in decodings]) and all(c < 0.1 for c in ctc_limit):
                s = "Found an alignment for each example:"
                for d, p, t in zip(decodings, probs, target_phrases):
                    s += "\nTarget: {t} | Decoding: {d} | Probs: {p:.3f}".format(
                        t=t,
                        d=d,
                        p=p,
                    )
                log(s, wrap=True)
                break
