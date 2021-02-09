import tensorflow as tf
import numpy as np


class BaseLoss:
    def __init__(self, sess, batch_size, weight_initial=1.0, weight_increment=1.0):

        self.weights = tf.Variable(
            tf.ones(batch_size, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        initial_vals = weight_initial * np.ones([batch_size], dtype=np.float32)
        sess.run(self.weights.assign(initial_vals))

        self.increment = float(weight_increment)

    def update(self, sess, idx):
        weights = sess.run(self.weights)
        weights[idx] += self.increment
        sess.run(self.weights.assign(weights))


class CarliniL2Loss(BaseLoss):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack_graph):

        super().__init__(
            attack_graph.sess,
            attack_graph.batch.size,
            weight_initial=1.0,
            weight_increment=1.0
        )

        # N.B. original code did `reduce_mean` on `(advex - original) ** 2`...
        # `tf.reduce_mean` on `deltas` is exactly the same with fewer variables

        l2delta = tf.reduce_mean(attack_graph.bounded_deltas ** 2, axis=1)
        self.loss_fn = l2delta / self.weights


class CTCLoss(BaseLoss):
    """
    Simple adversarial CTC Loss from https://arxiv.org/abs/1801.01944

    N.B. This loss does *not* conform to l(x + d, t) <= 0 <==> C(x + d) = t
    """
    def __init__(self, attack_graph):

        super().__init__(
            attack_graph.sess,
            attack_graph.batch.size,
            weight_initial=1.0,
            weight_increment=1.0
        )

        self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            attack_graph.graph.placeholders.targets,
            attack_graph.graph.placeholders.target_lengths
        )

        self.loss_fn = tf.nn.ctc_loss(
            labels=tf.cast(self.ctc_target, tf.int32),
            inputs=attack_graph.victim.raw_logits,
            sequence_length=attack_graph.batch.audios.feature_lengths
        ) * self.weights


class EntropyLoss(BaseLoss):
    def __init__(self, attack_graph):

        super().__init__(
            attack_graph.sess,
            attack_graph.batch.size,
            weight_initial=1.0,
            weight_increment=1.0
        )

        x = attack_graph.victim.logits

        log_mult_sum = tf.reduce_sum(x * tf.log(x), axis=2)
        neg_max = tf.reduce_max(-log_mult_sum, axis=1)

        self.loss_fn = neg_max * self.weights


class SampleL2Loss(BaseLoss):
    """
    Modified CTC Loss with L2 from the original code.
    """
    def __init__(self, attack_graph):

        super().__init__(
            attack_graph.sess,
            attack_graph.batch.size,
            weight_initial=1.0,
            weight_increment=1.0
        )

        original = attack_graph.placeholders.audios
        delta = attack_graph.bounded_deltas

        self.l2original = tf.reduce_sum(tf.abs(original ** 2), axis=1)
        self.l2delta = tf.abs(delta) ** 2
        self.l2_loss = tf.reduce_sum(self.l2delta, axis=1) / self.l2original

        self.loss_fn = self.l2_loss * self.weights


class CWImproved(BaseLoss):
    def __init__(self, attack_graph, target_logits, k=0.0):
        """
        Low Confidence Adversarial Audio Loss as per the original work in
        https://arxiv.org/abs/1801.01944

        It should be noted that this was developed to create `more efficient`
        adversarial examples -- ctc loss makes already certain logits more
        adversarial without needing to.

        **This formulation optimises until the target tokens are the most likely
        class.**

        We use the argmax of the raw logits (target alignment and current)
        to figure out if the target character is currently the most likely.

        If the target character isn't most likely, calculate the max difference
        between the target and current logits:

            - If the most likely character is far more likely than the target
                then we get a big difference.

            - If the most likely character is only just more likely than the
                target then we get a small difference.

        Otherwise, set the difference to -k (we set k=0). Using a negative k
        means that the difference between already optimised characters (e.g. -1)
        and yet to be optimised (e.g. 40) can be made much larger.

        N.B. This loss does *not* conform to l(x + d, t) <= 0 <==> C(x + d) = t
        The loss can be >>> 0 for a successful decoding (e.g. 400)

        :param attack_graph:
        :param target_logits:
        :param importance:
        :param k:
        :param loss_weight:
        """

        super().__init__(
            attack_graph.sess,
            attack_graph.batch.size,
            weight_initial=1.0,
            weight_increment=1.0
        )

        self.target = target_logits
        self.target_max = tf.reduce_max(self.target, axis=2)
        self.argmax_target = tf.argmax(self.target, dimension=2)

        self.current = tf.transpose(attack_graph.victim.raw_logits, [1, 0, 2])
        self.current_max = tf.reduce_max(self.current, axis=2)
        self.argmax_current = tf.argmax(self.current, dimension=2)

        self.argmax_diff = tf.where(
            tf.equal(self.argmax_target, self.argmax_current),
            -k * tf.ones(self.argmax_target.shape, dtype=tf.float32),
            tf.reduce_max(self.target - self.current, axis=2)
        )
        self.loss_fn = tf.reduce_sum(self.argmax_diff, axis=1) * self.weights


class CWMaxDiff(BaseLoss):
    def __init__(self, attack_graph, target_logits, k=0.5):
        """
        This is f_{6} from https://arxiv.org/abs/1608.04644 using the gradient
        clipping update method.

        Difference of:
        - target logits value (B)
        - max other logits value (A -- 2nd most likely)

        Once  B > A, then B is most likely and we can stop optimising.

        Unless -k > B, then k acts as a confidence threshold and continues
        optimisation.

        This will push B to become even less likely.

        N.B. This loss does *not* seems to conform to:
            l(x + d, t) <= 0 <==> C(x + d) = t

        But it does a much better job than the ArgmaxLowConfidence
        implementation as 0 <= l(x + d, t) < 1.0 for a successful decoding

        TODO: This needs testing.
        TODO: normalise to 0 <= x + d <= 1 and convert to tanh space for `change
              of variable` optimisation
        """

        super().__init__(
            attack_graph.sess,
            attack_graph.batch.size,
            weight_initial=1.0,
            weight_increment=1.0
        )

        # We have to set k > 0 for this loss function because k = 0 will only
        # cause the probability of the target character to exactly match the
        # next most likely character...
        # Which means we wouldn't ever achieve success!
        assert type(k) is float

        g = attack_graph

        # We only use the argmax of the generated alignments so we don't have
        # to worry about finding "exact" alignments
        # target_logits should be [b, feats, chars]
        self.target = tf.transpose(target_logits, [1, 0, 2])
        self.target_argmax = tf.argmax(self.target, dimension=2)  # [b x feats]

        # Current logits is [b, feats, chars]
        # current_argmax is for debugging purposes only
        self.current = tf.transpose(g.victim.raw_logits, [1, 0, 2])
        self.current_argmax = tf.argmax(self.current, axis=2)

        # Create one hot matrices to multiply by current logits.
        # These essentially act as a filter to keep only the target logit or
        # the rest of the logits (non-target).
        targ_onehot = tf.one_hot(
            self.target_argmax,
            self.current.shape.as_list()[2],
            on_value=1.0,
            off_value=0.0
        )
        others_onehot = tf.one_hot(
            self.target_argmax,
            self.current.shape.as_list()[2],
            on_value=0.0,
            off_value=1.0
        )

        self.others = self.current * others_onehot
        self.targ = self.current * targ_onehot

        # + 1e-10 covers logit == zero edge case in subsequent where clauses
        # self.others = (self.current + 1e-6) * others_onehot
        # self.targ = (self.current + 1e-6) * targ_onehot

        # DeepSpeech zero-valued Softmax logits are anything < 0.
        # If we don't replace the zero off + on values after one hot
        # multiplication then optimisation could be halted prematurely => the
        # zero can become the most likely class in some cases

        # self.targ = tf.where(
        #     tf.equal(self.targ, tf.zeros(self.targ.shape, dtype=tf.float32)),
        #     -40.0 * tf.ones(self.targ.shape, dtype=tf.float32),
        #     self.targ
        # )
        #
        # self.others = tf.where(
        #     tf.equal(self.others, tf.zeros(self.others.shape, dtype=tf.float32)),
        #     -40.0 * tf.ones(self.others.shape, dtype=tf.float32),
        #     self.others
        # )

        # Get the maximums of:
        # - target logit (should just be the target logit value)
        # - all other logits (should be next most likely class)

        self.target_logit = tf.reduce_max(self.targ, axis=2)
        self.max_other_logit = tf.reduce_max(self.others, axis=2)

        # If target logit is most likely, then the optimiser has done a good job
        # and loss will become negative.
        # Keep optimising until we reached the confidence threshold -- how much
        # distance between logits do we want can be controlled by k

        self.max_diff_abs = self.max_other_logit - self.target_logit

        # MR addition
        # Multiply by minimisation weighting only when target < next class to
        # encourage further optimisation -- we want *highly confident* examples

        # self.max_diff = tf.where(
        #     tf.less_equal(self.max_diff, tf.zeros(self.max_diff.shape, dtype=tf.float32)),
        #     tf.maximum(self.max_diff, -k),
        #     self.max_diff * importance
        # )
        # TODO: per character importance -- could this be adaptive?
        #       e.g. use percentage maxdiff compared to abs. max of maxdiff for
        #       each character? Then, when a character is *VERY* different, it
        #       gets a higher weighting, while other get a lower weighting
        #       (i.e. the characters which are already closer).

        # hacky implementation of character importance, only dealing with
        # repeats and space characters
        #
        # self.max_diff_character_weighted = tf.where(
        #     tf.equal(self.target_argmax, 28),
        #     0.5 * self.max_diff_abs,
        #     self.max_diff_abs,
        # )
        #
        # self.max_diff_character_weighted = tf.where(
        #     tf.equal(self.target_argmax, 0),
        #     0.5 * self.max_diff_abs,
        #     self.max_diff_abs,
        # )
        # Take the maximum between the max diffs or confidence threshold.
        # We add k at the end so the loss is always non-negative.
        # This is for sanity checks and has no impact on optimisation.

        self.max_diff = tf.maximum(self.max_diff_abs, -k) + k
        self.loss_fn = tf.reduce_sum(self.max_diff, axis=1)

        self.loss_fn = self.loss_fn * self.weights
