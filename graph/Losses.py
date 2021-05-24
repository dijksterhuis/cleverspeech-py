"""
These are the "stable" adversarial loss/objective functions used to generate
adversarial examples.

**IMPORTANT**: These loss functions do not perform perturbation
minimisation, they only make examples adversarial (it is a common misconception
that adversarial examples should be minimally perturbed -- Biggio et al. 2017)

"stable" means they've been battle tested whilst living in some custom_loss.py
script for a while and are a valid way of running an attack.

The results, of course, depend on the loss function!

--------------------------------------------------------------------------------
"""


import tensorflow as tf
import numpy as np

from cleverspeech.utils.Utils import log


class BaseLoss:
    """
    Abstract base loss which enables us to perform any loss weightings applied
    during optimisation uniformly across all child classes.

    :param: sess: a tensorflow session object.
    :param: batch_size: the size of the input batch.
    :param: weight_settings: how to weight this loss object, tuple of floats
        with length 2 where the first entry is the initial weighting and the
        second entry is how much to add to the current weighting after an
        update.
    """
    def __init__(self, sess, batch_size: int, weight_settings: tuple = (None, None)):

        assert type(weight_settings) in [list, tuple]
        assert all(type(t) in [float, int] for t in weight_settings)
        assert len(weight_settings) == 2

        initial = weight_settings[0]
        increment = weight_settings[1]

        self.__sess = sess

        self.weights = tf.Variable(
            tf.ones(batch_size, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        initial_vals = initial * np.ones([batch_size], dtype=np.float32)
        self.__sess.run(self.weights.assign(initial_vals))

        self.increment = float(increment)

    def update_one(self, idx: int):
        """
        Apply the loss weighting updates to only one example in a batch.

        :param idx: the batch index of the example to update
        """
        weights = self.__sess.run(self.weights)
        weights[idx] += self.increment
        self.__sess.run(self.weights.assign(weights))

    def update_many(self, batch_successes: list):
        """
        Apply the loss weighting updates to only one example in a batch.

        :param batch_successes: a list of True/False false indicating whether
            the loss weighting should be updated for each example in a batch
        """

        weights = self.__sess.run(self.weights)

        for idx, success_check in enumerate(batch_successes):
            if success_check is True:
                weights[idx] += self.increment

        self.__sess.run(self.weights.assign(weights))


class CarliniL2Loss(BaseLoss):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack.sess,
            attack.batch.size,
            weight_settings=weight_settings
        )

        # N.B. original code did `reduce_mean` on `(advex - original) ** 2`...
        # `tf.reduce_mean` on `deltas` is exactly the same with fewer variables

        l2delta = tf.reduce_mean(attack.perturbations ** 2, axis=1)
        self.loss_fn = l2delta / self.weights


class CTCLoss(BaseLoss):
    """
    Simple adversarial CTC Loss from https://arxiv.org/abs/1801.01944

    N.B. This loss does *not* conform to l(x + d, t) <= 0 <==> C(x + d) = t
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack.sess,
            attack.batch.size,
            weight_settings=weight_settings
        )

        self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            attack.placeholders.targets,
            attack.placeholders.target_lengths
        )

        self.loss_fn = tf.nn.ctc_loss(
            labels=tf.cast(self.ctc_target, tf.int32),
            inputs=attack.victim.raw_logits,
            sequence_length=attack.batch.audios["ds_feats"]
        ) * self.weights


class CTCLossV2(BaseLoss):
    """
    Simple adversarial CTC Loss from https://arxiv.org/abs/1801.01944

    N.B. This loss does *not* conform to l(x + d, t) <= 0 <==> C(x + d) = t
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack.sess,
            attack.batch.size,
            weight_settings=weight_settings
        )

        self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            attack.placeholders.targets,
            attack.placeholders.target_lengths
        )

        self.loss_fn = tf.nn.ctc_loss_v2(
            labels=tf.cast(self.ctc_target, tf.int32),
            logits=attack.victim.raw_logits,
            label_length=attack.placeholders.target_lengths,
            logit_length=attack.batch.audios["ds_feats"],
            blank_index=-1
        ) * self.weights


class EntropyLoss(BaseLoss):
    """
    Try to minimise the maximum entropy measure as it was used by Lea Schoenherr
    to try to detect adversarial examples.
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0)):

        assert type(weight_settings) in list, tuple

        super().__init__(
            attack.sess,
            attack.batch.size,
            weight_settings=weight_settings
        )

        x = attack.victim.logits

        log_mult_sum = tf.reduce_sum(x * tf.log(x), axis=2)
        neg_max = tf.reduce_max(-log_mult_sum, axis=1)

        self.loss_fn = neg_max * self.weights


class SampleL2Loss(BaseLoss):
    """
    Normalised L2 loss component from https://arxiv.org/abs/1801.01944

    Use the original example's L2 norm for normalisation.
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0)):

        assert type(weight_settings) in list, tuple

        super().__init__(
            attack.sess,
            attack.batch.size,
            weight_settings=weight_settings
        )

        original = attack.placeholders.audios
        delta = attack.perturbations

        self.l2original = tf.reduce_sum(tf.abs(original ** 2), axis=1)
        self.l2delta = tf.abs(delta) ** 2
        self.l2_loss = tf.reduce_sum(self.l2delta, axis=1) / self.l2original

        self.loss_fn = self.l2_loss * self.weights


class BaseLogitDiffLoss(BaseLoss):
    """
    Base class that can be used for logits difference losses, like CW f_6
    and the adaptive kappa variant.

    :param: attack_graph:
    :param: target_argmax:
    :param: softmax:
    :param: weight_settings:
    """
    def __init__(self, attack, target_argmax, softmax=False, weight_settings=(None, None)):

        super().__init__(
            attack.sess,
            attack.batch.size,
            weight_settings=weight_settings
        )

        # We only use the argmax of the generated alignments so we don't have
        # to worry about finding "exact" alignments
        # target_logits should be [b, feats, chars]
        self.target_argmax = target_argmax  # [b x feats]

        # Current logits is [b, feats, chars]
        # current_argmax is for debugging purposes only

        if softmax is False:
            self.current = tf.transpose(attack.victim.raw_logits, [1, 0, 2])
        else:
            self.current = attack.victim.logits

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

        # Get the maximums of:
        # - target logit (should just be the target logit value)
        # - all other logits (should be next most likely class)

        self.target_logit = tf.reduce_sum(self.targ, axis=2)
        self.max_other_logit = tf.reduce_max(self.others, axis=2)


class MaximiseTargetFramewiseSoftmax(BaseLogitDiffLoss):
    def __init__(self, attack, target_logits, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack,
            target_logits,
            softmax=True,
            weight_settings=weight_settings,
        )
        n_frames = self.target_logit.shape.as_list()[1]
        self.loss_fn = tf.reduce_sum(-self.target_logit, axis=1) + n_frames

        self.loss_fn *= self.weights


class MaximiseTargetFramewiseActivations(BaseLogitDiffLoss):
    def __init__(self, attack, target_logits, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack,
            target_logits,
            softmax=False,
            weight_settings=weight_settings,
        )
        self.loss_fn = tf.reduce_sum(-self.target_logit, axis=1) * self.weights


class BiggioMaxMin(BaseLogitDiffLoss):
    def __init__(self, attack, target_logits, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack,
            target_logits,
            weight_settings=weight_settings,
        )

        self.max_min = - self.target_logit + self.max_other_logit
        self.loss_fn = tf.reduce_sum(self.max_min, axis=1)

        self.loss_fn = self.loss_fn * self.weights


class MaxOfBiggioMaxMinLogits(BaseLogitDiffLoss):
    def __init__(self, attack, target_logits, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack,
            target_logits,
            weight_settings=weight_settings,
        )

        self.max_min = - self.target_logit + self.max_other_logit
        self.loss_fn = tf.reduce_max(self.max_min, axis=1)

        self.loss_fn = self.loss_fn * self.weights


class BiggioMaxMinSoftmax(BaseLogitDiffLoss):
    def __init__(self, attack, target_logits, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack,
            target_logits,
            softmax=True,
            weight_settings=weight_settings,
        )

        self.max_min = - self.target_logit + self.max_other_logit
        self.loss_fn = tf.reduce_sum(self.max_min, axis=1)

        self.loss_fn = self.loss_fn * self.weights


class MaxOfBiggioMaxMinSoftmax(BaseLogitDiffLoss):
    def __init__(self, attack, target_logits, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack,
            target_logits,
            softmax=True,
            weight_settings=weight_settings,
        )

        self.max_min = - self.target_logit + self.max_other_logit
        self.loss_fn = tf.reduce_max(self.max_min, axis=1)

        n_frames = self.max_min.shape.as_list()[1]

        self.loss_fn = (self.loss_fn + n_frames) * self.weights


class CWImproved(BaseLoss):
    """
    Improved Loss as per the original work in https://arxiv.org/abs/1801.01944

    It should be noted that this was developed to create `more efficient`
    adversarial examples -- ctc loss makes already certain logits more
    adversarial without needing to.

    **This formulation optimises until the target tokens are the most likely
    class.**

    We use the argmax of the raw logits (target alignment and current)
    to figure out if the target character is currently the most likely.

    If the target character isn't most likely, calculate the max difference
    between the target and current logits. If the most likely character is:

    - far more likely than the target then we get a big difference.
    - only just more likely than the target then we get a small difference.

    Otherwise, set the difference to -k (we set k=0). Using a negative k
    means that the difference between already optimised characters (e.g. -1)
    and yet to be optimised (e.g. 40) can be made much larger.

    N.B. This loss does *not* conform to l(x + d, t) <= 0 <==> C(x + d) = t
    The loss can be >>> 0 for a successful decoding (e.g. 400)

    :param attack:
    :param target_logits:
    :param k:
    :param weight_settings:
    """
    def __init__(self, attack, target_logits, k=0.0, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack.sess,
            attack.batch.size,
            weight_settings=weight_settings
        )

        self.target = target_logits
        self.target_max = tf.reduce_max(self.target, axis=2)
        self.argmax_target = tf.argmax(self.target, dimension=2)

        self.current = tf.transpose(attack.victim.raw_logits, [1, 0, 2])
        self.current_max = tf.reduce_max(self.current, axis=2)
        self.argmax_current = tf.argmax(self.current, dimension=2)

        self.argmax_diff = tf.where(
            tf.equal(self.argmax_target, self.argmax_current),
            -k * tf.ones(self.argmax_target.shape, dtype=tf.float32),
            tf.reduce_max(self.target - self.current, axis=2)
        )
        self.loss_fn = tf.reduce_sum(self.argmax_diff, axis=1) * self.weights


class CWMaxDiff(BaseLogitDiffLoss):
    """
    This is f_{6} from https://arxiv.org/abs/1608.04644 using the gradient
    clipping update method.

    Difference of:

    - target logits value (B)
    - max other logits value (A -- most likely other)

    Once B > A, then B is most likely and we can stop optimising.

    Unless -k > B, then k acts as a confidence threshold and continues
    optimisation.

    This will push B to become even more likely.

    N.B. This loss does *not* seems to conform to:
        l(x + d, t) <= 0 <==> C(x + d) = t

    But it does a much better job than the ArgmaxLowConfidence
    implementation as 0 <= l(x + d, t) < 1.0 for a successful decoding

    :param: attack:
    :param: target_logits:
    :param: k:
    :param: weight_settings:
    """
    def __init__(self, attack, target_logits, k=0.5, weight_settings=(1.0, 1.0)):

        assert k >= 0

        super().__init__(
            attack,
            target_logits,
            weight_settings=weight_settings,
        )

        self.max_diff_abs = - self.target_logit + self.max_other_logit
        self.max_diff = tf.maximum(self.max_diff_abs, -k) + k
        self.loss_fn = tf.reduce_sum(self.max_diff, axis=1)

        self.loss_fn = self.loss_fn * self.weights


class CWMaxDiffSoftmax(BaseLogitDiffLoss):
    """
    :param: attack:
    :param: target_logits:
    :param: k:
    :param: weight_settings:
    """
    def __init__(self, attack, target_logits, k=0.5, weight_settings=(1.0, 1.0)):

        assert 0 <= k <= 1.0

        super().__init__(
            attack,
            target_logits,
            softmax=True,
            weight_settings=weight_settings,
        )

        self.max_diff_abs = - self.target_logit + self.max_other_logit
        self.max_diff = tf.maximum(self.max_diff_abs, -k) + k
        self.loss_fn = tf.reduce_sum(self.max_diff, axis=1)

        self.loss_fn = self.loss_fn * self.weights


class AdaptiveKappaMaxDiff(BaseLogitDiffLoss):
    """
    This is a modified version of f_{6} from https://arxiv.org/abs/1608.04644
    using the gradient clipping update method.

    Difference of:
    - target logits value (B)
    - max other logits value (A -- 2nd most likely)

    Once  B > A, then B is most likely and we can stop optimising.

    Unless -kappa > B, then kappa acts as a confidence threshold and continues
    optimisation.

    Where kappa is an adaptive value based on the results of the reference
    function on the softmax vector values for that frame.

    Basically, each frame step should have a different k constant, so we
    calculate it adaptively based on the frame's probability distribution.

    :param: attack_graph
    :param: target_argmax
    :param: k
    :param: ref_fn
    :param: weight_settings

    """
    def __init__(self, attack, target_argmax, k=0.5, ref_fn=tf.reduce_min, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack,
            target_argmax,
            weight_settings=weight_settings,
        )

        # We have to set k > 0 for this loss function because k = 0 will only
        # cause the probability of the target character to exactly match the
        # next most likely character...
        assert type(k) is float
        assert ref_fn in [tf.reduce_max, tf.reduce_min, tf.reduce_mean]

        self.kappa_distrib = self.max_other_logit - ref_fn(self.others, axis=2)

        # each target logit frame must be at least k * other logits difference
        # for loss to minimise.
        self.kappas = self.kappa_distrib * k

        # If target logit is most likely, then the optimiser has done a good job
        # and loss will become negative.
        # Add kappa on the end so that loss is zero when minimised
        self.max_diff_abs = self.max_other_logit - self.target_logit
        self.max_diff = tf.maximum(self.max_diff_abs, -self.kappas) + self.kappas
        self.loss_fn = tf.reduce_sum(self.max_diff, axis=1)

        self.loss_fn = self.loss_fn * self.weights


class AlignmentsCTCLoss(BaseLoss):
    """
    Adversarial CTC Loss that uses an alignment as a target instead of a
    transcription.

    This means we can perform maximum likelihood estimation optimisation for a
    specified target alignment. But it is inefficient as we don't need all the
    CTC rules if we have a known target alignment that maps to a target
    transcription.

    Does not work for target transcriptions on their own as they end up like
    `----o--o-oo-o----p- ...` which merges down to `ooop ...` instead of `open`.

    N.B. This loss does *not* conform to l(x + d, t) <= 0 <==> C(x + d) = t

    :param: attack_graph:
    :param: alignment
    :param: weight_settings
    """
    def __init__(self, attack, alignment=None, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack.sess,
            attack.batch.size,
            weight_settings=weight_settings,
        )

        seq_lengths = attack.batch.audios["ds_feats"]

        if alignment is not None:
            log("Using CTC alignment search.", wrap=True)
            self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
                alignment,
                attack.batch.audios["ds_feats"],
            )
        else:
            log("Using repeated alignment.", wrap=True)
            self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
                attack.placeholders.targets,
                attack.placeholders.target_lengths,
            )

        logits_shape = attack.victim.raw_logits.get_shape().as_list()

        blank_token_pad = tf.zeros(
            [logits_shape[0], logits_shape[1], 1],
            tf.float32
        )

        logits_mod = tf.concat(
            [attack.victim.raw_logits, blank_token_pad],
            axis=2
        )

        self.loss_fn = tf.nn.ctc_loss(
            labels=tf.cast(self.ctc_target, tf.int32),
            inputs=logits_mod,
            sequence_length=seq_lengths,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=False,
        ) * self.weights


class GreedyOtherAlignmentsCTCLoss(BaseLoss):
    """
    Goal: Make all other logits values *less likely* by optimising with CTC
    loss for all valid alignments *except* our target alignment.

    :param: attack_graph
    :param: alignment
    :param: weight_settings

    """
    def __init__(self, attack, alignment=None, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack.sess,
            attack.batch.size,
            weight_settings=weight_settings
        )

        seq_lengths = attack.batch.audios["ds_feats"]

        self.target_argmax = alignment  # [b x feats]

        # Current logits is [b, feats, chars]
        # current_argmax is for debugging purposes only
        self.current = tf.transpose(attack.victim.raw_logits, [1, 0, 2])

        # Create one hot matrices to multiply by current logits.
        # These essentially act as a filter to keep only the target logit or
        # the rest of the logits (non-target).

        targs_onehot = tf.one_hot(
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

        # AntiCTC sjould make all valid alignments that are *not* the target
        # alignment *less* likely, so we modify the logits to include the
        # classes per frame that are not in our target alignment.

        logits_mod = others_onehot * self.current
        self.targs = tf.reduce_sum(targs_onehot * self.current, axis=2)

        if alignment is not None:
            log("Using CTC alignment search.", wrap=True)
            self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
                alignment,
                attack.batch.audios["ds_feats"],
            )
        else:
            log("Using repeated alignment.", wrap=True)
            self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
                attack.placeholders.targets,
                attack.placeholders.target_lengths,
            )

        logits_shape = attack.victim.raw_logits.get_shape().as_list()

        blank_token_pad = tf.zeros(
            [logits_shape[0], logits_shape[1], 1],
            tf.float32
        )

        self.logits_mod = tf.concat(
            [tf.transpose(logits_mod, [1, 0, 2]), blank_token_pad],
            axis=2
        )

        self.loss_fn = -tf.nn.ctc_loss(
            labels=tf.cast(self.ctc_target, tf.int32),
            inputs=self.logits_mod,
            sequence_length=seq_lengths,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=False,
        ) * self.weights

