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
    def __init__(self, attack, weight_settings: tuple = (None, None), updateable: bool = False):

        assert type(weight_settings) in [list, tuple]
        assert type(updateable) is bool
        assert all(type(t) in [float, int] for t in weight_settings)
        assert len(weight_settings) == 2

        initial = weight_settings[0]
        increment = weight_settings[1]

        self.updateable = updateable
        self.__sess = attack.sess

        self.weights = tf.Variable(
            tf.ones(attack.batch.size, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        initial_vals = initial * np.ones([attack.batch.size], dtype=np.float32)
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
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        # N.B. original code did `reduce_mean` on `(advex - original) ** 2`...
        # `tf.reduce_mean` on `deltas` is exactly the same with fewer variables

        l2delta = tf.reduce_mean(attack.perturbations ** 2, axis=1)
        self.loss_fn = l2delta * self.weights


class SampleL2Loss(BaseLoss):
    """
    Normalised L2 loss component from https://arxiv.org/abs/1801.01944

    Use the original example's L2 norm for normalisation.
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        assert type(weight_settings) in list, tuple

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        original = attack.placeholders.audios
        delta = attack.perturbations

        self.l2original = tf.reduce_sum(tf.abs(original ** 2), axis=1)
        self.l2delta = tf.abs(delta) ** 2
        self.l2_loss = tf.reduce_sum(self.l2delta, axis=1) / self.l2original

        self.loss_fn = self.l2_loss * self.weights


class LinfLoss(BaseLoss):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        # N.B. original code did `reduce_mean` on `(advex - original) ** 2`...
        # `tf.reduce_mean` on `deltas` is exactly the same with fewer variables

        l2delta = tf.reduce_max(attack.perturbations, axis=1)
        self.loss_fn = l2delta * self.weights


class EntropyLoss(BaseLoss):
    """
    Try to minimise the maximum entropy measure as it was used by Lea Schoenherr
    to try to detect adversarial examples.
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        assert type(weight_settings) in list, tuple

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        x = attack.victim.logits

        log_mult_sum = tf.reduce_sum(x * tf.log(x), axis=2)
        neg_max = tf.reduce_max(-log_mult_sum, axis=1)

        self.loss_fn = neg_max * self.weights


class CTCLoss(BaseLoss):
    """
    Simple adversarial CTC Loss from https://arxiv.org/abs/1801.01944

    N.B. This loss does *not* conform to l(x + d, t) <= 0 <==> C(x + d) = t
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
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
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
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


class BasePathsLoss(BaseLoss):
    """
    Base class that can be used for logits difference losses, like CW f_6
    and the adaptive kappa variant.

    :param: attack: an attack class.
    :param: target_argmax: frame length vector of desired target class indices
    :param: softmax: bool type. whether to use softmax or activations
    :param: weight_settings: how to update this loss function on success
    """

    def __init__(self, attack, use_softmax=False, weight_settings=(None, None), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        # Indices of the specified alignment per frame
        self.target_argmax = attack.placeholders.targets

        # we want to be able to choose either softmax or activations depending
        # on the attack

        if use_softmax is False:

            # logits are time major so transpose them
            self.current = tf.transpose(attack.victim.raw_logits, [1, 0, 2])

        else:

            # softmax is batch major so all is well
            self.current = attack.victim.logits

        # Use one hot matrix as a filter on current network outputs

        targ_onehot = tf.one_hot(
            self.target_argmax,
            self.current.shape.as_list()[2],
            on_value=1.0,
            off_value=0.0
        )

        # targ will only be non-zero for the target class in each frame
        # so we can just do a sum after filtering to get the current value

        self.targ = self.current * targ_onehot
        self.target_logit = tf.reduce_sum(self.targ, axis=2)

        # the max other class per frame is slightly more tricky if we're using
        # activations

        others_onehot = tf.one_hot(
            self.target_argmax,
            self.current.shape.as_list()[2],
            on_value=0.0,
            off_value=1.0
        )

        if use_softmax is False:

            # if the most likely other activation for a frame is negative we
            # need to make sure we don't accidentally select the 0.0 one hot
            # filter value

            # get the current maximal value over the entire activations matix

            maximal = tf.reduce_max(self.current, axis=0)

            # set the onehot zero values to the negative of 2 * the biggest
            # current activation entry to guarantee we *never* choose it
            # ==> we could minus by a constant, but this could lead to weird
            # edge case behaviour if an attack were to do *really* well

            self.others = tf.where(
                tf.equal(others_onehot, tf.zeros_like(others_onehot)),
                tf.zeros_like(others_onehot) - (2 * maximal),
                self.current * others_onehot
            )

        else:

            # softmax is guaranteed to be in the 0 -> 1 range, so we don't need
            # to worry about doing this

            self.others = self.current * others_onehot

        # finally, we can do the max_{k' \neq k} op.

        self.max_other_logit = tf.reduce_max(self.others, axis=2)


class TargetClassesFramewise(BasePathsLoss):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False, use_softmax: bool = False):

        super().__init__(
            attack,
            use_softmax=use_softmax,
            weight_settings=weight_settings,
            updateable=updateable,
        )
        n_frames = self.target_logit.shape.as_list()[1]
        self.loss_fn = tf.reduce_sum(-self.target_logit, axis=1) + n_frames

        self.loss_fn *= self.weights


class BiggioMaxMin(BasePathsLoss):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False, use_softmax: bool = False):

        super().__init__(
            attack,
            use_softmax=use_softmax,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        self.max_min = - self.target_logit + self.max_other_logit
        self.loss_fn = tf.reduce_sum(self.max_min, axis=1)

        self.loss_fn = self.loss_fn * self.weights


class MaxOfMaxMin(BasePathsLoss):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False, use_softmax: bool = False):

        super().__init__(
            attack,
            use_softmax=use_softmax,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        self.max_min = - self.target_logit + self.max_other_logit
        self.loss_fn = tf.reduce_max(self.max_min, axis=1)

        self.loss_fn = self.loss_fn * self.weights


class CWMaxMin(BasePathsLoss):
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
    def __init__(self, attack, k=0.0, weight_settings=(1.0, 1.0), updateable: bool = False, use_softmax: bool = False):

        assert k >= 0

        super().__init__(
            attack,
            use_softmax=use_softmax,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        self.max_diff_abs = - self.target_logit + self.max_other_logit
        self.max_diff = tf.maximum(self.max_diff_abs, -k) + k
        self.loss_fn = tf.reduce_sum(self.max_diff, axis=1)

        self.loss_fn = self.loss_fn * self.weights


class AdaptiveKappaMaxMin(BasePathsLoss):
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
    def __init__(self, attack, k=0.0, ref_fn=tf.reduce_min, weight_settings=(1.0, 1.0), updateable: bool = False, use_softmax: bool = False):

        super().__init__(
            attack,
            use_softmax=use_softmax,
            weight_settings=weight_settings,
            updateable=updateable,
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


class SinglePathCTCLoss(BaseLoss):
    """
    Adversarial CTC Loss that uses an alignment as a target instead of a
    transcription.

    This means we can perform maximum likelihood estimation optimisation for a
    specified target alignment. But it is inefficient as we don't need all the
    CTC rules if we have a known target alignment that maps to a target
    transcription.

    Does not work for target transcriptions on their own as they end up like
    `----o--o-oo-o----p- ...` which merges down to `ooop ...` instead of `open`.

    Notes:
        - This loss does *not* conform to l(x + d, t) <= 0 <==> C(x + d) = t
        - This loss is pretty much the same as SumofLogProbs Losses, except it's
        more computationally expensive.

    :param: attack_graph:
    :param: alignment
    :param: weight_settings
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        seq_lengths = attack.batch.audios["ds_feats"]

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


class WeightedMaxMin(BasePathsLoss):
    def __init__(self, attack, k=0.0, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack,
            use_softmax=True,
            weight_settings=weight_settings,
        )

        self.c = c = 1 - self.target_logit
        self.weighted_diff = c * (tf.maximum(- self.target_logit + self.max_other_logit, -k) + k)

        self.loss_fn = tf.reduce_sum(self.weighted_diff, axis=1)
        self.loss_fn *= self.weights


class BaseSumOfLogProbsLoss(BasePathsLoss):
    def __init__(self, attack, weight_settings=(None, None)):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            use_softmax=True
        )

        self.log_smax = tf.log(self.target_logit)

        self.fwd_target_log_probs = tf.reduce_sum(
            self.log_smax, axis=-1
        )
        self.back_target_log_probs = tf.reduce_sum(
            tf.reverse(self.log_smax, axis=[-1]), axis=-1
        )


class BaseCumulativeLogProbsLoss(BasePathsLoss):
    def __init__(self, attack_graph, weight_settings=(None, None)):

        super().__init__(
            attack_graph,
            weight_settings=weight_settings,
            use_softmax=True
        )

        self.log_smax = tf.log(self.target_logit)

        self.fwd_target = self.target_probs(self.log_smax)
        self.back_target = self.target_probs(self.log_smax, backward_pass=True)

        self.fwd_target_log_probs = self.fwd_target[:, -1]
        self.back_target_log_probs = self.back_target[:, -1]

    @staticmethod
    def target_probs(x_t, backward_pass=False):
        probability_vector = tf.cumsum(
            x_t,
            exclusive=False,
            reverse=backward_pass,
            axis=1
        )
        if backward_pass:
            probability_vector = tf.reverse(probability_vector, axis=[1])

        return probability_vector


class FwdOnlyLogProbsLoss(BaseSumOfLogProbsLoss):
    def __init__(self, attack, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs
        self.loss_fn *= -self.weights


class BackOnlyLogProbsLoss(BaseSumOfLogProbsLoss):
    def __init__(self, attack, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.back_target_log_probs
        self.loss_fn *= -self.weights


class FwdPlusBackLogProbsLoss(BaseSumOfLogProbsLoss):
    def __init__(self, attack, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs + self.back_target_log_probs
        self.loss_fn *= -self.weights


class FwdMultBackLogProbsLoss(BaseSumOfLogProbsLoss):
    def __init__(self, attack, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs * self.back_target_log_probs
        self.loss_fn *= -self.weights


CONFIDENCE_MEASURES = {
    "max-entropy": EntropyLoss,
}

PATH_BASED_LOSSES = {
    "ctc-fixed-path": SinglePathCTCLoss,
    "sumlogprobs-fwd": FwdOnlyLogProbsLoss,
    "sumlogprobs-back": BackOnlyLogProbsLoss,
    "cw": CWMaxMin,
    "biggio": BiggioMaxMin,
    "maxofmaxmin": MaxOfMaxMin,
    "adaptive-kappa": AdaptiveKappaMaxMin,
    "maxtargetonly": TargetClassesFramewise,
    "weightedmaxmin": WeightedMaxMin,
}

GREEDY_SEARCH_ADV_LOSSES = {
    "ctc": CTCLoss,
    "ctcv2": CTCLossV2,
    "cw": CWMaxMin,
    "biggio": BiggioMaxMin,
    "maxofmaxmin": MaxOfMaxMin,
    "adaptive-kappa": AdaptiveKappaMaxMin,
    "maxtargetonly": TargetClassesFramewise,
    "weightedmaxmin": WeightedMaxMin,
}

BEAM_SEARCH_ADV_LOSSES = {
    "ctc": CTCLoss,
    "ctcv2": CTCLossV2,
    "ctc-fixed-path": SinglePathCTCLoss,
    "sumlogprobs-fwd": FwdOnlyLogProbsLoss,
    "sumlogprobs-back": BackOnlyLogProbsLoss,
    # "cumulativelogprobs-fwd": None,
    # "cumulativelogprobs-back": None,
    # "logprobsgreedydiff": None,
    # "maxconfctc": None,
}

ALL_ADV_LOSS_TERMS = {
    "ctc": CTCLoss,
    "ctcv2": CTCLossV2,
    "cw": CWMaxMin,
    "biggio": BiggioMaxMin,
    "maxofmaxmin": MaxOfMaxMin,
    "ctc-fixed-path": SinglePathCTCLoss,
    "adaptive-kappa": AdaptiveKappaMaxMin,
    "maxtargetonly": TargetClassesFramewise,
    "single-path-ctc": SinglePathCTCLoss,
    "weightedmaxmin": WeightedMaxMin,

}

DISTANCE_LOSS_TERMS = {
    "l2-carlini": CarliniL2Loss,
    "l2-sample": SampleL2Loss,
    "linf": LinfLoss,
    "rms-energy": None,
    "energy": None,
    "peak-to-peak": None,
}
