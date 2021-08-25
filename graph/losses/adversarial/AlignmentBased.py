import tensorflow as tf
from cleverspeech.graph.losses import Bases


class TargetClassesFramewise(Bases.BaseAlignmentLoss):
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


class BiggioMaxMin(Bases.BaseAlignmentLoss):
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


class MaxOfMaxMin(Bases.BaseAlignmentLoss):
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


class CWMaxMin(Bases.BaseAlignmentLoss):
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


class AdaptiveKappaMaxMin(Bases.BaseAlignmentLoss):
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


class WeightedMaxMin(Bases.BaseAlignmentLoss):
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


# GreedyPathTokenWeightingBinarySearch


class BaseSumOfLogProbsLoss(Bases.BaseAlignmentLoss):
    def __init__(self, attack, weight_settings=(None, None), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            use_softmax=True,
            updateable=updateable,
        )

        self.log_smax = tf.log(self.target_logit * self.weights)

        self.fwd_target_log_probs = tf.reduce_sum(
            self.log_smax, axis=-1
        )
        self.back_target_log_probs = tf.reduce_sum(
            tf.reverse(self.log_smax, axis=[-1]), axis=-1
        )


class BaseCumulativeLogProbsLoss(Bases.BaseAlignmentLoss):
    def __init__(self, attack_graph, weight_settings=(None, None)):

        super().__init__(
            attack_graph,
            weight_settings=weight_settings,
            use_softmax=True
        )

        self.log_smax = tf.log(self.target_logit * self.weights)

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
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):
        """
        """

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        self.loss_fn = - self.fwd_target_log_probs

