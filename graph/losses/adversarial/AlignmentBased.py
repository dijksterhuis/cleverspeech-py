import tensorflow as tf
from cleverspeech.graph.losses import Bases


class TargetClassesFramewise(Bases.BaseAlignmentLoss, Bases.SimpleWeightings):
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


class BiggioMaxMin(Bases.SimpleGreedySearchTokenWeights):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False, use_softmax: bool = False):

        super().__init__(
            attack,
            use_softmax=use_softmax,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        self.max_min = - self.target_logit + self.max_other_logit
        self.loss_fn = tf.reduce_sum(self.max_min * self.weights, axis=1)

        self.loss_fn = self.loss_fn


class MaxOfMaxMin(Bases.SimpleGreedySearchTokenWeights):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False, use_softmax: bool = False):

        super().__init__(
            attack,
            use_softmax=use_softmax,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        self.max_min = - self.target_logit + self.max_other_logit
        self.loss_fn = tf.reduce_max(self.max_min * self.weights, axis=1)

        self.loss_fn = self.loss_fn


class CWMaxMin(Bases.SimpleGreedySearchTokenWeights):
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
        self.loss_fn = tf.reduce_sum(self.max_diff * self.weights, axis=1)

        self.loss_fn = self.loss_fn


class AdaptiveKappaMaxMin(Bases.KappaGreedySearchTokenWeights):
    def __init__(self, attack, k=0.0, weight_settings=(1.0, 1.0), updateable: bool = False, use_softmax: bool = False):

        assert k >= 0

        super().__init__(
            attack,
            use_softmax=use_softmax,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        self.max_diff_abs = - self.target_logit + self.max_other_logit
        self.max_diff = tf.maximum(self.max_diff_abs, -self.kappa) + self.kappa
        self.loss_fn = tf.reduce_sum(self.max_diff * self.weights, axis=1)

        self.loss_fn = self.loss_fn

    def check_kappa(self, batch_successes):

        tf_vars = [
            self.kappa, self.weights, self.tf_tokens_test_check
        ]

        kaps, weights, kap_test = self.attack.procedure.tf_run(tf_vars)

        z = zip(batch_successes, kaps, weights, kap_test)
        for idx, (suc, k, w, t) in enumerate(z):

            # N.B. `t` is a numpy.bool, which is not a real bool so `t is True`
            # actually evaluates to False when t really is True... one of the
            # joys of dynamically typed languages:
            # https://stackoverflow.com/a/37744300/5945794

            if suc is False and bool(t) is True:
                # increment kappa upwards so the perturbation can become more
                # confident
                kaps[idx] += 0.1
                log("\n updated kappa {k}".format(k=kaps[idx]), wrap=False)

        self.attack.sess.run(self.kappa.assign(kaps))

        # self.attack.sess.run(
        #     [self.kappa.assign(kaps), self.weights.assign(weights)]
        # )


class WeightedMaxMin(Bases.BaseAlignmentLoss, Bases.SimpleWeightings):
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


class SinglePathCTCLoss(Bases.SimpleWeightings):
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


class SumLogProbsForward(Bases.BaseAlignmentLoss, Bases.SimpleWeightings):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):
        """
        """

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
            use_softmax=True,
        )

        self.log_smax = tf.log(self.target_logit)

        self.fwd_target_log_probs = tf.reduce_sum(
            self.log_smax, axis=-1
        )
        self.back_target_log_probs = tf.reduce_sum(
            tf.reverse(self.log_smax, axis=[-1]), axis=-1
        )

        self.loss_fn = - self.fwd_target_log_probs * self.weights


class SumLogProbsBackward(Bases.BaseAlignmentLoss, Bases.SimpleWeightings):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):
        """
        """

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
            use_softmax=True,
        )

        self.log_smax = tf.log(self.target_logit)

        self.fwd_target_log_probs = tf.reduce_sum(
            self.log_smax, axis=-1
        )
        self.back_target_log_probs = tf.reduce_sum(
            tf.reverse(self.log_smax, axis=[-1]), axis=-1
        )

        self.loss_fn = - self.back_target_log_probs * self.weights


class SumLogProbsProduct(Bases.SimpleWeightings):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):
        """
        """

        super().__init__(
            attack, weight_settings=weight_settings, updateable=updateable
        )

        forwards = SumLogProbsBackward(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        backwards = SumLogProbsBackward(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        self.loss_fn = tf.log(forwards.loss_fn) + tf.log(backwards.loss_fn)


class CumulativeLogProbsForward(Bases.SimpleBeamSearchTokenWeights):
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

        self.loss_fn = - self.fwd_target_log_probs

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


class CumulativeLogProbsBackward(Bases.SimpleBeamSearchTokenWeights):
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

        self.loss_fn = - self.back_target_log_probs

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


class CumulativeLogProbsProduct(Bases.SimpleBeamSearchTokenWeights):
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

        self.loss_fn = -(self.back_target_log_probs * self.fwd_target_log_probs)

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


GREEDY = {
    "cw": CWMaxMin,
    "biggio": BiggioMaxMin,
    "maxofmaxmin": MaxOfMaxMin,
    "adaptive-kappa": AdaptiveKappaMaxMin,
    "maxtargetonly": TargetClassesFramewise,
    "weightedmaxmin": WeightedMaxMin,
}


NON_GREEDY = {
    "ctc-fixed-path": SinglePathCTCLoss,
    "sumlogprobs-fwd": SumLogProbsForward,
    "sumlogprobs-back": SumLogProbsBackward,
    "sumlogprobs-mult": SumLogProbsProduct,
}
