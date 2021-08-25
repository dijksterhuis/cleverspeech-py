import tensorflow as tf
from cleverspeech.graph.losses import Bases

from tensorflow.python.framework import ops
from tensorflow.python.ops.ctc_ops import (
    _ctc_state_trans,
    _get_dim,
    ctc_state_log_probs,
    _forward_backward_log,
    _state_to_olabel_unique,
    _state_to_olabel,
    _ilabel_to_state,
    ctc_loss
)
from tensorflow.python.ops import (
    nn_ops,
    array_ops,
    math_ops,
    custom_gradient
)


def ctc_mod(labels, logits, label_length, logit_length, logits_time_major=True,
        unique=None, blank_index=None, name=None, apply_softmax=True):
    """
    Set everything up for CTC
    """
    if blank_index is None:
        blank_index = -1

    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    label_length = ops.convert_to_tensor(
        label_length, name="label_length"
    )
    logit_length = ops.convert_to_tensor(
        logit_length, name="logit_length"
    )

    if not logits_time_major:
        logits = array_ops.transpose(logits, perm=[1, 0, 2])

    if apply_softmax:
        logits = nn_ops.softmax(logits)

    if blank_index != 0:
        if blank_index < 0:
            blank_index += _get_dim(logits, 2)
        logits = array_ops.concat([
            logits[:, :, blank_index:blank_index + 1],
            logits[:, :, :blank_index],
            logits[:, :, blank_index + 1:],
        ],
            axis=2
        )
        labels = array_ops.where(
            labels < blank_index, labels + 1, labels
        )

    args = [logits, labels, label_length, logit_length]

    if unique:
        unique_y, unique_idx = unique
        args.extend([unique_y, unique_idx])

    @custom_gradient.custom_gradient
    def compute_ctc_loss(logits_t, labels_t, label_length_t, logit_length_t,
            *unique_t):

        """
        Compute CTC loss.
        """

        logits_t.set_shape(logits.shape)
        labels_t.set_shape(labels.shape)
        label_length_t.set_shape(label_length.shape)
        logit_length_t.set_shape(logit_length.shape)

        kwargs = dict(
            logits=logits_t,
            labels=labels_t,
            label_length=label_length_t,
            logit_length=logit_length_t
        )
        if unique_t:
            kwargs["unique"] = unique_t

        result = ctc_loss_and_grad(**kwargs)

        def grad(grad_loss):
            grad = [
                array_ops.reshape(grad_loss, [1, -1, 1]) * result[1]
            ]
            grad += [None] * (len(args) - len(grad))
            return grad

        return result[0], grad

    return compute_ctc_loss(*args)


def ctc_loss_and_grad(logits, labels, label_length, logit_length, unique=None):
    """
    Computes the actual loss and gradients.
    """

    num_labels = _get_dim(logits, 2)
    max_label_seq_length = _get_dim(labels, 1)

    # this line is what needed to be changed to be able to get softmax
    ilabel_log_probs = tf.log(logits)

    state_log_probs = _ilabel_to_state(labels, num_labels, ilabel_log_probs)
    state_trans_probs = _ctc_state_trans(labels)

    initial_state_log_probs, final_state_log_probs = ctc_state_log_probs(
        label_length, max_label_seq_length
    )

    fwd_bwd_log_probs, log_likelihood = _forward_backward_log(
        state_trans_log_probs=math_ops.log(state_trans_probs),
        initial_state_log_probs=initial_state_log_probs,
        final_state_log_probs=final_state_log_probs,
        observed_log_probs=state_log_probs,
        sequence_length=logit_length)

    if unique:
        olabel_log_probs = _state_to_olabel_unique(
            labels, num_labels, fwd_bwd_log_probs, unique
        )
    else:
        olabel_log_probs = _state_to_olabel(
            labels, num_labels, fwd_bwd_log_probs
        )

    grad = math_ops.exp(ilabel_log_probs) - math_ops.exp(olabel_log_probs)
    loss = -log_likelihood
    return loss, grad


class CTCLoss(Bases.SimpleWeightings):
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

        self.loss_fn = ctc_loss(
            labels=tf.cast(self.ctc_target, tf.int32),
            inputs=attack.victim.raw_logits,
            sequence_length=attack.batch.audios["ds_feats"]
        ) * self.weights


class CTCLossV2(Bases.SimpleWeightings):
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


class SoftmaxCTCLoss(Bases.SimpleWeightings):
    """
    Tensorflow implementatons of CTC Loss take the activations/logits as input and
    convert them to time major softmax values for us... but this means we can't take a
    partial derivative w.r.t. the softmax outputs (only the logits).
    This class fixes that problem.
    """

    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):
        super().__init__(
            attack, weight_settings=weight_settings, updateable=updateable
        )

        self.attack = attack

        self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            attack.placeholders.targets,
            attack.placeholders.target_lengths
        )

        self.loss_fn = ctc_mod(
            labels=attack.placeholders.targets,
            logits=attack.victim.raw_logits,
            label_length=attack.placeholders.target_lengths,
            logit_length=attack.batch.audios["ds_feats"],
            blank_index=-1,
            apply_softmax=True,
            logits_time_major=True,
        )


class _BaseCTCGradientsPath:
    @staticmethod
    def get_argmin_softmax_gradient(attack):
        with tf.GradientTape() as tape:
            tape.watch(attack.victim.logits)

            path_gradient_loss = ctc_mod(
                labels=attack.placeholders.targets,
                logits=attack.victim.logits,
                label_length=attack.placeholders.target_lengths,
                logit_length=attack.batch.audios["ds_feats"],
                blank_index=-1,
                logits_time_major=False,
                apply_softmax=False,

            )

        gradients = tape.gradient(
            path_gradient_loss,
            attack.victim.logits
        )

        argmin_grads = tf.argmin(gradients, axis=-1)

        # your softmax tensor is not batch major
        if argmin_grads.shape[0] != attack.batch.size:
            argmin_grads = tf.transpose(argmin_grads, [1, 0])

        return argmin_grads


class CWMaxMin(Bases.KappaGreedySearchTokenWeights, _BaseCTCGradientsPath):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False, use_softmax: bool = False):

        self.gradient_argmin = self.get_argmin_softmax_gradient(attack)

        super().__init__(
            attack,
            custom_target=self.gradient_argmin,
            use_softmax=use_softmax,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        kap = self.kappa[:, tf.newaxis]

        self.max_diff_abs = - self.target_logit + self.max_other_logit
        self.max_diff = tf.maximum(self.max_diff_abs, -kap) + kap

        self.loss_fn = tf.reduce_sum(self.max_diff * self.weights, axis=1)


class SumLogProbsForward(Bases.SimpleBeamSearchTokenWeights, _BaseCTCGradientsPath):
    def __init__(self, attack, weight_settings=(None, None), updateable: bool = False):

        # flip the increment round so it's actually doing the opposite but don't
        # make the scripts worry about it

        weight_settings = list(weight_settings)
        weight_settings[1] = 1 / weight_settings[1]
        weight_settings = tuple(weight_settings)

        self.gradient_argmin = self.get_argmin_softmax_gradient(attack)

        super().__init__(
            attack,
            custom_target=self.gradient_argmin,
            use_softmax=True,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        self.log_smax = tf.log(self.target_logit * self.weights)

        self.fwd_target_log_probs = tf.reduce_sum(
            self.log_smax, axis=-1
        )

        self.back_target_log_probs = tf.reduce_sum(
            tf.reverse(self.log_smax, axis=[-1]), axis=-1
        )

        self.loss_fn = - self.fwd_target_log_probs


GRADIENT_PATHS = {
    "cw": CWMaxMin,
    "sumlogprobs-fwd": SumLogProbsForward,
}

CTC = {
    "ctc": CTCLoss,
    "ctc2": CTCLossV2,
    "ctc-softmax": SoftmaxCTCLoss,
}