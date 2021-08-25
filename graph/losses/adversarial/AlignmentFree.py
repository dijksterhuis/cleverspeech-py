import tensorflow as tf
from cleverspeech.graph.losses import Bases

from tensorflow_core.python.framework import ops
from tensorflow_core.python.ops.ctc_ops import (
    _ctc_state_trans,
    _get_dim,
    ctc_state_log_probs,
    _forward_backward_log,
    _state_to_olabel_unique,
    _ilabel_to_state
)
from tensorflow_core.python.ops import (
    nn_ops,
    array_ops,
    math_ops,
    custom_gradient
)


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

        self.loss_fn = tf.nn.ctc_loss(
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

        self.loss_fn = self.ctc(
            labels=attack.placeholders.targets,
            logits=attack.victim.raw_logits,
            label_length=attack.placeholders.target_lengths,
            logit_length=attack.batch.audios["ds_feats"],
            blank_index=-1,
            apply_softmax=True,
            logits_time_major=True,
        )

    def ctc(self, labels, logits, label_length, logit_length, logits_time_major=True, unique=None, blank_index=None, name=None, apply_softmax=True):
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
        def compute_ctc_loss(logits_t, labels_t, label_length_t, logit_length_t, *unique_t):

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

            result = self.ctc_loss_and_grad(**kwargs)

            def grad(grad_loss):
                grad = [
                    array_ops.reshape(grad_loss, [1, -1, 1]) * result[1]
                ]
                grad += [None] * (len(args) - len(grad))
                return grad

            return result[0], grad

        return compute_ctc_loss(*args)

    def ctc_loss_and_grad(self, logits, labels, label_length, logit_length, unique=None):

        """
        Computes the actual loss and gradients.
        """

        num_labels = _get_dim(logits, 2)
        max_label_seq_length = _get_dim(labels, 1)

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
            # olabel_log_probs = _state_to_olabel(labels, num_labels,
            #                                     fwd_bwd_log_probs)
            olabel_log_probs = self.custom_state_to_olabel_max(
                labels, num_labels, fwd_bwd_log_probs
            )

        grad = math_ops.exp(ilabel_log_probs) - math_ops.exp(olabel_log_probs)
        loss = -log_likelihood
        return loss, grad

    @staticmethod
    def custom_state_to_olabel_max(labels, num_labels, states):
        """
        Sum state log probs to ilabel log probs.
        """

        num_label_states = _get_dim(labels, 1) + 1
        label_states = states[:, :, 1:num_label_states]
        blank_states = states[:, :, num_label_states:]

        one_hot = array_ops.one_hot(
            labels - 1,
            depth=(num_labels - 1),
            on_value=0.0,
            off_value=math_ops.log(0.0)
        )
        one_hot = array_ops.expand_dims(one_hot, axis=0)

        label_states = array_ops.expand_dims(label_states, axis=3)

        label_olabels = math_ops.reduce_logsumexp(
            label_states + one_hot, axis=2
        )
        blank_olabels = math_ops.reduce_logsumexp(
            blank_states, axis=2, keepdims=True
        )

        return array_ops.concat([blank_olabels, label_olabels], axis=-1)


class AlignmentFreeCWMaxMin(Bases.KappaGreedySearchTokenWeights, SoftmaxCTCLoss):
    def __init__(self, attack, k=0.0, weight_settings=(1.0, 1.0), updateable: bool = False, use_softmax: bool = False):

        assert k >= 0

        if use_softmax is True:
            watch_gradient_var = attack.victim.logits
            time_major = False
            apply_softmax = False

        else:

            watch_gradient_var = attack.victim.logits
            time_major = False
            apply_softmax = False

        with tf.GradientTape() as tape:

            tape.watch(watch_gradient_var)

            path_gradient_loss = self.ctc(
                labels=attack.placeholders.targets,
                logits=watch_gradient_var,
                label_length=attack.placeholders.target_lengths,
                logit_length=attack.batch.audios["ds_feats"],
                blank_index=-1,
                logits_time_major=time_major,
                apply_softmax=apply_softmax,

            )

        gradients = tape.gradient(
            path_gradient_loss,
            watch_gradient_var
        )
        argmin_grads = tf.argmin(gradients, axis=-1)

        self.grad_logits = gradients

        if argmin_grads.shape[-1] == attack.batch.size:
            self.gradient_argmin = tf.transpose(argmin_grads, [1, 0])
        else:
            self.gradient_argmin = argmin_grads

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

