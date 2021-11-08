import tensorflow as tf
from cleverspeech.graph.losses import Bases
from cleverspeech.utils.Utils import l_map
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

        result = ctc_loss_and_grad(apply_softmax=apply_softmax, **kwargs)

        def grad(grad_loss):
            grad = [
                array_ops.reshape(grad_loss, [1, -1, 1]) * result[1]
            ]
            grad += [None] * (len(args) - len(grad))
            return grad

        return result[0], grad

    return compute_ctc_loss(*args)


def ctc_loss_and_grad(logits, labels, label_length, logit_length, apply_softmax=False, unique=None):
    """
    Computes the actual loss and gradients.
    """

    num_labels = _get_dim(logits, 2)
    max_label_seq_length = _get_dim(labels, 1)

    # this line is what needed to change from tf implementation to use softmax
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

    # revert gradient calculations back to softmax if required
    if not apply_softmax:
        # TODO: the equations in the below references do not match up. need time
        #       to work through them.
        #       -
        #       http://www.cs.toronto.edu/%7Egraves/icml_2006.pdf
        #       http://www.cs.toronto.edu/~graves/preprint.pdf
        pass

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
    Tensorflow implementatons of CTC Loss take the activations/logits as input
    and convert them to time major softmax values for us... but this means we
    can't take a partial derivative w.r.t. softmax outputs (only the logits).
    This class fixes that problem.
    """

    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):
        super().__init__(
            attack, weight_settings=weight_settings, updateable=updateable
        )

        self.loss_fn = ctc_mod(
            labels=attack.placeholders.targets,
            logits=attack.victim.logits,
            label_length=attack.placeholders.target_lengths,
            logit_length=attack.batch.audios["ds_feats"],
            blank_index=-1,
            apply_softmax=False,
            logits_time_major=False,
        )
        self.loss_fn *= self.weights


class _BaseCTCGradientsPath:
    @staticmethod
    def get_argmin_softmax_gradient(attack):

        with tf.GradientTape() as tape:

            tape.watch(attack.victim.logits)

            # ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            #     attack.placeholders.targets,
            #     attack.placeholders.target_lengths
            # )
            #
            # loss_fn = ctc_loss(
            #     labels=tf.cast(ctc_target, tf.int32),
            #     inputs=grad_var,
            #     sequence_length=attack.batch.audios["ds_feats"]
            # )

            loss_fn = ctc_mod(
                labels=attack.placeholders.targets,
                logits=attack.victim.logits,
                label_length=attack.placeholders.target_lengths,
                logit_length=attack.batch.audios["ds_feats"],
                blank_index=-1,
                logits_time_major=False,
                apply_softmax=False,
            )

        gradients = tape.gradient(
            loss_fn,
            attack.victim.logits
        )

        # your tensor is not batch major
        if gradients.shape[0] != attack.batch.size:
            gradients = tf.transpose(gradients, [1, 0, 2])

        return tf.argmin(gradients, axis=-1), gradients

    @staticmethod
    def get_argmin_activations_gradient(attack):

        with tf.GradientTape() as tape:

            tape.watch(attack.victim.raw_logits)

            ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
                attack.placeholders.targets,
                attack.placeholders.target_lengths
            )

            loss_fn = ctc_loss(
                labels=tf.cast(ctc_target, tf.int32),
                inputs=attack.victim.raw_logits,
                sequence_length=attack.batch.audios["ds_feats"]
            )

            # loss_fn = ctc_mod(
            #     labels=attack.placeholders.targets,
            #     logits=attack.victim.raw_logits,
            #     label_length=attack.placeholders.target_lengths,
            #     logit_length=attack.batch.audios["ds_feats"],
            #     blank_index=-1,
            #     logits_time_major=True,
            #     apply_softmax=True,
            # )

        gradients = tape.gradient(
            loss_fn,
            attack.victim.raw_logits
        )
        # your tensor is not batch major
        if gradients.shape[0] != attack.batch.size:
            gradients = tf.transpose(gradients, [1, 0, 2])

        return tf.argmin(gradients, axis=-1), gradients


class CWMaxMin(Bases.KappaGreedySearchTokenWeights, _BaseCTCGradientsPath):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        self.gradient_argmin, self.gradients = self.get_argmin_activations_gradient(attack)

        super().__init__(
            attack,
            custom_target=self.gradient_argmin,
            use_softmax=False,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        kap = self.kappa[:, tf.newaxis]

        self.max_diff_abs = - self.target_logit + self.max_other_logit
        self.max_diff = tf.maximum(self.max_diff_abs, -kap) + kap

        self.loss_fn = tf.reduce_sum(self.max_diff * self.weights, axis=1)


class SumLogProbsForward(Bases.SimpleGreedySearchTokenWeights, _BaseCTCGradientsPath):
    def __init__(self, attack, weight_settings=(None, None), updateable: bool = False):

        # flip the increment round so it's actually doing the opposite but don't
        # make the scripts worry about it

        weight_settings = list(weight_settings)
        weight_settings[1] = 1 / weight_settings[1]
        weight_settings = tuple(weight_settings)

        self.gradient_argmin, _ = self.get_argmin_activations_gradient(attack)

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


class _BaseMinimumEnergy(Bases.BaseAlignmentLoss):
    def init_path_search_graph(self, attack, n_paths=10):

        unstacked_examples = tf.unstack(attack.adversarial_examples, axis=0)
        unstacked_paths = []

        for idx, example in enumerate(unstacked_examples):

            frame_size = int(0.032 * 16000)
            frame_step = int(0.02 * 16000)

            frames = tf.signal.frame(
                example,
                frame_size,
                frame_step,
                pad_end=True,
            )

            target_length = attack.batch.targets["lengths"][idx]
            actual_features = attack.batch.audios["real_feats"][idx]
            max_features = attack.batch.audios["max_feats"]

            available_positions = actual_features - target_length

            def tf_energy(x):
                return tf.reduce_sum(tf.square(tf.abs(x)))

            energy_per_window = l_map(
                lambda i: tf_energy(frames[i: i + target_length]),
                range(available_positions)
            )

            stacked_energy_windows = tf.stack(energy_per_window)

            top_ten = tf.argsort(stacked_energy_windows)[:n_paths]

            top_ten = tf.unstack(top_ten, axis=-1)

            top_ns = []
            for top_n in top_ten:

                start_pad_shape = top_n
                end_pad_shape = actual_features - (top_n + target_length)

                start_pad = 28 * tf.ones(start_pad_shape, dtype=tf.int32)
                end_pad = 28 * tf.ones(end_pad_shape, dtype=tf.int32)

                # FIXME: we shouldn't need select elems up to actual_features, but
                #  for some reason we get weird padding lengths if we don't.
                #  ==> Could be to do with shape of signal frames?

                path_concat = tf.concat(
                    [start_pad, attack.placeholders.targets[idx], end_pad],
                    axis=0
                )[0:actual_features]

                max_pad_shape = max_features - actual_features
                max_pad = 28 * tf.ones(max_pad_shape, dtype=tf.int32)

                path_concat = tf.concat(
                    [path_concat, max_pad],
                    axis=0
                )
                top_ns.append(path_concat)
            top_ten_padded = tf.stack(top_ns, axis=-1)

            unstacked_paths.append(top_ten_padded)
        stacked_paths = tf.stack(unstacked_paths, axis=0)

        return stacked_paths


class CWMaxMinMinimumEnergy(_BaseMinimumEnergy, Bases.SimpleWeightings):
    def __init__(self, attack, n_paths=10, weight_settings=(1.0, 1.0), updateable: bool = False):

        self.target_path = self.init_path_search_graph(attack, n_paths=n_paths)

        super().__init__(
            attack,
            custom_target=self.target_path,
            use_softmax=False,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        kap = tf.constant(0.0, dtype=tf.float32)

        self.maximise = self.target_logit
        self.minimise = self.max_other_logit

        self.path_wise_min_max = - self.maximise + self.minimise

        top_n_weights = tf.range(n_paths + 1, 1, delta=-1, dtype=tf.float32)
        top_n_weights *= tf.cast(1.0 / n_paths, dtype=tf.float32)
        # top_n_weights *= tf.constant(2.0)
        top_n_weights = tf.exp(top_n_weights)

        self.clamped = tf.maximum(self.path_wise_min_max, -kap) + kap
        self.min_max = tf.reduce_sum(self.clamped * top_n_weights, axis=-1)
        self.loss_fn = tf.reduce_sum(self.min_max, axis=1) * self.weights


GRADIENT_PATHS = {
    "cw": CWMaxMin,
    "cw-nrg": CWMaxMinMinimumEnergy,
    "sumlogprobs-fwd": SumLogProbsForward,
}

CTC = {
    "ctc": CTCLoss,
    "ctc2": CTCLossV2,
    "ctc-softmax": SoftmaxCTCLoss,
}