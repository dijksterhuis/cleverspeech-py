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
    ctc_loss,
    _scan
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


class DynamicCWMaxMin(Bases.SimpleWeightings):

    def cw_ctc_mod(self,
            labels, logits, label_length, logit_length, logits_time_major=True,
            unique=None, blank_index=None, name=None, apply_softmax=True
            ):
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
            logits = array_ops.concat(
                [
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

        # @custom_gradient.custom_gradient
        def cw_compute_ctc_loss(
                logits_t, labels_t, label_length_t, logit_length_t, *unique_t
                ):

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

            result = self.cw_ctc_loss_and_grad(apply_softmax=apply_softmax, **kwargs)
            return result
            # def grad(grad_loss):
            #     grad = [
            #         array_ops.reshape(grad_loss, [1, -1, 1]) * result[1]
            #     ]
            #     grad += [None] * (len(args) - len(grad))
            #     return grad
            #
            # return result[0], grad

        return cw_compute_ctc_loss(*args)

    def _forward_backward_log(self,
            state_trans_log_probs, initial_state_log_probs,
            final_state_log_probs, observed_log_probs,
            sequence_length
            ):
        """Forward-backward algorithm computed in log domain.

        Args:
          state_trans_log_probs: tensor of shape [states, states] or if different
            transition matrix per batch [batch_size, states, states]
          initial_state_log_probs: tensor of shape [batch_size, states]
          final_state_log_probs: tensor of shape [batch_size, states]
          observed_log_probs: tensor of shape [frames, batch_size, states]
          sequence_length: tensor of shape [batch_size]

        Returns:
          forward backward log probabilites: tensor of shape [frames, batch, states]
          log_likelihood: tensor of shape [batch_size]

        Raises:
          ValueError: If state_trans_log_probs has unknown or incorrect rank.
        """

        if state_trans_log_probs.shape.ndims == 2:
            perm = [1, 0]
        elif state_trans_log_probs.shape.ndims == 3:
            perm = [0, 2, 1]
        else:
            raise ValueError(
                "state_trans_log_probs rank must be known and == 2 or 3, is: %s" %
                state_trans_log_probs.shape.ndims
            )

        bwd_state_trans_log_probs = array_ops.transpose(
            state_trans_log_probs, perm
            )
        batch_size = _get_dim(observed_log_probs, 1)

        def _forward(state_log_prob, obs_log_prob):
            state_log_prob = array_ops.expand_dims(
                state_log_prob, axis=1
                )  # Broadcast.
            state_log_prob *= state_trans_log_probs
            state_log_prob = math_ops.reduce_max(state_log_prob, axis=-1)
            state_log_prob += obs_log_prob
            log_prob_sum = math_ops.reduce_max(
                state_log_prob, axis=-1, keepdims=True
            )
            state_log_prob -= log_prob_sum
            return state_log_prob

        fwd = _scan(
            _forward, observed_log_probs, initial_state_log_probs,
            inclusive=True
        )

        self.fwd = fwd
        self.obs = observed_log_probs
        self.init_lp = initial_state_log_probs

        def _backward(accs, elems):
            """Calculate log probs and cumulative sum masked for sequence length."""
            state_log_prob, cum_log_sum = accs
            obs_log_prob, mask = elems
            state_log_prob += obs_log_prob
            state_log_prob = array_ops.expand_dims(
                state_log_prob, axis=1
                )  # Broadcast.
            state_log_prob += bwd_state_trans_log_probs
            state_log_prob = math_ops.reduce_max(state_log_prob, axis=-1)

            log_prob_sum = math_ops.reduce_max(
                state_log_prob, axis=-1, keepdims=True
            )
            state_log_prob -= log_prob_sum

            cum_log_sum += array_ops.squeeze(log_prob_sum) * mask
            batched_mask = array_ops.expand_dims(mask, axis=1)
            out = state_log_prob * batched_mask
            out += final_state_log_probs * (1.0 - batched_mask)
            return out, cum_log_sum

        zero_log_sum = array_ops.zeros([batch_size])
        maxlen = _get_dim(observed_log_probs, 0)
        mask = array_ops.sequence_mask(sequence_length, maxlen, tf.dtypes.float32)
        mask = array_ops.transpose(mask, perm=[1, 0])

        bwd, cum_log_sum = _scan(
            _backward, (observed_log_probs, mask),
            (final_state_log_probs, zero_log_sum),
            reverse=True,
            inclusive=True
        )
        self.bwd = bwd

        fwd_bwd_log_probs = fwd[1:] #+ bwd[1:]
        fwd_bwd_log_probs_sum = math_ops.reduce_max(
            fwd_bwd_log_probs, axis=2, keepdims=True
        )
        fwd_bwd_log_probs -= fwd_bwd_log_probs_sum
        # fwd_bwd_log_probs += math_ops.log(array_ops.expand_dims(mask, axis=2))
        fwd_bwd_log_probs *= array_ops.expand_dims(mask, axis=2)

        # log_likelihood = bwd[0, :, 0] + cum_log_sum[0]
        log_likelihood = cum_log_sum[0]

        return fwd_bwd_log_probs, log_likelihood

    def cw_ctc_loss_and_grad(self,
            logits, labels, label_length, logit_length, apply_softmax=False,
            unique=None
            ):
        """
        Computes the actual loss and gradients.
        """

        num_labels = _get_dim(logits, 2)
        max_label_seq_length = _get_dim(labels, 1)

        # this line is what needed to change from tf implementation to use softmax
        # ilabel_log_probs = tf.log(logits)
        ilabel_log_probs = logits

        self.ilabel_init = ilabel_log_probs

        def _cw_ilabel_to_state(labels, num_labels, ilabel_log_probs):
            """Project ilabel log probs to state log probs."""

            num_label_states = _get_dim(labels, 1)
            blank = ilabel_log_probs[:, :, :1]
            blank = array_ops.tile(blank, [1, 1, num_label_states + 1])
            one_hot = array_ops.one_hot(labels, depth=num_labels)
            one_hot = array_ops.expand_dims(one_hot, axis=0)
            ilabel_log_probs = array_ops.expand_dims(ilabel_log_probs, axis=2)
            dummy = ilabel_log_probs * one_hot
            print("122", dummy)
            state_log_probs = math_ops.reduce_sum(
                ilabel_log_probs * one_hot, axis=3
            )
            state_log_probs = array_ops.concat([state_log_probs, blank], axis=2)
            return array_ops.pad(
                state_log_probs, [[0, 0], [0, 0], [1, 0]],
                constant_values=0.0
            )

        state_log_probs = _cw_ilabel_to_state(
            labels, num_labels, ilabel_log_probs
        )
        self.state_log_probs = state_log_probs

        def cw_ctc_state_trans(label_seq):
            """Compute CTC alignment model transition matrix.

            Args:
              label_seq: tensor of shape [batch_size, max_seq_length]

            Returns:
              tensor of shape [batch_size, states, states] with a state transition matrix
              computed for each sequence of the batch.
            """

            with ops.name_scope("ctc_state_trans"):
                label_seq = ops.convert_to_tensor(label_seq, name="label_seq")
                batch_size = _get_dim(label_seq, 0)
                num_labels_inner = _get_dim(label_seq, 1)

                num_label_states = num_labels_inner + 1
                num_states = 2 * num_label_states

                label_states = math_ops.range(num_label_states)
                blank_states = label_states + num_label_states

                # Start state to first label.
                start_to_label = [[1, 0]]

                # Blank to label transitions.
                blank_to_label = array_ops.stack(
                    [label_states[1:], blank_states[:-1]], 1
                    )

                # Label to blank transitions.
                label_to_blank = array_ops.stack(
                    [blank_states, label_states], 1
                    )

                # Scatter transitions that don't depend on sequence.
                indices = array_ops.concat(
                    [start_to_label, blank_to_label, label_to_blank],
                    0
                    )
                values = array_ops.ones([_get_dim(indices, 0)])
                trans = array_ops.scatter_nd(
                    indices, values, shape=[num_states, num_states]
                )
                trans += tf.eye(num_states)  # Self-loops.

                # Label to label transitions. Disallow transitions between repeated labels
                # with no blank state in between.
                batch_idx = array_ops.zeros_like(label_states[2:])
                indices = array_ops.stack(
                    [batch_idx, label_states[2:], label_states[1:-1]],
                    1
                    )
                indices = array_ops.tile(
                    array_ops.expand_dims(indices, 0), [batch_size, 1, 1]
                )
                batch_idx = array_ops.expand_dims(
                    math_ops.range(batch_size), 1
                    ) * [1, 0, 0]
                indices += array_ops.expand_dims(batch_idx, 1)
                repeats = math_ops.equal(label_seq[:, :-1], label_seq[:, 1:])
                values = 1.0 - math_ops.cast(repeats, tf.dtypes.float32)
                batched_shape = [batch_size, num_states, num_states]
                label_to_label = array_ops.scatter_nd(
                    indices, values, batched_shape
                    )

                # return array_ops.expand_dims(trans, 0) + label_to_label
                return label_to_label


        state_trans_probs = cw_ctc_state_trans(labels)
        self.state_trans_probs = state_trans_probs


        tmp = []
        print(state_trans_probs.shape.as_list())
        # for i in range(state_trans_probs.shape.as_list()[-1]):
        d = state_trans_probs * state_log_probs
        # tmp.append(d)
        # stacked = tf.stack(tmp)
        print(state_trans_probs)
        print(state_log_probs)
        print("1111", d.shape)
        # self.d = tf.transpose(d, [0, 2, 1])
        self.d = d
        self.d2 = tf.reduce_sum(self.d, axis=-1)
        self.d3 = tf.reduce_max(self.d2, axis=-1)

        inverse_state_trans = tf.where(
            tf.equal(state_trans_probs, tf.ones_like(state_trans_probs)),
            tf.zeros_like(state_trans_probs),
            tf.ones_like(state_trans_probs),
        )

        others = inverse_state_trans * state_log_probs
        self.others = others
        self.others_reduced = tf.reduce_max(
            tf.reduce_max(
                others,
                axis=-1
            ),
            axis=-1
        )

        self.others_tmp = tf.reduce_max(logits, axis=-1)
        print("32222", self.others.shape)

        loss = tf.reduce_sum(tf.maximum(self.others_tmp - self.d3, 0), axis=-1)

        def cw_ctc_state_log_probs(seq_lengths, max_seq_length):
            """Computes CTC alignment initial and final state log probabilities.

            Create the initial/final state values directly as log values to avoid
            having to take a float64 log on tpu (which does not exist).

            Args:
              seq_lengths: int tensor of shape [batch_size], seq lengths in the batch.
              max_seq_length: int, max sequence length possible.

            Returns:
              initial_state_log_probs, final_state_log_probs
            """

            batch_size = _get_dim(seq_lengths, 0)
            num_label_states = max_seq_length + 1
            num_duration_states = 2
            num_states = num_duration_states * num_label_states
            # log_0 = math_ops.cast(
            #     math_ops.log(math_ops.cast(0, tf.dtypes.float64) + 1e-307),
            #     tf.dtypes.float32
            # )
            log_0 = math_ops.cast(
                math_ops.cast(0, tf.dtypes.float64) + 1e-10,
                tf.dtypes.float32
            )

            initial_state_log_probs = array_ops.one_hot(
                indices=array_ops.zeros([batch_size], dtype=tf.dtypes.int32),
                depth=num_states,
                on_value=1.0,
                off_value=log_0,
                axis=1
            )

            label_final_state_mask = array_ops.one_hot(
                seq_lengths, depth=num_label_states, axis=0
            )
            duration_final_state_mask = array_ops.ones(
                [num_duration_states, 1, batch_size]
            )
            final_state_mask = duration_final_state_mask * label_final_state_mask
            final_state_log_probs = (1.0 - final_state_mask) * log_0
            final_state_log_probs = array_ops.reshape(
                final_state_log_probs,
                [num_states, batch_size]
            )

            return initial_state_log_probs, array_ops.transpose(
                final_state_log_probs
            )

        initial_state_log_probs, final_state_log_probs = cw_ctc_state_log_probs(
            label_length, max_label_seq_length
        )

        self.initial_state_log_probs = initial_state_log_probs
        self.final_state_log_probs = final_state_log_probs

        fwd_bwd_log_probs, log_likelihood = self._forward_backward_log(
            # state_trans_log_probs=math_ops.log(state_trans_probs),
            state_trans_log_probs=state_trans_probs,
            initial_state_log_probs=initial_state_log_probs,
            final_state_log_probs=final_state_log_probs,
            observed_log_probs=state_log_probs,
            sequence_length=logit_length
        )

        self.fwd_bwd_log_probs, self.log_likelihood = fwd_bwd_log_probs, log_likelihood


        if unique:
            olabel_log_probs = _state_to_olabel_unique(
                labels, num_labels, fwd_bwd_log_probs, unique
            )
        else:
            olabel_log_probs = _state_to_olabel(
                labels, num_labels, fwd_bwd_log_probs
            )

        # grad = math_ops.exp(ilabel_log_probs) - math_ops.exp(olabel_log_probs)
        grad = tf.identity(ilabel_log_probs)  # - olabel_log_probs)

        # revert gradient calculations back to softmax if required
        if not apply_softmax:
            # TODO: the equations in the below references do not match up. need time
            #       to work through them.
            #       -
            #       http://www.cs.toronto.edu/%7Egraves/icml_2006.pdf
            #       http://www.cs.toronto.edu/~graves/preprint.pdf
            pass

        not_loss = -log_likelihood

        # return loss, grad
        return loss

    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack, weight_settings=weight_settings, updateable=updateable
        )

        self.loss_fn = self.cw_ctc_mod(
            labels=attack.placeholders.targets,
            logits=attack.victim.logits,
            label_length=attack.placeholders.target_lengths,
            logit_length=attack.batch.audios["ds_feats"],
            blank_index=-1,
            apply_softmax=False,
            logits_time_major=False,
        )
        self.loss_fn *= self.weights


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


class _BaseMinimumEnergy:
    @staticmethod
    def init_path_search_graph(attack):

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
            minimum_window_arg = tf.argmin(stacked_energy_windows)

            start_pad_shape = minimum_window_arg
            end_pad_shape = actual_features - (minimum_window_arg + target_length)

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

            unstacked_paths.append(path_concat)

        return tf.stack(unstacked_paths, axis=0)


class CWMaxMinMinimumEnergy(Bases.SimpleGreedySearchTokenWeights, _BaseMinimumEnergy):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        self.target_path = self.init_path_search_graph(attack)

        super().__init__(
            attack,
            custom_target=self.target_path,
            use_softmax=False,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        kap = tf.constant(0.0, dtype=tf.float32)

        self.max_diff_abs = - self.target_logit + self.max_other_logit
        self.max_diff = tf.maximum(self.max_diff_abs, -kap) + kap

        self.loss_fn = tf.reduce_sum(self.max_diff * self.weights, axis=1)


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