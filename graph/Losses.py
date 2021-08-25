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

from abc import ABC, abstractmethod
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
from cleverspeech.utils.Utils import log
import numpy as np


class BaseLoss(ABC):

    def __init__(self, attack, weight_settings: tuple = (None, None), updateable: bool = False):
        assert type(updateable) is bool

        self.updateable = updateable
        self.attack = attack
        self.weights = None
        self.initial = None
        self.increment = None

    @abstractmethod
    def init_weights(self, attack, weight_settings):
        pass

    @abstractmethod
    def update_one(self, idx: int):
        pass

    @abstractmethod
    def update_many(self, batch_successes: list):
        pass


class SimpleWeightings(BaseLoss):

    def __init__(self, attack, weight_settings: tuple = (None, None), updateable: bool = False):

        assert type(weight_settings) in [list, tuple]
        assert all(type(t) in [float, int] for t in weight_settings)
        assert len(weight_settings) == 2

        self.lower_bound = None
        self.upper_bound = None
        self.weights = None
        self.initial = None
        self.increment = None

        super().__init__(attack, updateable=updateable)

        self.init_weights(attack, weight_settings)

    def init_weights(self, attack, weight_settings):

        weight_settings = [float(s) for s in weight_settings]

        if len(weight_settings) == 2:

            self.initial, self.increment = weight_settings
            self.upper_bound = self.initial

        elif len(weight_settings) == 3:
            self.initial, self.increment, n = weight_settings
            # never be more than N steps away from initial value
            self.upper_bound = self.initial
            self.lower_bound = self.initial * ((1 / self.increment) ** n)

        else:
            raise ValueError

        self.weights = tf.Variable(
            tf.ones(attack.batch.size, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        initial_vals = self.initial * np.ones([attack.batch.size], dtype=np.float32)
        attack.sess.run(self.weights.assign(initial_vals))

    def update_one(self, idx: int):
        raise NotImplementedError
        pass

    def update_many(self, batch_successes: list):

        w = self.attack.sess.run(self.weights)

        incr, upper, lower = self.increment, self.upper_bound, self.lower_bound

        for idx, success_check in enumerate(batch_successes):
            if success_check is True:
                w[idx] = w[idx] * incr if w[idx] * incr > lower else lower
            else:
                w[idx] = w[idx] / incr if w[idx] / incr < upper else upper

        self.attack.sess.run(self.weights.assign(w))

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from cleverspeech.graph.losses import Bases


class L2CarliniLoss(Bases.SimpleWeightings):
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

        l2delta = tf.reduce_mean(attack.delta_graph.perturbations ** 2, axis=1)
        self.loss_fn = l2delta * self.weights


class L2SquaredLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        l2delta = tf.reduce_sum(attack.delta_graph.perturbations ** 2, axis=-1)
        self.loss_fn = l2delta * self.weights


class L2TanhSquaredLoss(Bases.SimpleWeightings):
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

        l2delta = tf.reduce_sum(
            tf.tanh(attack.delta_graph.final_deltas / 2**15) ** 2, axis=-1
        )
        self.loss_fn = l2delta * self.weights


class L2TanhCarliniLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        l2delta = tf.reduce_mean(
            tf.tanh(attack.delta_graph.final_deltas / 2**15) ** 2, axis=-1
        )
        self.loss_fn = l2delta * self.weights


class L2Log10Loss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        def log10(x):
            numerator = tf.log(x + 1e-8)
            denominator = tf.log(tf.constant(10.0, dtype=tf.float32))
            return tf.cast(tf.cast(numerator / denominator, dtype=tf.int32), dtype=tf.float32)

        l2delta = tf.reduce_sum(attack.perturbations ** 2, axis=-1)
        self.loss_fn = l2delta / tf.pow(10.0, log10(l2delta) + 1)
        self.loss_fn *= self.weights


class L2SampleLoss(Bases.SimpleWeightings):
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


class LinfLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        l2delta = tf.reduce_max(tf.abs(attack.delta_graph.perturbations), axis=1)
        self.loss_fn = l2delta * self.weights


class LinfTanhLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        l2delta = tf.reduce_max(
            tf.abs(tf.tanh(attack.delta_graph.final_deltas / 2**15)),
            axis=1
        )
        self.loss_fn = l2delta * self.weights


class RootMeanSquareLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        l2delta = tf.sqrt(tf.reduce_mean(attack.perturbations ** 2, axis=-1))
        self.loss_fn = l2delta * self.weights


class EntropyLoss(Bases.SimpleWeightings):
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


class BasePathsLoss(SimpleWeightings):
    """
    Base class that can be used for logits difference losses, like CW f_6
    and the adaptive kappa variant.

    :param: attack: an attack class.
    :param: target_argmax: frame length vector of desired target class indices
    :param: softmax: bool type. whether to use softmax or activations
    :param: weight_settings: how to update this loss function on success
    """

    def __init__(self, attack, custom_target=None, use_softmax=False, weight_settings=(None, None), updateable: bool = False):

        self.use_softmax = use_softmax

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        # Indices of the specified alignment per frame
        if custom_target is None:
            self.target_argmax = attack.placeholders.targets
        else:
            self.target_argmax = custom_target

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



class CTCLoss(SimpleWeightings):
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


class CTCLossV2(SimpleWeightings):
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


class SoftmaxCTCLoss(SimpleWeightings):
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




class GreedyPathTokenWeightingBinarySearch(BasePathsLoss):

    def init_weights(self, attack, weight_settings):

        weight_settings = [float(setting) for setting in weight_settings]
        self.initial, self.increment = weight_settings

        if self.use_softmax is False:
            self.upper_bound = self.initial
        else:
            self.upper_bound = 1.0

        # never be more than N checks away from initial value
        n = 5
        self.lower_bound = self.initial * (self.increment ** n)

        self.kappa = tf.Variable(
            tf.zeros(attack.batch.size, dtype=tf.float32),
        )
        self.attack.sess.run(
            self.kappa.assign(0.0 * np.zeros(attack.batch.size)))

        shape = [attack.batch.size, attack.batch.audios["ds_feats"][0]]

        self.weights = tf.Variable(
            tf.ones(shape, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        initial_vals = self.initial * np.ones(shape, dtype=np.float32)
        attack.sess.run(self.weights.assign(initial_vals))

    def update_one(self, idx: int):
        raise Exception

    def update_many(self, batch_successes: list):
        """
        Apply the loss weighting updates to only one example in a batch.

        :param batch_successes: a list of True/False false indicating whether
            the loss weighting should be updated for each example in a batch
        """

        # current_entropy = - tf.reduce_sum(
        #     self.attack.victim.logits * tf.log(self.attack.victim.logits),
        #     axis=-1
        # )
        smax = self.attack.victim.logits

        kaps, losses, msamx = self.attack.procedure.tf_run(
            [self.kappa, self.attack.loss[0].loss_fn, smax]
        )
        z = zip(batch_successes, kaps, losses)
        for idx, (suc, k, l) in enumerate(z):
            s = "step: {i} {s}, {k}, {l}".format(
                i=self.attack.procedure.current_step,
                s=suc, k=k, l=l
            )

            s += "\nsmax max: {}".format(  # mean: {} min: {}".format(
                np.max(msamx, axis=-1)
                # , np.mean(msamx, axis=-1), np.min(msamx, axis=-1)
            )
            if suc is False and l <= 0:
                kaps[idx] += 1

                s += "\nIncreased kappa by 1 to {k} for sample {i}".format(
                    k=kaps[idx], i=self.attack.batch.audios["basenames"][idx]
                )
            log(
                s,
                wrap=False,
                stdout=False,
                outdir=self.attack.settings["outdir"],
                fname="kappa_updates.txt"
            )

        kaps_assign = self.kappa.assign(kaps)
        self.attack.sess.run(kaps_assign)

        current_argmax = tf.cast(
            tf.argmax(self.current, axis=-1), dtype=tf.int32
        )

        target_argmax = tf.cast(
            self.target_argmax, dtype=tf.int32
        )

        argmax_test = tf.where(
            tf.equal(target_argmax, current_argmax),
            tf.multiply(self.increment, self.weights),
            tf.multiply(1 / self.increment, self.weights),
        )
        upper_bound = tf.ones_like(argmax_test,
                                   dtype=tf.float32) * self.upper_bound
        lower_bound = tf.ones_like(argmax_test,
                                   dtype=tf.float32) * self.lower_bound

        max_clamp = tf.where(
            tf.greater(argmax_test, upper_bound),
            upper_bound,
            argmax_test,
        )
        new_weights = tf.where(
            tf.less(max_clamp, lower_bound),
            lower_bound,
            max_clamp,
        )

        new_weights = self.attack.procedure.tf_run(new_weights)

        self.attack.sess.run(self.weights.assign(new_weights))




class CWMaxMinWithCTCGrads(GreedyPathTokenWeightingBinarySearch):
    def __init__(self, attack, k=0.0, weight_settings=(1.0, 1.0), updateable: bool = False, use_softmax: bool = False):

        assert k >= 0

        ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            attack.placeholders.targets,
            attack.placeholders.target_lengths
        )

        if use_softmax is True:
            watch_gradient_var = attack.victim.logits
            time_major = False
            apply_softmax = False
            print(watch_gradient_var.shape)

        else:

            watch_gradient_var = attack.victim.logits
            print(watch_gradient_var.shape)
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
        print(gradients)
        #gradients = tf.transpose(gradients, [1, 0, 2])
        print(gradients)
        argmin_grads = tf.argmin(gradients, axis=-1)
        print(argmin_grads)
        self.grad_logits = gradients
        if argmin_grads.shape[-1] == attack.batch.size:
            print("lkdsj\f")
            self.gradient_argmin = tf.transpose(argmin_grads, [1, 0])
        else:
            self.gradient_argmin = argmin_grads

        print(self.gradient_argmin.shape)

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
        self.loss_fn = self.loss_fn

    def ctc(self, labels, logits, label_length, logit_length,
            logits_time_major=True, unique=None, blank_index=None, name=None,
            apply_softmax=True):
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

            result = self.ctc_loss_and_grad(**kwargs)

            def grad(grad_loss):
                grad = [
                    array_ops.reshape(grad_loss, [1, -1, 1]) * result[1]]
                grad += [None] * (len(args) - len(grad))
                return grad

            return result[0], grad

        return compute_ctc_loss(*args)

    def ctc_loss_and_grad(self, logits, labels, label_length, logit_length,
            unique=None):

        """
        Computes the actual loss and gradients.
        """

        num_labels = _get_dim(logits, 2)
        max_label_seq_length = _get_dim(labels, 1)

        ilabel_log_probs = tf.log(logits)
        state_log_probs = _ilabel_to_state(labels, num_labels, ilabel_log_probs)
        state_trans_probs = _ctc_state_trans(labels)
        initial_state_log_probs, final_state_log_probs = ctc_state_log_probs(
            label_length, max_label_seq_length)
        fwd_bwd_log_probs, log_likelihood = _forward_backward_log(
            state_trans_log_probs=math_ops.log(state_trans_probs),
            initial_state_log_probs=initial_state_log_probs,
            final_state_log_probs=final_state_log_probs,
            observed_log_probs=state_log_probs,
            sequence_length=logit_length)

        if unique:
            olabel_log_probs = _state_to_olabel_unique(labels, num_labels,
                                                       fwd_bwd_log_probs,
                                                       unique)
        else:
            # olabel_log_probs = _state_to_olabel(labels, num_labels,
            #                                     fwd_bwd_log_probs)
            olabel_log_probs = self.custom_state_to_olabel_max(labels,
                                                               num_labels,
                                                               fwd_bwd_log_probs)

        grad = math_ops.exp(ilabel_log_probs) - math_ops.exp(olabel_log_probs)
        loss = -log_likelihood
        return loss, grad

    @staticmethod
    def custom_state_to_olabel_max(labels, num_labels, states):
        """Sum state log probs to ilabel log probs."""

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



class SuperGreedyPathTokenWeightingBinarySearch(BasePathsLoss):

    def init_weights(self, attack, weight_settings):

        weight_settings = [float(setting) for setting in weight_settings]
        self.initial, self.increment = weight_settings

        if self.use_softmax is False:
            self.upper_bound = self.initial
        else:
            self.upper_bound = 1.0

        self.lower_bound = 1.0e-6

        self.super_upper_bound = self.initial
        n = 25
        self.super_lower_bound = self.initial * (self.increment ** n)

        # never be more than N checks away from initial value
        n = 25
        self.lower_bound = self.initial * (self.increment ** n)

        self.super_weights = tf.Variable(
            tf.ones(attack.batch.size, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        shape = [attack.batch.size, attack.batch.audios["ds_feats"][0]]

        self.weights = tf.Variable(
            tf.ones(shape, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        initial_vals = self.upper_bound * np.ones(shape, dtype=np.float32)
        super_vals = self.super_upper_bound * np.ones(attack.batch.size,
                                                      dtype=np.float32)

        attack.sess.run([
            self.weights.assign(initial_vals),
            self.super_weights.assign(super_vals),
        ])

    def update_one(self, idx: int):
        raise Exception

    def update_many(self, batch_successes: list):

        current_argmax = tf.cast(
            tf.argmax(self.current, axis=-1), dtype=tf.int32
        )

        target_argmax = tf.cast(
            self.target_argmax, dtype=tf.int32
        )
        # swap around vs. simple greedy weightings
        argmax_test = tf.where(
            tf.equal(target_argmax, current_argmax),
            tf.multiply(self.increment, self.weights),
            tf.multiply(1 / self.increment, self.weights),
        )
        upper_bound = tf.ones_like(argmax_test,
                                   dtype=tf.float32) * self.upper_bound
        lower_bound = tf.ones_like(argmax_test,
                                   dtype=tf.float32) * self.lower_bound

        max_clamp = tf.where(
            tf.greater(argmax_test, upper_bound),
            upper_bound,
            argmax_test,
        )
        new_weights = tf.where(
            tf.less(max_clamp, lower_bound),
            lower_bound,
            max_clamp,
        )

        new_weights, super_weights = self.attack.procedure.tf_run([
            new_weights, self.super_weights
        ])
        # print("loss_calc", new_weights)

        for idx, success_check in enumerate(batch_successes):
            if success_check is True:
                test_weight = super_weights[idx] * self.increment
                if test_weight < self.super_lower_bound:
                    test_weight = self.super_lower_bound
            else:

                test_weight = super_weights[idx] / self.increment
                if test_weight > self.super_upper_bound:
                    test_weight = self.super_upper_bound

            super_weights[idx] = test_weight

        # print("loss calc supers", super_weights)

        self.attack.sess.run([
            self.weights.assign(new_weights),
            self.super_weights.assign(super_weights)
        ])


class SuperBeamSearchPathTokenWeightingBinarySearch(BasePathsLoss):

    def init_weights(self, attack, weight_settings):

        weight_settings = [float(setting) for setting in weight_settings]
        self.initial, self.increment = weight_settings

        if self.use_softmax is False:
            self.upper_bound = self.initial
        else:
            self.upper_bound = 1.0

        self.lower_bound = 1.0e-6

        self.super_upper_bound = self.initial
        n = 25
        self.super_lower_bound = self.initial * (self.increment ** n)

        # never be more than N checks away from initial value
        n = 25
        self.lower_bound = self.initial * (self.increment ** n)

        self.super_weights = tf.Variable(
            tf.ones(attack.batch.size, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        shape = [attack.batch.size, attack.batch.audios["ds_feats"][0]]

        self.weights = tf.Variable(
            tf.ones(shape, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        initial_vals = self.upper_bound * np.ones(shape, dtype=np.float32)
        super_vals = self.super_upper_bound * np.ones(attack.batch.size,
                                                      dtype=np.float32)

        attack.sess.run([
            self.weights.assign(initial_vals),
            self.super_weights.assign(super_vals),
        ])

    def update_one(self, idx: int):
        raise Exception

    def update_many(self, batch_successes: list):

        (
            labellings,
            probs,
            token_order,
            timestep_switches
        ) = self.attack.victim.ds_decode_batch_no_lm(
            self.attack.procedure.tf_run(
                self.attack.victim.logits
            ),
            self.attack.batch.audios["ds_feats"],
            top_five=False, with_metadata=True
        )

        blanks = np.ones(self.target_argmax.shape, dtype=np.int32) * 28

        for tok, time in zip(token_order, timestep_switches):
            blanks[0][time] = tok

        self.blanks = blanks

        # current_argmax = tf.cast(blanks, dtype=tf.int32)

        current_argmax = tf.cast(
            tf.argmax(self.current, axis=-1), dtype=tf.int32
        )

        target_argmax = tf.cast(
            self.target_argmax, dtype=tf.int32
        )
        # swap around vs. simple greedy weightings
        argmax_test = tf.where(
            tf.equal(target_argmax, current_argmax),
            tf.multiply(self.increment, self.weights),
            tf.multiply(1 / self.increment, self.weights),
        )
        upper_bound = tf.ones_like(argmax_test,
                                   dtype=tf.float32) * self.upper_bound
        lower_bound = tf.ones_like(argmax_test,
                                   dtype=tf.float32) * self.lower_bound

        max_clamp = tf.where(
            tf.greater(argmax_test, upper_bound),
            upper_bound,
            argmax_test,
        )
        new_weights = tf.where(
            tf.less(max_clamp, lower_bound),
            lower_bound,
            max_clamp,
        )

        new_weights, super_weights = self.attack.procedure.tf_run([
            new_weights, self.super_weights
        ])
        # print("loss_calc", new_weights)

        loss_value = self.attack.procedure.tf_run(self.attack.loss[0].loss_fn)

        for idx, (success_check, loss) in enumerate(
                zip(batch_successes, loss_value)):
            if success_check is False and loss <= 0:
                print("BAD SAMPLE {}".format(idx))
            #     # somethings gone wrong, re-ini the perturbation
            #
            #     print("RESETTING SAMPLE {idx}".format(idx=idx))
            #     assign_op = self.attack.delta_graph.raw_deltas[idx].assign(
            #         tf.random.uniform(
            #             [self.attack.batch.audios["max_samples"]],
            #             minval=1000,
            #             maxval=-1000,
            #             dtype=tf.float32
            #         ),
            #     )
            #     self.attack.sess.run(assign_op)

            if success_check is True:
                test_weight = super_weights[idx] * self.increment
                if test_weight < self.super_lower_bound:
                    test_weight = self.super_lower_bound
            else:

                test_weight = super_weights[idx] / self.increment
                if test_weight > self.super_upper_bound:
                    test_weight = self.super_upper_bound

            # perform a safety check

            super_weights[idx] = test_weight

        # print("loss calc supers", super_weights)

        self.attack.sess.run([
            self.weights.assign(new_weights),
            self.super_weights.assign(super_weights)
        ])



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


class CWMaxMinWithPerTokenWeights(GreedyPathTokenWeightingBinarySearch):
    """
    As above but with the binary search step for the c parameter
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
        self.loss_fn = tf.reduce_sum(
            tf.multiply(self.max_diff, self.weights), axis=1
        )


class AdaptiveKappaMaxMin(BasePathsLoss):
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


class SinglePathCTCLoss(SimpleWeightings):
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


class LogProbsTokenWeightingBinarySearch(BasePathsLoss):

    def init_weights(self, attack, weight_settings):

        weight_settings = [float(setting) for setting in weight_settings]
        self.initial, self.increment = weight_settings

        self.super_weights = tf.Variable(
            tf.ones(attack.batch.size, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )
        self.upper_bound = tf.cast(1.0, tf.float32)
        self.lower_bound = tf.cast(1.0e-6, tf.float32)


        # never be more than N checks away from initial value
        n = 25
        self.super_lower_bound = self.initial * (self.increment ** n)
        self.super_upper_bound = 1.0e3

        shape = [attack.batch.size, attack.batch.audios["ds_feats"][0]]

        self.weights = tf.Variable(
            tf.ones(shape, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        initial_vals = 1e-8 * np.ones(shape, dtype=np.float32)
        super_vals = self.super_upper_bound * np.ones(attack.batch.size, dtype=np.float32)

        attack.sess.run([
            self.weights.assign(initial_vals),
            self.super_weights.assign(super_vals),
        ])

    def update_one(self, idx: int):
        raise Exception

    def update_many(self, batch_successes: list):
        """
        Apply the loss weighting updates to only one example in a batch.

        :param batch_successes: a list of True/False false indicating whether
            the loss weighting should be updated for each example in a batch
        """

        # [!!!] DO NOT USE THE TENSORFLOW BEAM SEARCH DECODER
        # ==> it's very inefficient time wise.
        # for a batch of 100 examples it takes 7 seconds to perform decoding
        # while the DeepSpeech batch decoder only takes 2 or 3 seconds.

        (
            labellings,
            probs,
            token_order,
            timestep_switches
        ) = self.attack.victim.ds_decode_batch_no_lm(
            self.attack.procedure.tf_run(
                self.attack.victim.logits
            ),
            self.attack.batch.audios["ds_feats"],
            top_five=False, with_metadata=True
        )

        blanks = np.ones(self.target_argmax.shape, dtype=np.int32) * 28

        for tok, time in zip(token_order, timestep_switches):
            blanks[0][time] = tok

        self.blanks = blanks


        # current_argmax = tf.cast(
        #     tf.argmax(self.current, axis=-1), dtype=tf.int32
        # )

        current_argmax = tf.cast(blanks, dtype=tf.int32)

        target_argmax = tf.cast(
            self.target_argmax, dtype=tf.int32
        )
        # swap around vs. simple greedy weightings
        argmax_test = tf.where(
            tf.equal(target_argmax, current_argmax),
            tf.multiply(1/self.increment, self.weights),
            tf.multiply(self.increment, self.weights),
        )
        upper_bound = tf.ones_like(argmax_test, dtype=tf.float32) * self.upper_bound
        lower_bound = tf.ones_like(argmax_test, dtype=tf.float32) * self.lower_bound

        max_clamp = tf.where(
            tf.greater(argmax_test, upper_bound),
            upper_bound,
            argmax_test,
        )
        new_weights = tf.where(
            tf.less(max_clamp, lower_bound),
            lower_bound,
            max_clamp,
        )

        new_weights, super_weights = self.attack.procedure.tf_run([
            new_weights, self.super_weights
        ])
        # print("loss_calc", new_weights)

        for idx, success_check in enumerate(batch_successes):
            if success_check is True:
                test_weight = super_weights[idx] * self.increment
                if test_weight < self.super_lower_bound:
                    test_weight = self.super_lower_bound
            else:

                test_weight = super_weights[idx] / self.increment
                if test_weight > self.super_upper_bound:
                    test_weight = self.super_upper_bound

            super_weights[idx] = test_weight

        # print("loss calc supers", super_weights)

        self.attack.sess.run([
            self.weights.assign(new_weights),
            self.super_weights.assign(super_weights)
        ])


class BaseSumOfLogProbsLossOld(GreedyPathTokenWeightingBinarySearch):
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


class BaseSumOfLogProbsLoss(LogProbsTokenWeightingBinarySearch):
    def __init__(self, attack, weight_settings=(None, None), updateable: bool = False):

        ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            attack.placeholders.targets,
            attack.placeholders.target_lengths
        )

        with tf.GradientTape() as tape:
            tape.watch(attack.victim.raw_logits)

            path_gradient_loss = tf.nn.ctc_loss(
                labels=ctc_target,
                inputs=attack.victim.raw_logits,
                sequence_length=attack.placeholders.audio_lengths,
            )

        gradients = tape.gradient(
            path_gradient_loss,
            attack.victim.raw_logits
        )
        gradients = tf.transpose(gradients, [1, 0, 2])
        argmin_grads = tf.argmin(gradients, axis=-1)

        self.gradient_argmin = argmin_grads
        self.gradient_argmax = tf.argmax(gradients, axis=-1)

        super().__init__(
            attack,
            custom_target=argmin_grads,
            use_softmax=True,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        self.log_smax = tf.log(self.target_logit * self.weights)

        self.fwd_target_log_probs = tf.reduce_sum(
            self.log_smax, axis=-1
        ) * self.super_weights
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


class FwdOnlyLogProbsLoss(BaseSumOfLogProbsLossOld):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):
        """
        """

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        self.loss_fn = - self.fwd_target_log_probs
        # self.loss_fn *= -self.weights


class FwdOnlyLogProbsLossWithGradsPath(BaseSumOfLogProbsLoss):
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):
        """
        """

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable
        )

        self.loss_fn = - self.fwd_target_log_probs
        # self.loss_fn *= -self.weights


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
    "cw-toks": CWMaxMinWithPerTokenWeights,
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
    "cw-toks": CWMaxMinWithPerTokenWeights,
    "cw-gradientpath": CWMaxMinWithCTCGrads,
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
    "logprobs-gradientpath": FwdOnlyLogProbsLossWithGradsPath,
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
    "l2-carlini": L2SquaredLoss,
    "l2-sample": L2SquaredLoss,
    "linf": LinfLoss,
    "rms-energy": None,
    "energy": None,
    "peak-to-peak": None,
}


class SoftmaxCTCLoss(SimpleWeightings):
    """
    Tensorflow implementatons of CTC Loss take the activations/logits as input and
    convert them to time major softmax values for us... but this means we can't take a
    partial derivative w.r.t. the softmax outputs (only the logits).
    This class fixes that problem.
    """

    def __init__(self, attack, weight_settings=(1.0, 1.0),
            updateable: bool = False):
        super().__init__(attack, weight_settings=weight_settings,
                         updateable=updateable)

        self.attack = attack

        self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            attack.placeholders.targets,
            attack.placeholders.target_lengths
        )

        self.loss_fn = self.ctc(
            labels=attack.placeholders.targets,
            logits=attack.victim.logits,
            label_length=attack.placeholders.target_lengths,
            logit_length=attack.batch.audios["ds_feats"],
            blank_index=-1
        )

    def ctc(self, labels, logits, label_length, logit_length,
            logits_time_major=True, unique=None, blank_index=None, name=None,
            apply_softmax=True):
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

            result = self.ctc_loss_and_grad(**kwargs)

            def grad(grad_loss):
                grad = [
                    array_ops.reshape(grad_loss, [1, -1, 1]) * result[1]]
                grad += [None] * (len(args) - len(grad))
                return grad

            return result[0], grad

        return compute_ctc_loss(*args)

    def ctc_loss_and_grad(self, logits, labels, label_length, logit_length,
            unique=None):

        """
        Computes the actual loss and gradients.
        """

        num_labels = _get_dim(logits, 2)
        max_label_seq_length = _get_dim(labels, 1)

        ilabel_log_probs = tf.log(logits)
        state_log_probs = _ilabel_to_state(labels, num_labels, ilabel_log_probs)
        state_trans_probs = _ctc_state_trans(labels)
        initial_state_log_probs, final_state_log_probs = ctc_state_log_probs(
            label_length, max_label_seq_length)
        fwd_bwd_log_probs, log_likelihood = _forward_backward_log(
            state_trans_log_probs=math_ops.log(state_trans_probs),
            initial_state_log_probs=initial_state_log_probs,
            final_state_log_probs=final_state_log_probs,
            observed_log_probs=state_log_probs,
            sequence_length=logit_length)

        if unique:
            olabel_log_probs = _state_to_olabel_unique(labels, num_labels,
                                                       fwd_bwd_log_probs,
                                                       unique)
        else:
            # olabel_log_probs = _state_to_olabel(labels, num_labels,
            #                                     fwd_bwd_log_probs)
            olabel_log_probs = self.custom_state_to_olabel_max(labels,
                                                               num_labels,
                                                               fwd_bwd_log_probs)

        grad = math_ops.exp(ilabel_log_probs) - math_ops.exp(olabel_log_probs)
        loss = -log_likelihood
        return loss, grad

    @staticmethod
    def custom_state_to_olabel_max(labels, num_labels, states):
        """Sum state log probs to ilabel log probs."""

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


class CWMaxMinWithCTCGrads(GreedyPathTokenWeightingBinarySearch):
    def __init__(self, attack, k=0.0, weight_settings=(1.0, 1.0),
            updateable: bool = False, use_softmax: bool = False):

        assert k >= 0

        ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            attack.placeholders.targets,
            attack.placeholders.target_lengths
        )

        if use_softmax is True:
            watch_gradient_var = attack.victim.logits
            time_major = False
            ctc_fn = self.ctc_call
            apply_softmax = False
            print(watch_gradient_var.shape)

        else:

            watch_gradient_var = attack.victim.logits
            print(watch_gradient_var.shape)
            time_major = False
            # ctc_fn = tf.nn.ctc_loss_v2
            ctc_fn = self.ctc_call
            apply_softmax = False

        with tf.GradientTape() as tape:

            tape.watch(watch_gradient_var)

            path_gradient_loss = self.ctc_call(
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
        print(gradients)
        # gradients = tf.transpose(gradients, [1, 0, 2])
        print(gradients)
        argmin_grads = tf.argmin(gradients, axis=-1)
        print(argmin_grads)
        self.grad_logits = gradients
        if argmin_grads.shape[-1] == attack.batch.size:
            print("lkdsj\f")
            self.gradient_argmin = tf.transpose(argmin_grads, [1, 0])
        else:
            self.gradient_argmin = argmin_grads

        print(self.gradient_argmin.shape)

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
        self.loss_fn = self.loss_fn
