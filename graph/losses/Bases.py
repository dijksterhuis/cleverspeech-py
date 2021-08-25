import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod
from cleverspeech.utils.Utils import log


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


class BaseAlignmentLoss(BaseLoss):
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


class _BasePathSearch(BaseAlignmentLoss):
    def get_most_likely_path_as_tf(self):
        pass


class _GreedySearchPath(_BasePathSearch):
    def get_most_likely_path_as_tf(self):
        return tf.cast(
            tf.argmax(self.current, axis=-1), dtype=tf.int32
        )


class _BeamSearchPath(_BasePathSearch):
    def get_most_likely_path_as_tf(self):

        smax = self.attack.procedure.tf_run(self.attack.victim.logits)
        lengths = self.attack.batch.audios["ds_feats"]

        _, _, toks, times = self.attack.victim.ds_decode_batch_no_lm(
            smax, lengths, top_five=False, with_metadata=True
        )

        blanks = np.ones(self.target_argmax.shape, dtype=np.int32) * 28

        for idx, (tok, time) in enumerate(zip(toks, times)):
            blanks[idx][time] = tok

        return tf.cast(blanks, dtype=tf.int32)


class _BaseBinarySearches(_BasePathSearch):
    def __init__(self, attack, custom_target=None, use_softmax=False, weight_settings=(None, None), updateable: bool = False):

        assert type(weight_settings) in [list, tuple]
        assert all(type(t) in [float, int] for t in weight_settings)

        self.lower_bound = None
        self.upper_bound = None
        self.weights = None
        self.initial = None
        self.increment = None

        super().__init__(
            attack,
            custom_target=custom_target,
            weight_settings=weight_settings,
            updateable=updateable,
            use_softmax=use_softmax,
        )

        self.init_weights(attack, weight_settings)


class _SimpleTokenWeights(_BaseBinarySearches):

    def init_weights(self, attack, weight_settings):

        weight_settings = [float(s) for s in weight_settings]

        if len(weight_settings) == 2:

            self.initial, self.increment = weight_settings
            self.upper_bound = self.initial
            self.lower_bound = 0.0

        elif len(weight_settings) == 3:
            self.initial, self.increment, n = weight_settings
            # never be more than N steps away from initial value
            self.upper_bound = self.initial
            self.lower_bound = self.initial * (self.increment ** n)

        else:
            raise ValueError

        # handle the case where we might take log probs and want to set initial
        # value as small and double for bad frames (increase their influence)

        if self.use_softmax is True:
            self.upper_bound = 1.0

        shape = [attack.batch.size, attack.batch.audios["ds_feats"][0]]

        self.weights = tf.Variable(
            tf.ones(shape, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        initial_vals = self.initial * np.ones(shape, dtype=np.float32)
        attack.sess.run(self.weights.assign(initial_vals))

    def check_token_weights(self):

        current_most_likely_path = self.get_most_likely_path_as_tf()

        target_argmax = tf.cast(
            self.target_argmax, dtype=tf.int32
        )

        argmax_test = tf.where(
            tf.equal(target_argmax, current_most_likely_path),
            tf.multiply(self.increment, self.weights),
            tf.multiply(1 / self.increment, self.weights),
        )
        upper_bound = self.upper_bound * tf.ones_like(
            argmax_test, dtype=tf.float32
        )
        lower_bound = self.lower_bound * tf.ones_like(
            argmax_test, dtype=tf.float32
        )

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

    def update_one(self, idx: int):
        raise Exception

    def update_many(self, batch_successes: list):
        self.check_token_weights()


class _DoubleTokenWeights(_SimpleTokenWeights):

    def __init__(self, attack, custom_target=None, use_softmax=False, weight_settings=(None, None), updateable: bool = False):

        self.super_upper_bound = None
        self.super_lower_bound = None
        self.super_weights = None

        super().__init__(
            attack,
            custom_target=custom_target,
            weight_settings=weight_settings,
            updateable=updateable,
            use_softmax=use_softmax,
        )

    def init_weights(self, attack, weight_settings):

        super().init_weights(attack, weight_settings)

        n = 1
        self.super_upper_bound = self.initial
        self.super_lower_bound = self.initial * (self.increment ** n)

        self.super_weights = tf.Variable(
            tf.ones(attack.batch.size, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )
        super_vals = self.super_upper_bound * np.ones(attack.batch.size, dtype=np.float32)

        attack.sess.run(self.super_weights.assign(super_vals))

    def check_super_weights(self, batch_successes):

        w = self.attack.procedure.tf_run(self.super_weights)

        incr, upper, lower = self.increment, self.upper_bound, self.lower_bound

        for idx, success_check in enumerate(batch_successes):
            if success_check is True:
                w[idx] = w[idx] * incr if w[idx] * incr > lower else lower
            else:
                w[idx] = w[idx] / incr if w[idx] / incr < upper else upper

        self.attack.sess.run(self.super_weights.assign(w))

    def update_one(self, idx: int):
        raise Exception

    def update_many(self, batch_successes: list):

        super().update_many(batch_successes)
        self.check_super_weights(batch_successes)


class _KappaTokenWeights(_DoubleTokenWeights):
    def __init__(self, attack, custom_target=None, use_softmax=False, weight_settings=(None, None), updateable: bool = False):

        self.kappa = None

        super().__init__(
            attack,
            custom_target=custom_target,
            weight_settings=weight_settings,
            updateable=updateable,
            use_softmax=use_softmax,
        )

    def init_weights(self, attack, weight_settings):

        super().init_weights(attack, weight_settings)

        self.kappa = tf.Variable(
            tf.zeros(attack.batch.size, dtype=tf.float32),
        )
        self.attack.sess.run(
            self.kappa.assign(0.0 * np.zeros(attack.batch.size))
        )

    def check_kappa(self, batch_successes):

        kaps, losses, = self.attack.procedure.tf_run(
            [self.kappa, self.attack.loss[0].loss_fn]
        )
        z = zip(batch_successes, kaps, losses)
        for idx, (suc, k, l) in enumerate(z):
            if suc is False and l <= 0:
                kaps[idx] += 1
            print(idx, suc, l, kaps)

        self.attack.sess.run(self.kappa.assign(kaps))

    def update_one(self, idx: int):
        raise Exception

    def update_many(self, batch_successes: list):
        super().update_many(batch_successes)
        self.check_kappa(batch_successes)


class SimpleGreedySearchTokenWeights(_SimpleTokenWeights, _GreedySearchPath):
    pass


class DoubleGreedySearchTokenWeights(_DoubleTokenWeights, _GreedySearchPath):
    pass


class KappaGreedySearchTokenWeights(_KappaTokenWeights, _GreedySearchPath):
    pass


class SimpleBeamSearchTokenWeights(_SimpleTokenWeights, _BeamSearchPath):
    pass


class DoubleBeamSearchTokenWeights(_DoubleTokenWeights, _BeamSearchPath):
    pass


class KappaBeamSearchTokenWeights(_KappaTokenWeights, _BeamSearchPath):
    pass


class _InvertedSimpleTokenWeights(_BaseBinarySearches):

    def init_weights(self, attack, weight_settings):

        weight_settings = [float(s) for s in weight_settings]

        if len(weight_settings) == 2:

            self.initial, self.increment = weight_settings

        elif len(weight_settings) == 3:
            self.initial, self.increment, n = weight_settings
            # never be more than N steps away from initial value
            self.lower_bound = self.initial * ((1 / self.increment) ** n)

        else:
            raise ValueError

        weight_settings = [float(setting) for setting in weight_settings]
        self.initial, self.increment = weight_settings

        # softmax has a known maximum, minimum is (usually) never zero though
        self.upper_bound = 1.0

        # never be more than N checks away from initial value
        self.lower_bound = self.initial * (self.increment ** n)

        shape = [attack.batch.size, attack.batch.audios["ds_feats"][0]]

        self.weights = tf.Variable(
            tf.ones(shape, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        initial_vals = self.initial * np.ones(shape, dtype=np.float32)
        attack.sess.run(self.weights.assign(initial_vals))

    def check_token_weights(self):
        current_most_likely_path = self.get_most_likely_path_as_tf()

        target_argmax = tf.cast(
            self.target_argmax, dtype=tf.int32
        )

        argmax_test = tf.where(
            tf.equal(target_argmax, current_most_likely_path),
            tf.multiply(self.increment, self.weights),
            tf.multiply(1 / self.increment, self.weights),
        )
        upper_bound = self.upper_bound * tf.ones_like(
            argmax_test, dtype=tf.float32
        )
        lower_bound = self.lower_bound * tf.ones_like(
            argmax_test, dtype=tf.float32
        )

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

    def update_one(self, idx: int):
        raise Exception

    def update_many(self, batch_successes: list):
        self.check_token_weights()

