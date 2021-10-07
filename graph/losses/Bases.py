import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod
from cleverspeech.utils.Utils import log, l_map


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
            self.lower_bound = 1e-10

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

    def __init__(self, attack, *args, **kwargs):

        super().__init__(attack, *args, **kwargs)

        self.requires_update = False
        self.most_likely_path = None
        self.get_most_likely_path_as_tf()

    def get_most_likely_path_as_tf(self):
        pass


class _GreedySearchPath(_BasePathSearch):

    def get_most_likely_path_as_tf(self):
        self.most_likely_path = tf.cast(
            tf.argmax(self.attack.victim.logits, axis=-1), dtype=tf.int32
        )


class _BeamSearchPath(_BasePathSearch):

    def __init__(self, attack, *args, **kwargs):
        super().__init__(attack, *args, **kwargs)

        self.requires_update = True

    def init(self):
        self.most_likely_path = tf.cast(
            tf.argmax(self.attack.victim.logits, axis=-1), dtype=tf.int32
        )

    def get_most_likely_path_as_tf(self):

        # check if we've got a graph with tf variables initialised

        if self.attack.procedure is None:
            # graph not loaded, we can't get values from sess.run so initialise
            # with greedy tokens for now
            self.most_likely_path = tf.cast(
                tf.argmax(self.attack.victim.logits, axis=-1), dtype=tf.int32
            )

        else:
            # we have a graph! now we can query for the logits
            smax = self.attack.sess.run(
                self.attack.victim.logits,
                feed_dict=self.attack.feeds.attack
            )

            lengths = self.attack.batch.audios["ds_feats"]

            _, _, toks, times = self.attack.victim.ds_decode_batch(
                smax, lengths, top_five=False, with_metadata=True
            )
            shape = [
                self.attack.batch.size,
                self.attack.batch.audios["max_feats"]
            ]
            blanks = np.ones(shape, dtype=np.int32) * 28

            for idx, (tok, time) in enumerate(zip(toks, times)):
                blanks[idx, time] = tok

            print(blanks)

            self.most_likely_path = tf.cast(blanks, dtype=tf.int32)


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

    def safe_lower_bound(self, n):
        return self.initial * (self.increment ** n)


class _SimpleTokenWeights(_BaseBinarySearches):

    def __init__(self, attack, custom_target=None, use_softmax=False, weight_settings=(None, None), updateable: bool = False):

        self.tf_new_weights = None
        self.test = None

        super().__init__(
            attack,
            custom_target=custom_target,
            weight_settings=weight_settings,
            updateable=updateable,
            use_softmax=use_softmax,
        )

    def init_weights(self, attack, weight_settings):

        weight_settings = [float(s) for s in weight_settings]

        if len(weight_settings) == 2:

            self.initial, self.increment = weight_settings
            n = 6  # use a small default value!

        elif len(weight_settings) == 3:
            self.initial, self.increment, n = weight_settings

        else:
            raise ValueError

        if self.use_softmax is True:

            # if we're using softmax then override user values for safety

            self.upper_bound = 1.0
            self.lower_bound = 1.0e-6
            self.initial = 1.0e-6

        else:

            # never be more than N steps away from upper bound value

            self.upper_bound = self.initial
            self.lower_bound = self.safe_lower_bound(n)

        shape = [attack.batch.size, attack.batch.audios["ds_feats"][0]]

        self.weights = tf.Variable(
            tf.ones(shape, dtype=tf.float32),
            trainable=False,
            validate_shape=True,
            name="qq_loss_weight"
        )

        initial_vals = self.initial * np.ones(shape, dtype=np.float32)
        attack.sess.run(self.weights.assign(initial_vals))

        self.tf_new_weights = self.create_tf_tokens_test_graph()

    def check_for_resets(self, successes, weights):

        inits = self.initial * np.ones(weights.shape[1:], dtype=np.float32)

        res = np.asarray(
            l_map(
                lambda x: inits if x[0] else x[1],  zip(successes, weights)
            ),
            dtype=np.float32
        )

        return res

    def create_tf_token_test_bool_graph(self):

        target_argmax = tf.cast(
            self.target_argmax, dtype=tf.int32
        )

        self.test = tf.equal(target_argmax, self.most_likely_path)

    def create_tf_tokens_test_graph(self):

        self.create_tf_token_test_bool_graph()

        updated = tf.where(
            self.test,
            tf.multiply(self.increment, self.weights),
            tf.multiply(1 / self.increment, self.weights),
        )

        upper_bound = self.upper_bound * tf.ones_like(
            updated, dtype=tf.float32
        )
        lower_bound = self.lower_bound * tf.ones_like(
            updated, dtype=tf.float32
        )

        max_clamp = tf.where(
            tf.greater(updated, upper_bound),
            upper_bound,
            updated,
        )
        return tf.where(
            tf.less(max_clamp, lower_bound),
            lower_bound,
            max_clamp,
        )

    def check_token_weights(self, batch_successes):

        if self.requires_update is True:
            self.get_most_likely_path_as_tf()

        new_weights = self.attack.procedure.tf_run(self.tf_new_weights)
        new_weights = self.check_for_resets(batch_successes, new_weights)

        self.attack.sess.run(self.weights.assign(new_weights))

    def update_one(self, idx: int):
        raise Exception

    def update_many(self, batch_successes: list):
        self.check_token_weights(batch_successes)


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
        self.super_lower_bound = self.safe_lower_bound(n)

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


class _KappaTokenWeights(_SimpleTokenWeights):
    def __init__(self, attack, custom_target=None, use_softmax=False, weight_settings=(None, None), updateable: bool = False):

        self.kappa = None

        super().__init__(
            attack,
            custom_target=custom_target,
            weight_settings=weight_settings,
            updateable=updateable,
            use_softmax=use_softmax,
        )

        self.tf_tokens_test_check = tf.reduce_all(
            self.test, axis=-1
        )

    def init_weights(self, attack, weight_settings):

        super().init_weights(attack, weight_settings)

        self.kappa = tf.Variable(
            tf.zeros(attack.batch.size, dtype=tf.float32), name="qq_kappa"
        )
        kappa_init = (self.increment ** 5) * np.ones(attack.batch.size, dtype=np.float32)

        self.attack.sess.run(
            self.kappa.assign(kappa_init)
        )

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
                kaps[idx] *= 1/self.increment

        self.attack.sess.run(self.kappa.assign(kaps))

        # self.attack.sess.run(
        #     [self.kappa.assign(kaps), self.weights.assign(weights)]
        # )

    def update_one(self, idx: int):
        raise Exception

    def update_many(self, batch_successes: list):
        self.check_kappa(batch_successes)
        super().update_many(batch_successes)


class SimpleGreedySearchTokenWeights(_SimpleTokenWeights, _GreedySearchPath):
    pass


class DoubleGreedySearchTokenWeights(_DoubleTokenWeights, _GreedySearchPath):
    pass


class KappaGreedySearchTokenWeights(_KappaTokenWeights, _GreedySearchPath):
    pass


# TODO: The most likely path from the DeepSpeech beam search decoder is compared
#       against the argmax in _SimpleTokWeights, which is not ideal!

class SimpleBeamSearchTokenWeights(_SimpleTokenWeights, _BeamSearchPath):
    pass


class DoubleBeamSearchTokenWeights(_DoubleTokenWeights, _BeamSearchPath):
    pass


class KappaBeamSearchTokenWeights(_KappaTokenWeights, _BeamSearchPath):
    pass

