"""
Hard constraints to use with Clipped Gradient Descent Perturbation Graphs.
--------------------------------------------------------------------------------
"""


import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod

from cleverspeech.utils.Utils import lcomp, np_arr


class AbstractConstraint(ABC):
    """
    Abstract class for constraints, funnily enough.
    """
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method="geom", lowest_bound=None):

        assert type(r_constant) == float or type(r_constant) == np.float32
        assert 0 < r_constant < 1.0

        if lowest_bound is not None:
            assert lowest_bound > 0
            assert type(lowest_bound) in [float, int, np.int16, np.int32, np.float32]
            self.lowest_bound = float(lowest_bound)
        else:
            self.lowest_bound = None

        assert update_method in ["lin", "geom", "log"]

        self.__bit_depth = bit_depth
        self.r_constant = r_constant

        self.tf_run = sess.run
        self.update_method = update_method

        self.bounds = tf.Variable(
            tf.zeros([batch.size, 1]),
            trainable=False,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_masks'
        )

        self.initial_taus = np_arr(
            lcomp(self._gen_tau(batch.audios["n_samples"])),
            np.float32
        )

        self.tf_run(self.bounds.assign(self.initial_taus))

    def _gen_tau(self, act_lengths):
        """
        Generate the initial bounds based on the the maximum possible value for
        a perturbation and it's actual un-padded length (i.e. number of audio
        samples).
        """
        for lengths in act_lengths:
            yield self.analyse([self.__bit_depth for _ in range(lengths)])

    @abstractmethod
    def analyse(self, x):
        """
        Only implemented by child classes.
        """
        pass

    @abstractmethod
    def clip(self, x):
        """
        Only implemented by child classes.
        """
        pass

    def get_new_bound(self, distance):
        """
        Get a new bound with geometric progression.
        """
        return distance * self.r_constant

    def update(self, deltas, successes):
        """
        Update bounds for all perturbations in a batch, if they've found success
        """

        current_bounds = self.tf_run(self.bounds)
        current_distances = self.analyse(deltas)[0]

        for b, d, s in zip(current_bounds, current_distances, successes):
            if s is True:
                b[0] = self.get_new_bound(d)

        self.tf_run(self.bounds.assign(current_bounds))


class L2(AbstractConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, lowest_bound=None, update_method=None):

        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
            lowest_bound=lowest_bound,
            update_method=update_method
        )

    def analyse(self, x):
        res = np.power(np.sum(np.power(np.abs(x), 2), axis=-1), 1 / 2)
        if type(res) != list:
            res = [res]
        return res

    def clip(self, x):
        # N.B. The `axes` flag for `p=2` must be used as as tensorflow runs
        # the over *all* tensor dimensions by default.
        return tf.clip_by_norm(x, self.bounds, axes=[1])


class Linf(AbstractConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, lowest_bound=None, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
            lowest_bound=lowest_bound,
            update_method=update_method,
        )

    def analyse(self, x):
        res = np.max(np.abs(x), axis=-1)
        if type(res) != list:
            res = [res]
        return res

    def clip(self, x):
        # N.B. There is no `axes` flag for `p=inf` as tensorflow runs the
        # operation on the last tensor dimension by default.
        return tf.clip_by_value(x, -self.bounds, self.bounds)


class Energy(AbstractConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, lowest_bound=None, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
            lowest_bound=lowest_bound,
            update_method=update_method,
        )

    def analyse(self, x):
        res = np.sum(np.abs(x) ** 2, axis=-1)
        if type(res) != list:
            res = [res]
        return res

    def clip(self, x):
        energy = tf.reduce_sum(tf.abs(x ** 2), axis=-1)
        return x * (self.bounds / tf.maximum(self.bounds, energy))


class RMS(AbstractConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, lowest_bound=None, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
            lowest_bound=lowest_bound,
            update_method=update_method,
        )

    def analyse(self, x):
        res = np.sqrt(np.mean(np.abs(x) ** 2, axis=-1))
        if type(res) != list:
            res = [res]
        return res

    def clip(self, x):

        rms = tf.sqrt(tf.reduce_mean(tf.abs(x ** 2), axis=-1))
        return x * (self.bounds / tf.maximum(self.bounds, rms))

