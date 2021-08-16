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
    def __init__(self, sess, batch, bit_depth=2 ** 15, r_constant=0.95, update_method="geom", lowest_bound=None):

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
        for l in act_lengths:
            yield self.analyse([self.__bit_depth for _ in range(l)])

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

    def get_new_bound(self, bound, distance):
        """
        Get a new bound using the method defined by the `update_method`
        attribute.
        """

        if self.update_method == "lin":
            return self.get_new_linear(bound)
        elif self.update_method == "log":
            return self.get_new_log(bound)
        elif self.update_method == "geom":
            return self.get_new_geometric(bound, distance)

    def get_new_geometric(self, bound, distance):
        """
        Get a new rescale constant with geometric progression.
        """
        rc = self.r_constant
        new = distance * rc if bound > distance else bound * rc

        return np.ceil(new)

    def get_new_linear(self, bound):
        """
        Get a new rescale constant linearly.
        """
        new_bound = np.round(bound - bound * self.r_constant, 6)

        # there's some weird rounding things that happen between tf and numpy
        # floats... the rescale can actually be something like 0.1000000002572
        # so we have to perform a check to make sure the new value is sensible
        # and set it to the minimum if not.
        if bound - new_bound <= self.r_constant:
            new_bound = self.r_constant

        precision = int(np.ceil(np.log10(1 / self.r_constant)))
        new_bound = np.round(new_bound, precision)

        return np.ceil(new_bound)

    def get_new_log(self, bound):
        """
        Get a new rescale constant according to a log scale.

        Note -- make sure to set rescale to something like 0.1 so it doesn't
        become a geometric progression.
        """
        new_bound = np.round(bound, 8) * self.r_constant

        return np.ceil(new_bound)

    def update_one(self, delta, index):

        """
        Only update the bound (tau) for one perturbation at a time.

        :param delta: the perturbation to update
        :param index: the index of the perturbation within the batch
        :return: None
        """

        current_bounds = self.tf_run(self.bounds)
        current_bound = current_bounds[index][0]
        current_distance = self.analyse(delta)

        new_bound = self.get_new_bound(current_bound, current_distance)

        current_bounds[index][0] = new_bound
        self.tf_run(self.bounds.assign(current_bounds))

    def update_many(self, deltas, successes):

        """
        Update bounds for all perturbations in a batch, if they've found success
        """

        current_bounds = self.tf_run(self.bounds)
        current_distances = self.analyse(deltas)

        for b, dist, success in zip(current_bounds, current_distances, successes):
            if success is True:
                b[0] = self.get_new_bound(b[0], dist[0])

        self.tf_run(self.bounds.assign(current_bounds))


class L2(AbstractConstraint):
    def __init__(self, sess, batch, bit_depth=2 ** 15, r_constant=0.95, lowest_bound=None, update_method=None):

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
    def __init__(self, sess, batch, bit_depth=2 ** 15, r_constant=0.95, lowest_bound=None, update_method=None):
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
    def __init__(self, sess, batch, bit_depth=2 ** 15, r_constant=0.95, lowest_bound=None, update_method=None):
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
    def __init__(self, sess, batch, bit_depth=2 ** 15, r_constant=0.95, lowest_bound=None, update_method=None):
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

