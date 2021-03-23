import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod

from cleverspeech.utils.Utils import lcomp, np_arr


class LNorm(ABC):
    def __init__(self, sess, batch, maxval=2**15, r_constant=0.95, update_method="lin", lowest_bound=None):

        assert type(r_constant) == float or type(r_constant) == np.float32
        assert 0 < r_constant < 1.0

        if lowest_bound is not None:
            assert lowest_bound > 0
            assert type(lowest_bound) in [float, int, np.int16, np.int32, np.float32]
            self.lowest_bound = float(lowest_bound)
        else:
            self.lowest_bound = None

        assert update_method in ["lin", "geom", "log"]

        self.maximum = maxval
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
        for l in act_lengths:
            yield self.analyse([self.maximum for _ in range(l)])

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

        if bound > distance:
            # Sample is way over, rest bound to current distance and rescale
            rc = (distance / bound) * self.r_constant
        else:
            # else reduce bound by constant
            rc = self.r_constant

        return np.ceil(bound * rc)

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

        current_bounds = self.tf_run(self.bounds)
        current_bound = current_bounds[index][0]
        current_distance = self.analyse(delta)

        new_bound = self.get_new_bound(current_bound, current_distance)

        current_bounds[index][0] = new_bound
        self.tf_run(self.bounds.assign(current_bounds))

    def update_many(self, deltas, successes):

        current_bounds = self.tf_run(self.bounds)

        for index, (delta, success) in enumerate(zip(deltas, successes)):

            current_bound = current_bounds[index][0]

            if success is True:
                current_distance = self.analyse(delta)
                new_bound = self.get_new_bound(current_bound, current_distance)
                current_bounds[index][0] = new_bound

        self.tf_run(self.bounds.assign(current_bounds))

    def update(self, new_bounds):
        self.tf_run(self.bounds.assign(new_bounds))


class L2(LNorm):
    def __init__(self, sess, batch, maxval=2**15, r_constant=0.95, lowest_bound=None, update_method=None):

        super().__init__(
            sess,
            batch,
            maxval=maxval,
            r_constant=r_constant,
            lowest_bound=lowest_bound,
            update_method=update_method
        )

    def analyse(self, x):
        res = np.power(np.sum(np.power(np.abs(x), 2)), 1 / 2)
        if type(res) != list:
            res = [res]
        return res

    def clip(self, x):
        """
        N.B. The `axes` flag for `p=2` must be used as as tensorflow runs
        the over *all* tensor dimensions by default.
        """
        return tf.clip_by_norm(x, self.bounds, axes=[1])


class Linf(LNorm):
    def __init__(self, sess, batch, maxval=2**15, r_constant=0.95, lowest_bound=None, update_method=None):
        super().__init__(
            sess,
            batch,
            maxval=maxval,
            r_constant=r_constant,
            lowest_bound=lowest_bound,
            update_method=update_method,
        )

    def analyse(self, x):
        res = np.max(np.abs(x))
        if type(res) != list:
            res = [res]
        return res

    def clip(self, x):
        # N.B. There is no `axes` flag for `p=inf` as tensorflow runs the
        # operation on the last tensor dimension by default.
        return tf.clip_by_value(x, -self.bounds, self.bounds)

