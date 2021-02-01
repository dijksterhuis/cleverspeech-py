import numpy as np
import tensorflow as tf

from abc import ABC

from cleverspeech.utils.Utils import lcomp, np_one, np_arr, log


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
            lcomp(self._gen_tau(batch.audios.actual_lengths)),
            np.float32
        )

        sess.run(self.bounds.assign(self.initial_taus))

    def _gen_tau(self, act_lengths):
        for l in act_lengths:
            yield [self.analyse([self.maximum for _ in range(l)])]

    def analyse(self, x):
        return np.power(np.sum(np.power(np.abs(x), 2)), 1 / 2)

    def clip(self, x):
        """
        N.B. The `axes` flag for `p=2` must be used as as tensorflow runs
        the over *all* tensor dimensions by default.

        :param x:
        :param bound:
        :return:
        """
        return tf.clip_by_norm(x, self.bounds, axes=[1])

    def get_new_bound(self, bound, distance):

        if self.update_method == "lin":
            return self.get_new_linear(bound)
        elif self.update_method == "log":
            return self.get_new_log(bound)
        elif self.update_method == "geom":
            return self.get_new_geometric(bound, distance)

    def get_new_geometric(self, bound, distance):
        """
        Get a new rescale constant.
        """

        if bound > distance:
            # Sample is way over, rest bound to current distance and rescale
            rc = (distance / bound) * self.r_constant
        else:
            # else reduce bound by constant
            rc = self.r_constant

        new_bound = np.ceil(bound * rc)

        # if we are limiting perturbations to a maximum allowable value we
        # check if the current bound is less than the lower bound then reset
        # rescale so the current bound === lower bound

        # TODO: we need some logic here if we want to set a specific range of
        #       bound values for security evaluation curves.

        if self.lowest_bound:
            if new_bound < self.lowest_bound:
                new_bound = self.lowest_bound

        return new_bound

    def get_new_linear(self, bound):
        """
        Get a new rescale constant linearly -- helpful for security evaluations
        as we don't have to mess around with undoing the geometric progression
        just to work out whether we want to capture the current distance stats.
        """
        new_bound = np.round(bound - self.r_constant, 6)

        # there's some weird rounding things that happen between tf and numpy
        # floats... the rescale can actually be something like 0.1000000002572
        # so we have to perform a check to make sure the new value is sensible
        # and set it to the minimum if not.
        if bound - new_bound <= self.r_constant:
            new_bound = self.r_constant

        # if we are limiting perturbations to a maximum allowable value we
        # check if the current bound is less than the lower bound then reset
        # rescale so the current bound === lower bound

        # TODO: we need some logic here if we want to set a specific range of
        #       bound values for security evaluation curves.

        if self.lowest_bound:
            if new_bound < self.lowest_bound:
                new_bound = self.lowest_bound

        precision = int(np.ceil(np.log10(1 / self.r_constant)))

        return np.round(new_bound, precision)

    def get_new_log(self, bound):
        """
        Get a new rescale constant according to a log scale -- helpful for
        security evaluations as we don't have to mess around with undoing the
        geometric progression just to work out whether we want to capture the
        current distance stats.

        Note -- make sure to set rescale to something like 0.1 so it doesn't
        become a geometric progression.
        """
        new_bound = np.round(bound, 8) * self.r_constant

        # if we are limiting perturbations to a maximum allowable value we
        # check if the current bound is less than the lower bound then reset
        # rescale so the current bound === lower bound

        # TODO: we need some logic here if we want to set a specific range of
        #       bound values for security evaluation curves.

        if self.lowest_bound:
            if new_bound < self.lowest_bound:
                new_bound = self.lowest_bound

        return new_bound

    def update_one(self, delta, index):

        current_bounds = self.tf_run(self.bounds)
        current_bound = current_bounds[index][0]
        current_distance = self.analyse(delta)

        new_bound = self.get_new_bound(current_bound, current_distance)

        current_bounds[index][0] = new_bound
        self.tf_run(self.bounds.assign(current_bounds))

    def update(self, new_bounds):
        self.tf_run(self.bounds.assign(new_bounds))

    # def log_scale_init(self):
    #     self.lengths = np_arr(
    #         [[l] for l in g.batch.audios.actual_lengths],
    #         np.int32
    #     )
    #
    #     self.evaluation_current = np.ceil(
    #         np.log(g.initial_taus) - 1
    #     )
    #
    # def log_scale_check(self, bound, distance, rescale, eval_r, base_tau):
    #
    #     """
    #     TODO.
    #     Get a new rescale constant, but also decrease it at a controlled
    #     exponential rate based on the maximum possible distance for the example
    #     so we can evaluate based on allowed attack strength (i.e. size of
    #     perturbation).
    #     """
    #
    #     if int(bound - distance) > 0:
    #         # Bound is > than current distance, so reset rescale to a value that
    #         # moves us to the current distance -- we shouldn't get bigger.
    #         r = distance / float(base_tau)
    #
    #     else:
    #         # else reduce bound by r constant
    #         r = rescale * self.r_constant
    #
    #     if np.ceil(np.log(r * base_tau)) < eval_r:
    #         final_r = np.exp(eval_r).astype(np.float32)
    #         final_r = final_r / base_tau
    #
    #         # handle the case where the current example has reached the minimum
    #         # available evaluation value
    #         if eval_r - 1.0 == 0:
    #             eval_r = 1.0
    #         else:
    #             eval_r = eval_r - 1
    #     else:
    #         final_r = r
    #
    #     return final_r, eval_r


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
        return np.power(np.sum(np.power(np.abs(x), 2)), 1 / 2)

    def clip(self, x):
        """
        N.B. The `axes` flag for `p=2` must be used as as tensorflow runs
        the over *all* tensor dimensions by default.

        :param x:
        :param bound:
        :return:
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
        return np.max(np.abs(x))

    def clip(self, x):
        # N.B. There is no `axes` flag for `p=inf` as tensorflow runs the
        # operation on the last tensor dimension by default.
        return tf.clip_by_value(x, -self.bounds, self.bounds)

