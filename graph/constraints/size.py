import tensorflow as tf
import numpy as np

from cleverspeech.graph.constraints.bases import AbstractSizeConstraint


class L2(AbstractSizeConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method=None):

        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
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


class Linf(AbstractSizeConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
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


class Energy(AbstractSizeConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
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


class RMS(AbstractSizeConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
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

