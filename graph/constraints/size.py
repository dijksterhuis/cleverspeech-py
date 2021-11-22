import tensorflow as tf
import numpy as np
import python_speech_features as psf
from cleverspeech.graph.constraints.bases import (
    AbstractSizeConstraint, AbstractDecibelSizeConstraint
)


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


class Peak2Peak(AbstractSizeConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
            update_method=update_method,
        )
        print(self.bounds)

    def analyse(self, x):
        res = np.max(x, axis=-1) - np.min(x, axis=-1)
        if type(res) != list:
            res = [res]
        return res

    def _gen_tau(self, act_lengths):
        """
        Generate the initial bounds based on the the maximum possible value for
        a perturbation and it's actual un-padded length (i.e. number of audio
        samples).
        """
        for l in act_lengths:
            positives = [self.bit_depth for _ in range(l//2)]
            negatives = [-self.bit_depth for _ in range(l//2)]
            yield self.analyse(positives + negatives)

    def clip(self, x):
        peak_to_peak = tf.reduce_max(x, axis=-1) - tf.reduce_min(x, axis=-1)
        return x * self._tf_clipper(peak_to_peak)


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
        return x * self._tf_clipper(rms)


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
        return x * self._tf_clipper(energy)


class SegmentedMeanPeak(AbstractSizeConstraint):

    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
            update_method=update_method,
        )

    def analyse(self, x):

        assert type(x) is np.ndarray

        if len(x.shape) > 1:
            seg_sums = []
            for each_x in x:
                framed = psf.sigproc.framesig(each_x, 512, 512)
                seg_sum = np.mean(np.max(np.abs(framed) ** 2, axis=-1) / 512)
                seg_sums.append([seg_sum])

        else:
            framed = psf.sigproc.framesig(x, 512, 512)
            seg_sums = np.mean(np.max(np.abs(framed) ** 2, axis=-1) / 512)

        res = seg_sums
        if type(res) != list:
            res = [res]
        return res

    def clip(self, x):
        framed = tf.signal.frame(x, 512, 512)
        seg_sum = tf.reduce_mean(tf.reduce_max(tf.abs(framed) ** 2, axis=-1) / 512)
        return x * self._tf_clipper(seg_sum)


class SegmentedEnergy(AbstractSizeConstraint):

    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
            update_method=update_method,
        )

    def analyse(self, x):

        assert type(x) is np.ndarray

        if len(x.shape) > 1:
            seg_sums = []
            for each_x in x:
                framed = psf.sigproc.framesig(each_x, 512, 512)
                seg_sum = np.sum(np.sum(np.abs(framed) ** 2, axis=-1) / 512)
                seg_sums.append([seg_sum])

        else:
            framed = psf.sigproc.framesig(x, 512, 512)
            seg_sums = np.sum(np.sum(np.abs(framed) ** 2, axis=-1) / 512)

        res = seg_sums
        if type(res) != list:
            res = [res]
        return res

    def clip(self, x):
        framed = tf.signal.frame(x, 512, 512)
        seg_sum = tf.reduce_sum(tf.reduce_sum(tf.abs(framed) ** 2, axis=-1) / 512)
        return x * self._tf_clipper(seg_sum)


class PeakDB(AbstractDecibelSizeConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
            update_method=update_method,
        )

    def analyse(self, x):
        res = 20.0 * np.log10(np.max(np.abs(x), axis=-1) + 1e-10)
        if type(res) != list:
            res = [res]
        return res

    def clip(self, x):
        peak = 20.0 * self._log10(tf.reduce_max(tf.abs(x), axis=-1) + 1e-10)
        # NOTE: This doesn't **quite** work correctly
        return x * (tf.maximum(self.bounds, tf.expand_dims(peak, axis=-1)) / self.bounds)


class EnergyDB(AbstractDecibelSizeConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
            update_method=update_method,
        )

    def analyse(self, x):
        safe_energy = np.sum(np.square(np.abs(x)), axis=-1) + 1e-10
        res = np.log10(safe_energy)
        if type(res) != list:
            res = [res]
        return res

    def clip(self, x):
        safe_energy = tf.reduce_sum(tf.square(tf.abs(x)), axis=-1) + 1e-10
        energy = self._log10(safe_energy)
        # NOTE: This doesn't **quite** work correctly
        return x * (tf.maximum(self.bounds, tf.expand_dims(energy, axis=-1)) / self.bounds)


class RMSDB(AbstractDecibelSizeConstraint):
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method=None):
        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            r_constant=r_constant,
            update_method=update_method,
        )

    def analyse(self, x):
        res = 20.0 * np.log10(np.sqrt(np.mean(np.square(np.abs(x)), axis=-1)))
        if type(res) != list:
            res = [res]
        return res

    def clip(self, x):
        rms = 20.0 * self._log10(tf.sqrt(tf.reduce_mean(tf.square(tf.abs(x)), axis=-1)))
        # NOTE: This doesn't **quite** work correctly
        return x * (tf.maximum(self.bounds, tf.expand_dims(rms, axis=-1)) / self.bounds)

