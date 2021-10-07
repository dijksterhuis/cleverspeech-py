import tensorflow as tf

from cleverspeech.graph.constraints.bases import AbstractBoxConstraint


class ClippedBoxConstraint(AbstractBoxConstraint):
    def __init__(self, bit_depth=1.0):
        super().__init__(bit_depth=bit_depth,)

    def clip(self, x):
        # N.B. There is no `axes` flag for `p=inf` as tensorflow runs the
        # operation on the last tensor dimension by default.
        return tf.clip_by_value(x, -self.bit_depth, self.bit_depth)


class TanhBoxConstraint(AbstractBoxConstraint):
    def __init__(self, bit_depth=1.0):
        super().__init__(bit_depth=bit_depth,)

    def clip(self, x):
        return tf.tanh(x / self.bit_depth) * self.bit_depth
