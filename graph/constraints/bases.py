import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod
from cleverspeech.utils.Utils import np_arr, lcomp


class AbstractBoxConstraint(ABC):
    """
    Abstract class for constraints, funnily enough.
    """
    def __init__(self, bit_depth=1.0):

        self.bit_depth = bit_depth

    @abstractmethod
    def clip(self, x):
        """
        Only implemented by child classes.
        """
        pass


class AbstractSizeConstraint(AbstractBoxConstraint):
    """
    Abstract class for constraints, funnily enough.
    """
    def __init__(self, sess, batch, bit_depth=1.0, r_constant=0.95, update_method="geom"):

        super().__init__(bit_depth=bit_depth)

        assert type(r_constant) == float or type(r_constant) == np.float32
        assert 0 < r_constant < 1.0

        assert update_method in ["lin", "geom", "log", "geom_binary"]

        self.bit_depth = bit_depth
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

        self.previous = tf.Variable(
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

        sess.run(
            [
                self.bounds.assign(self.initial_taus),
                self.previous.assign(self.initial_taus)
            ]
        )

    def _gen_tau(self, act_lengths):
        """
        Generate the initial bounds based on the the maximum possible value for
        a perturbation and it's actual un-padded length (i.e. number of audio
        samples).
        """
        for l in act_lengths:
            yield self.analyse(np.asarray([self.bit_depth for _ in range(l)]))

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

    def _tf_clipper(self, current):
        denom = tf.maximum(self.bounds, tf.expand_dims(current, axis=-1))
        return self.bounds / denom

    def get_new_bound(self, dist):
        """
        Get a new bound with geometric progression.
        """
        return dist * self.r_constant

    def revert_bound(self, distance, previous):
        """
        Get a new bound with geometric progression.
        """
        diff = max(previous, distance) - min(previous, distance)
        reverted_bound = distance + diff * 0.5 if diff != 0.0 else previous
        return reverted_bound

    def update(self, deltas, successes):
        """
        Update bounds for all perturbations in a batch, if they've found success
        """
        tf_bounds = [self.bounds, self.previous]
        current_bounds, previous_bounds = self.tf_run(tf_bounds)
        current_distances = self.analyse(deltas)[0]

        z = zip(current_bounds, previous_bounds, current_distances, successes)
        for idx, (b, p, d, s) in enumerate(z):

            if s is True:
                p[0] = b[0]
                b[0] = self.get_new_bound(d)

            # if s is False and "_binary" in self.update_method:
            #     b[0] = self.revert_bound(d, p[0])

        bound_assigns = [
            self.bounds.assign(current_bounds),
            self.previous.assign(previous_bounds)
        ]

        self.tf_run(bound_assigns)


