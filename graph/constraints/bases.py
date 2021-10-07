import tensorflow as tf
import numpy as np
from copy import deepcopy

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

        self.bounds = np_arr(
            lcomp(self._gen_tau(batch.audios["n_samples"])),
            np.float32
        )
        self.initial_taus = deepcopy(self.bounds)

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

    def get_new_bound(self, distance):
        """
        Get a new bound with geometric progression.
        """
        return distance * self.r_constant

    def update(self, deltas, successes):
        """
        Update bounds for all perturbations in a batch, if they've found success
        """

        current_bounds = self.bounds
        current_distances = self.analyse(deltas)[0]

        for b, d, s in zip(self.bounds, current_distances, successes):
            if s is True:
                b[0] = self.get_new_bound(d)

        self.bounds = current_bounds
        # self.tf_run(self.bounds.assign(current_bounds))
