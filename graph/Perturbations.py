"""
Variable graphs determine how to create perturbations.

--------------------------------------------------------------------------------
"""


import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod
from cleverspeech.utils.Utils import np_arr, lcomp


class AbstractPerturbationsGraph(ABC):

    @staticmethod
    def _gen_mask(lengths, max_len):
        """
        Generate the zero / one valued mask for the perturbations in a batch
        Mask value == 1 when actual < max, == 0 when actual > max

        :param lengths: number of samples per audio example in a batch
        :param max_len: maximum numb. of samples of audio examples in a batch

        :yield: 0/1 valued mask vector of length max_len
        """
        for l in lengths:
            m = list()
            for i in range(max_len):
                if i < l:
                    m.append(1)
                else:
                    m.append(0)
            yield m

    def __init__(self, sess, batch, bit_depth=1.0, random_scale=0.002):

        self.batch = batch
        self.sess = sess
        self.bit_depth = bit_depth
        self.random_scale = random_scale

        self.raw_deltas = None
        self.opt_vars = None
        self.deltas = None

    @abstractmethod
    def create_perturbations(self, batch_size, max_len):
        pass

    @abstractmethod
    def create_graph(self, sess, batch):
        pass

    def initial_delta_values(self, shape):

        upper = self.bit_depth * self.random_scale
        lower = self.bit_depth * self.random_scale

        return tf.random.uniform(
            shape,
            minval=lower,
            maxval=upper,
            dtype=tf.float32
        )

    def get_masked_perturbations(self, sess):
        return sess.run(self.deltas)

    def get_raw_deltas(self, sess):
        return sess.run(self.raw_deltas)

    def get_optimisable_variables(self):
        return self.opt_vars


class IndependentVariables(AbstractPerturbationsGraph):

    def __init__(self, sess, batch, bit_depth=1.0, random_scale=0.002):

        super().__init__(
            sess, batch, bit_depth=bit_depth, random_scale=random_scale
        )

        self.create_graph(sess, batch)

    def create_perturbations(self, batch_size, max_len):

        self.raw_deltas = []

        for _ in range(batch_size):
            d = tf.Variable(
                self.initial_delta_values([max_len]),
                trainable=True,
                validate_shape=True,
                dtype=tf.float32,
                name='qq_delta'
            )
            self.raw_deltas.append(d)

        self.opt_vars = self.raw_deltas

        return tf.stack(self.raw_deltas, axis=0)

    def create_graph(self, sess, batch):

        batch_size = batch.size
        max_len = batch.audios["max_samples"]
        act_lengths = batch.audios["n_samples"]

        masks = tf.Variable(
            tf.zeros([batch_size, max_len]),
            trainable=False,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_masks'
        )

        # Generate a batch of delta variables which will be optimised as a batch

        self.deltas = self.create_perturbations(batch_size, max_len)

        # Mask deltas first so we zero value *any part of the signal* that is
        # zero value padded in the original audio
        self.deltas *= masks

        # initialise static variables
        initial_masks = np_arr(
            lcomp(self._gen_mask(act_lengths, max_len)),
            np.float32
        )

        sess.run(masks.assign(initial_masks))

    def projected_rounding(self, successes: list):

        def rounding_func(delta):
            signs = np.sign(delta)
            abs_floor = np.floor(np.abs(delta)).astype(np.int32)
            new_delta = np.round((signs * abs_floor).astype(np.float32))
            return new_delta

        deltas = self.sess.run(self.deltas)

        assign_ops = [
            self.raw_deltas[i].assign(rounding_func(d)) for i, d in enumerate(deltas)
        ]

        self.sess.run(assign_ops)

