"""
Variable graphs determine how to create perturbations.

--------------------------------------------------------------------------------
"""


import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod

from cleverspeech.graph.Placeholders import Placeholders
from cleverspeech.utils.Utils import np_arr, lcomp


class AbstractPerturbationSubGraph(ABC):
    """
    Abstract Base Class to define how an adversarial examples are created.
    Mostly to get tensorflow to do x_{adv} = x + delta correctly when working
    in batches by applying masks etc.

    :param sess: a tensorflow Session object
    :param batch: a cleverspeech.data.ingress.batch_generators.batch object
    """

    @abstractmethod
    def create_perturbations(self, batch_size, max_len):
        pass

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

    def __init__(self, sess, batch, bit_depth=2**15):

        batch_size = batch.size
        max_len = batch.audios["max_samples"]
        act_lengths = batch.audios["n_samples"]

        self.__bit_depth = bit_depth
        self.raw_deltas = None
        self.opt_vars = None

        masks = tf.Variable(
            tf.zeros([batch_size, max_len]),
            trainable=False,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_masks'
        )

        # Generate a batch of delta variables which will be optimised as a batch

        deltas = self.create_perturbations(batch_size, max_len)

        # Mask deltas first so we zero value *any part of the signal* that is
        # zero value padded in the original audio
        deltas *= masks

        # Restrict delta to valid space before applying constraints
        self.final_deltas = self.apply_box_constraint(deltas)

        # initialise static variables
        initial_masks = np_arr(
            lcomp(self._gen_mask(act_lengths, max_len)),
            np.float32
        )

        sess.run(masks.assign(initial_masks))

    def apply_box_constraint(self, deltas):
        lower = -self.__bit_depth
        upper = self.__bit_depth - 1

        return tf.clip_by_value(
            deltas,
            clip_value_min=lower,
            clip_value_max=upper
        )

    def get_valid_perturbations(self, sess):
        return sess.run(self.final_deltas)

    def get_raw_perturbations(self, sess):
        return sess.run(self.raw_deltas)

    def get_optimisable_variables(self):
        return self.opt_vars

    @abstractmethod
    def deltas_apply(self, sess, func):
        pass


class Independent(AbstractPerturbationSubGraph):
    """
    This graph creates a batch of B perturbations which are combined with B
    optimisers to directly optimise each delta_b with optimiser_b so that each
    example is optimised independently of other items in a batch

    To be used with classes that inherit the AbstractIndependentOptimiser ABC
    class.

    Uses more GPU memory than a Batch graph.

    TODO: Rename to IndependentPerturbationsSubGraph

    """

    def create_perturbations(self, batch_size, max_len):
        """
        Method to generate the perturbations as a B x [N] vectors for a
        independent variable graph.

        :param batch_size: size of the current batch
        :param max_len: maximum number of audio samples (don't forget padding!)
        :return: stacked raw_deltas vectors, a tf.Variable of size [B x N] with
            type float32
        """

        self.raw_deltas = []

        for _ in range(batch_size):
            d = tf.Variable(
                tf.zeros([max_len], dtype=tf.float32),
                trainable=True,
                validate_shape=True,
                dtype=tf.float32,
                name='qq_delta'
            )
            self.raw_deltas.append(d)

        self.opt_vars = self.raw_deltas

        return tf.stack(self.raw_deltas, axis=0)

    def deltas_apply(self, sess, func):

        deltas = sess.run(self.final_deltas)

        assign_ops = [
            self.raw_deltas[i].assign(func(d)) for i, d in enumerate(deltas)
        ]

        sess.run(assign_ops)


class Batch(AbstractPerturbationSubGraph):
    """
    This graph creates a batch of B perturbations to be optimised by a single
    optimiser. This seems to have side effects such as learning rates being
    affected by constraint updates on other examples in a batch

    To be used with optimisers that inherit from the AbstractBatchOptimiser
    class.

    Uses less GPU memory than the Independent graph.

    TODO: Rename to BatchPerturbationsSubGraph

    """
    def create_perturbations(self, batch_size, max_len):
        """
        Method to generate the perturbations as a [B x N] matrix for a batch
        variable graph.

        :param batch_size: size of the current batch
        :param max_len: maximum number of audio samples (don't forget padding!)
        :return: raw_deltas, a tf.Variable of size [B x N] with type float32
        """
        self.raw_deltas = tf.Variable(
            tf.zeros([batch_size, max_len], dtype=tf.float32),
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_delta'
        )

        self.opt_vars = [self.raw_deltas]
        return self.raw_deltas

    def deltas_apply(self, sess, func):

        new_deltas = []
        deltas = sess.run(self.final_deltas)

        for idx, delta in enumerate(deltas):

            new_delta = func(delta)
            new_deltas.append(new_delta)

        new_deltas = np.asarray(new_deltas)
        assign_op = [self.raw_deltas.assign(new_deltas)]

        sess.run(assign_op)


