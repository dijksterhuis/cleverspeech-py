import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod

from cleverspeech.graph.GraphConstructor import Placeholders
from cleverspeech.utils.Utils import np_arr, lcomp


class AbstractVariableGraph(ABC):

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

    def __init__(self, sess, batch, hard_constraint, placeholders=None):

        """
        :param sess: tf.Session which will be used in the attack.
        :param batch: A batch of data.
        :param hard_constraint: the constraint (Lnorm) used in the attack.

        """
        batch_size = batch.size
        max_len = batch.audios["max_samples"]
        act_lengths = batch.audios["n_samples"]

        self.raw_deltas = None
        self.opt_vars = None

        if placeholders is not None:
            self.placeholders = placeholders
        else:
            self.placeholders = Placeholders(batch_size, max_len)

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

        lower = -2.0 ** 15
        upper = 2.0 ** 15 - 1

        valid_deltas = tf.clip_by_value(
            deltas,
            clip_value_min=lower,
            clip_value_max=upper
        )

        self.final_deltas = hard_constraint.clip(valid_deltas)

        # clip example to valid range
        self.adversarial_examples = tf.clip_by_value(
            self.final_deltas + self.placeholders.audios,
            clip_value_min=lower,
            clip_value_max=upper
        )

        # initialise static variables
        initial_masks = np_arr(
            lcomp(self._gen_mask(act_lengths, max_len)),
            np.float32
        )

        sess.run(masks.assign(initial_masks))


class Independent(AbstractVariableGraph):
    """
    This graph creates a batch of B perturbations which are combined with B
    optimisers to directly optimise each delta_b with optimiser_b so that each
    example is optimised independently of other items in a batch

    To be used with *IndependentOptimiser classes. Each delta_b is optimised by
    the same optimiser. Uses more GPU memory than BatchVariableGraph.
    """

    def create_perturbations(self, batch_size, max_len):

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


class Batch(AbstractVariableGraph):
    """
    This graph creates a batch of B perturbations.

    To be used with *BatchOptimiser classes. Each delta_b is optimised by
    the same optimiser. This can have side effects such as learning rates
    being affected by constraint updates on other examples in a batch, but
    it also uses less GPU memory!
    """
    def create_perturbations(self, batch_size, max_len):
        self.raw_deltas = tf.Variable(
            tf.zeros([batch_size, max_len], dtype=tf.float32),
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_delta'
        )

        self.opt_vars = [self.raw_deltas]
        return self.raw_deltas


class Synthesis:
    """
    Attack graph from 2018 Carlini & Wager Targeted Audio Attack modified to
    create perturbations from differentiable synthesis methods.
    """

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

    def __init__(self, sess, batch, hard_constraint, synthesiser, placeholders=None):

        batch_size = batch.size
        max_len = batch.audios["max_samples"]
        act_lengths = batch.audios["n_samples"]

        if placeholders is not None:
            self.placeholders = placeholders
        else:
            self.placeholders = Placeholders(batch_size, max_len)

        self.masks = tf.Variable(
            tf.zeros([batch_size, max_len]),
            trainable=False,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_masks'
        )

        self.synthesiser = synthesiser
        self.opt_vars = synthesiser.opt_vars

        # Generate the delta synth parameter objects which we will optimise
        deltas = synthesiser.synthesise()

        # Mask deltas first so we zero value *any part of the signal* that is
        # zero value padded in the original audio
        deltas *= self.masks

        # Restrict delta to valid space before applying constraints

        lower = -2.0 ** 15
        upper = 2.0 ** 15 - 1

        valid_deltas = tf.clip_by_value(
            deltas,
            clip_value_min=lower,
            clip_value_max=upper
        )

        self.final_deltas = hard_constraint.clip(valid_deltas)

        # clip example to valid range
        self.adversarial_examples = tf.clip_by_value(
            self.final_deltas + self.placeholders.audios,
            clip_value_min=lower,
            clip_value_max=upper
        )

        # initialise static variables
        initial_masks = np_arr(
            lcomp(self._gen_mask(act_lengths, max_len)),
            np.float32
        )

        sess.run(self.masks.assign(initial_masks))


