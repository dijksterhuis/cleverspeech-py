"""
Variable graphs determine how to create perturbations.

TODO: Move hard constraint and adversarial example steps out to Constructors

TODO: Rename to PerturbationsSubGraph

TODO: Functional? Only need opt_vars, raw + final deltas (no class state)

TODO: Add rounding() method **here**, not in Procedures!



--------------------------------------------------------------------------------
"""


import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod

from cleverspeech.graph.GraphConstructor import Placeholders
from cleverspeech.utils.Utils import np_arr, lcomp


class AbstractVariableGraph(ABC):
    """
    Abstract Base Class to define how an adversarial examples are created.
    Mostly to get tensorflow to do x_{adv} = x + delta correctly when working
    in batches by applying masks etc.

    TODO: The hard constraint shouldn't be applied in AbstractVariableGraph.

    TODO: Rename to AbstractPerturbationsGraph.

    :param sess: a tensorflow Session object
    :param batch: a cleverspeech.data.ingress.batch_generators.batch object
    :param hard_constraint: a cleverspeech.graph.Constraints object to clip the
           perturbation, according to Dist(delta) < tau
    :param placeholders: optional custom tensorflow placeholders
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

    def __init__(self, sess, batch, hard_constraint, placeholders=None):

        batch_size = batch.size
        max_len = batch.audios["max_samples"]
        act_lengths = batch.audios["n_samples"]

        self.bit_depth = 2**15
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

        lower = -self.bit_depth
        upper = self.bit_depth - 1

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

    def get_valid_perturbations(self, sess):
        return sess.run(self.final_deltas)

    def get_raw_perturbations(self, sess):
        return sess.run(self.raw_deltas)

    def get_optimisable_variables(self):
        return self.opt_vars

    @abstractmethod
    def deltas_apply(self, deltas, func):
        pass

    def apply_perturbation_rounding(self, sess):

        """
        For each perturbation sample (delta_n) find the closest integer
        less than the current float value.

        This helps the attacks work against the deepspeech native client api
        which only accepts tf.int16 type inputs. Although it doesn't work 100%
        of the time so use `bin/classify.py` to get the true success rate.

        We could do this after running optimisation, but doing things during
        attacks seems to help find a 16 bit int solution... At least that's my
        excuse.

        N.B. Reassign perturbations that were bounded by the hard constraint
        => raw_delta samples values can be much larger than the final_delta
        sample values.

        :param sess: a tensorflow session object
        :return: None
        """
        def rounding_func(delta):
            signs = tf.sign(delta)
            abs_floor = tf.floor(tf.abs(delta))
            return signs * abs_floor

        deltas = sess.run(self.final_deltas)
        sess.run(self.deltas_apply(deltas, rounding_func))

    def apply_perturbation_randomisation(self, sess, bit_depth_percent=0.5):

        """
        Randomise each perturbation sample (delta_n) using a random uniform
        distribution, with max and min values based on a percentage of the
        perturbation bit depth.

        :param sess: a tensorflow session object
        :param bit_depth_percent: base the random uniform dist. on our bit depth
        :return: None
        """
        def random_uniform_func(delta):
            rand_uni = tf.random_uniform(
                delta.shape,
                minval=-bit_depth_percent * self.bit_depth,
                maxval=bit_depth_percent * self.bit_depth,
                dtype=tf.float32
            )
            return delta + rand_uni

        deltas = sess.run(self.final_deltas)
        sess.run(self.deltas_apply(deltas, random_uniform_func))


class Independent(AbstractVariableGraph):
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

    def deltas_apply(self, deltas, func):
        assign_ops = []
        for idx, delta in enumerate(deltas):

            new_delta = func(delta)

            assign_ops.append(
                self.raw_deltas[idx].assign(new_delta)
            )

        return assign_ops


class Batch(AbstractVariableGraph):
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

    def deltas_apply(self, deltas, func):
        new_deltas = []
        for idx, delta in enumerate(deltas):

            new_delta = func(delta)
            new_deltas.append(new_delta)

        new_deltas = np.asarray(new_deltas)
        assign_op = self.raw_deltas.assign(new_deltas)

        return assign_op


class Synthesis:
    """
    Attack graph from 2018 Carlini & Wager Targeted Audio Attack modified to
    create perturbations from differentiable synthesis methods.

    TODO: Rename to SynthesisPerturbationsSubGraph

    TODO: Make this inherit from AbstractPerturbationGraph

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


