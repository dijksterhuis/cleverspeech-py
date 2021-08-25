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

    def __init__(
            self,
            sess,
            batch,
            bit_depth=2**15,
            random_scale=0.002,
            updateable=False
    ):

        self.batch = batch
        self.sess = sess
        self.bit_depth = bit_depth
        self.random_scale = random_scale
        self.updateable = updateable

        self.raw_deltas = None
        self.opt_vars = None

        self.final_deltas = None
        self.hard_constraint = None
        self.perturbations = None
        self.adversarial_examples = None

    @abstractmethod
    def create_perturbations(self, batch_size, max_len):
        pass

    @abstractmethod
    def create_graph(self, sess, batch, placeholders):
        pass

    def initial_delta_values(self, shape):

        upper = (self.bit_depth - 1) * self.random_scale
        lower = -self.bit_depth * self.random_scale

        return tf.random.uniform(
            shape,
            minval=lower,
            maxval=upper,
            dtype=tf.float32
        )

    def get_valid_perturbations(self, sess):
        return sess.run(self.final_deltas)

    def get_raw_perturbations(self, sess):
        return sess.run(self.raw_deltas)

    def get_optimisable_variables(self):
        return self.opt_vars

    def apply_box_constraint(self, deltas):

        lower = -self.bit_depth
        upper = self.bit_depth - 1

        return tf.clip_by_value(
            deltas,
            clip_value_min=lower,
            clip_value_max=upper
        )


class IndependentVariables(AbstractPerturbationsGraph):

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

    @abstractmethod
    def create_graph(self, sess, batch, placeholders):
        pass

    @abstractmethod
    def pre_optimisation_updates(self, successess: list):
        pass

    @abstractmethod
    def post_optimisation_updates(self, successess: list):
        pass


class BoxConstraintOnly(IndependentVariables):

    def __init__(
            self,
            sess,
            batch,
            placeholders,
            bit_depth=2 ** 15,
            random_scale=0.002
    ):

        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            random_scale=random_scale,
        )
        self.create_graph(sess, batch, placeholders)

    def create_graph(self, sess, batch, placeholders):

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

        deltas = self.create_perturbations(batch_size, max_len)

        # Mask deltas first so we zero value *any part of the signal* that is
        # zero value padded in the original audio
        deltas *= masks

        # Restrict delta to valid space before applying constraints
        self.perturbations = self.final_deltas = self.apply_box_constraint(
            deltas)

        # clip example to valid range
        self.adversarial_examples = tf.clip_by_value(
            self.perturbations + placeholders.audios,
            clip_value_min=-self.bit_depth,
            clip_value_max=self.bit_depth - 1
        )

        # initialise static variables
        initial_masks = np_arr(
            lcomp(self._gen_mask(act_lengths, max_len)),
            np.float32
        )

        sess.run(masks.assign(initial_masks))

    def pre_optimisation_updates(self, successess: list):
        pass

    def post_optimisation_updates(self, successess: list):
        pass


class ProjectedGradientDescentRounding(BoxConstraintOnly):
    def __init__(
            self,
            sess,
            batch,
            placeholders,
            bit_depth=2 ** 15,
            random_scale=0.002,
    ):

        super().__init__(
            sess,
            batch,
            placeholders,
            bit_depth=bit_depth,
            random_scale=random_scale,
        )

        self.create_graph(sess, batch, placeholders)

    def pre_optimisation_updates(self, successes: list):
        pass

    def post_optimisation_updates(self, successes: list):

        def rounding_func(delta):
            signs = np.sign(delta)
            abs_floor = np.floor(np.abs(delta)).astype(np.int32)
            new_delta = np.round((signs * abs_floor).astype(np.float32))
            return new_delta

        deltas = self.sess.run(self.perturbations)

        assign_ops = [
            self.raw_deltas[i].assign(rounding_func(d)) for i, d in enumerate(deltas)
        ]

        self.sess.run(assign_ops)


class ClippedGradientDescent(IndependentVariables):

    def __init__(
            self,
            sess,
            batch,
            placeholders,
            constraint_cls=None,
            bit_depth=2 ** 15,
            random_scale=0.002,
            r_constant=0.95,
            update_method="geom",
    ):

        super().__init__(
            sess,
            batch,
            bit_depth=bit_depth,
            random_scale=random_scale,
            updateable=True,
        )

        self.create_graph(
            sess,
            batch,
            placeholders,
            constraint_cls=constraint_cls,
            r_constant=r_constant,
            update_method=update_method,
        )

    def create_graph(
            self,
            sess,
            batch,
            placeholders,
            constraint_cls=None,
            r_constant=0.95,
            update_method="geom",
    ):

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

        deltas = self.create_perturbations(batch_size, max_len)

        # Mask deltas first so we zero value *any part of the signal* that is
        # zero value padded in the original audio
        deltas *= masks

        # Restrict delta to valid space before applying constraints
        self.final_deltas = self.apply_box_constraint(deltas)

        self.hard_constraint = constraint_cls(
            self.sess,
            self.batch,
            r_constant=r_constant,
            update_method=update_method
        )

        self.perturbations = self.hard_constraint.clip(
            self.final_deltas
        )

        # clip example to valid range
        self.adversarial_examples = tf.clip_by_value(
            self.perturbations + placeholders.audios,
            clip_value_min=-self.bit_depth,
            clip_value_max=self.bit_depth - 1
        )

        # initialise static variables
        initial_masks = np_arr(
            lcomp(self._gen_mask(act_lengths, max_len)),
            np.float32
        )

        sess.run(masks.assign(initial_masks))

    def pre_optimisation_updates(self, successess: list):

        self.hard_constraint.update(
            self.sess.run(self.perturbations), successess
        )

    def post_optimisation_updates(self, successess: list):
        pass


class ClippedGradientDescentWithProjectedRounding(ClippedGradientDescent):

    def __init__(
            self,
            sess,
            batch,
            placeholders,
            constraint_cls=None,
            bit_depth=2 ** 15,
            random_scale=0.002,
            r_constant=0.95,
            update_method="geom",
    ):
        super().__init__(
                sess,
                batch,
                placeholders,
                constraint_cls=constraint_cls,
                bit_depth=bit_depth,
                random_scale=random_scale,
                r_constant=r_constant,
                update_method=update_method,
        )

    def post_optimisation_updates(self, successes: list):

        def rounding_func(delta):
            signs = np.sign(delta)
            abs_floor = np.floor(np.abs(delta)).astype(np.int32)
            new_delta = np.round((signs * abs_floor).astype(np.float32))
            return new_delta

        deltas = self.sess.run(self.perturbations)

        assign_ops = [
            self.raw_deltas[i].assign(rounding_func(d)) for i, d in enumerate(deltas)
        ]

        self.sess.run(assign_ops)






