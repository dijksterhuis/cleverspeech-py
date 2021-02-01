import tensorflow as tf
import numpy as np

from cleverspeech.graph.GraphConstructor import Placeholders
from cleverspeech.utils.Utils import np_arr, np_zero, np_one, lcomp


class SimpleAttack:
    def __init__(self, sess, batch, hard_constraint):

        """
        Attack graph from 2018 Carlini & Wager Targeted Audio Attack.
        # TODO: norm should be a class defined in Audio.Distance
        # TODO: Distance classes should have `bound()` and `analyse()` methods

        :param sess: tf.Session which will be used in the attack.
        :param tau: the upper bound for the size of the perturbation
        :param batch: The batch of fata
        :param synthesis: the `Audio.Synthesis` class which generates the perturbation
        :param constraint: the `Audio.Distance` class being used for the attack

        """
        batch_size = batch.size
        max_len = batch.audios.max_length
        act_lengths = batch.audios.actual_lengths

        self.placeholders = Placeholders(batch_size, max_len)

        masks = tf.Variable(
            tf.zeros([batch_size, max_len]),
            trainable=False,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_masks'
        )

        # Generate the delta synth parameter objects which we will optimise
        raw_deltas = tf.Variable(
            tf.zeros([batch.size, batch.audios.max_length], dtype=tf.float32),
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_delta'
        )

        # Mask deltas first so we zero value *any part of the signal* that is
        # zero value padded in the original audio
        deltas = raw_deltas * masks

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

        self.opt_vars = [raw_deltas]


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

