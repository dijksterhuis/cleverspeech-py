import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from cleverspeech.graph.losses import Bases


class L2CarliniLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        # N.B. original code did `reduce_mean` on `(advex - original) ** 2`...
        # `tf.reduce_mean` on `deltas` is exactly the same with fewer variables

        l2delta = tf.reduce_mean(attack.perturbations ** 2, axis=1)
        self.loss_fn = l2delta * self.weights


class L2SquaredLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        l2delta = tf.reduce_sum(attack.perturbations ** 2, axis=-1)
        self.loss_fn = l2delta * self.weights


class L2TanhSquaredLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        # N.B. original code did `reduce_mean` on `(advex - original) ** 2`...
        # `tf.reduce_mean` on `deltas` is exactly the same with fewer variables

        l2delta = tf.reduce_sum(
            tf.tanh(attack.perturbations / 2**15) ** 2, axis=-1
        )
        self.loss_fn = l2delta * self.weights


class L2TanhCarliniLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        l2delta = tf.reduce_mean(
            tf.tanh(attack.perturbations / 2**15) ** 2, axis=-1
        )
        self.loss_fn = l2delta * self.weights


class L2Log10Loss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        def log10(x):
            numerator = tf.log(x + 1e-8)
            denominator = tf.log(tf.constant(10.0, dtype=tf.float32))
            return tf.cast(tf.cast(numerator / denominator, dtype=tf.int32), dtype=tf.float32)

        l2delta = tf.reduce_sum(attack.perturbations ** 2, axis=-1)
        self.loss_fn = l2delta / tf.pow(10.0, log10(l2delta) + 1)
        self.loss_fn *= self.weights


class L2SampleLoss(Bases.SimpleWeightings):
    """
    Normalised L2 loss component from https://arxiv.org/abs/1801.01944

    Use the original example's L2 norm for normalisation.
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        assert type(weight_settings) in list, tuple

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        original = attack.placeholders.audios
        delta = attack.perturbations

        self.l2original = tf.reduce_sum(tf.abs(original ** 2), axis=1)
        self.l2delta = tf.abs(delta) ** 2
        self.l2_loss = tf.reduce_sum(self.l2delta, axis=1) / self.l2original

        self.loss_fn = self.l2_loss * self.weights


class LinfLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        l2delta = tf.reduce_max(tf.abs(attack.delta_graph.perturbations), axis=1)
        self.loss_fn = l2delta * self.weights


class LinfTanhLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        l2delta = tf.reduce_max(
            tf.abs(tf.tanh(attack.delta_graph.final_deltas / 2**15)),
            axis=1
        )
        self.loss_fn = l2delta * self.weights


class RootMeanSquareLoss(Bases.SimpleWeightings):
    """
    L2 loss component from https://arxiv.org/abs/1801.01944
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        l2delta = tf.sqrt(tf.reduce_mean(attack.perturbations ** 2, axis=-1))
        self.loss_fn = l2delta * self.weights


class EntropyLoss(Bases.SimpleWeightings):
    """
    Try to minimise the maximum entropy measure as it was used by Lea Schoenherr
    to try to detect adversarial examples.
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0), updateable: bool = False):

        assert type(weight_settings) in list, tuple

        super().__init__(
            attack,
            weight_settings=weight_settings,
            updateable=updateable,
        )

        x = attack.victim.logits

        log_mult_sum = tf.reduce_sum(x * tf.log(x), axis=2)
        neg_max = tf.reduce_max(-log_mult_sum, axis=1)

        self.loss_fn = neg_max * self.weights
