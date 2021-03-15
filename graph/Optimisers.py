import tensorflow as tf
from cleverspeech.utils.Utils import lcomp
from abc import ABC


class AbstractOptimiser(ABC):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
    ):
        """
        :param attack: The current attack graph perform optimisation with.
        :param learning_rate: Learning rate for optimiser (currently Adam)
        """

        self.attack = attack
        self.learning_rate = learning_rate

        self.train = None
        self.variables = None
        self.optimizer = None

    def create_optimiser(self):
        """
        Manage the computation of gradients from the loss and the delta variable
        """

        grad_var = self.optimizer.compute_gradients(
            self.attack.loss_fn,
            self.attack.graph.opt_vars,
            colocate_gradients_with_ops=True,
        )
        assert None not in lcomp(grad_var, i=0)
        self.train = self.optimizer.apply_gradients(grad_var)
        self.variables = self.optimizer.variables()

    def optimise(self, feed):
        """
        Run the optimisation operation.

        The second line resets the RNN cell's state so the current optimisation
        step is not affected by the leftover cell state from previous
        optimisation steps.

        :param feed: attack feed dict with relevant placeholder and data mapping
        """
        self.attack.victim.reset_state()
        self.attack.sess.run(self.train, feed_dict=feed)
        self.attack.victim.reset_state()


class GradientDescentOptimiser(AbstractOptimiser):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
    ):
        """
        Create gradient descent optimiser.
        """

        super().__init__(attack, learning_rate)

        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate,
        )


class AdamOptimiser(AbstractOptimiser):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-8,
    ):
        """
        Create an Adam optimiser.
        """
        super().__init__(attack, learning_rate)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
        )


class AdaGradOptimiser(AbstractOptimiser):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
            momementum: float = 0.9,
    ):
        """
        Create an AdaGrad optimiser.
        """
        super().__init__(attack, learning_rate)

        self.momentum = momementum

        self.optimizer = tf.train.AdagradOptimizer(
            learning_rate=self.learning_rate,
        )


class RMSPropOptimiser(AbstractOptimiser):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
            momementum: float = 0.9,
    ):
        """
        Create a RMSProp optimiser.
        """

        super().__init__(attack, learning_rate)

        self.momentum = momementum

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
        )


class MomentumOptimiser(AbstractOptimiser):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
            momentum: float = 0.9,
    ):
        """
        Create a Momentum optimiser.
        """
        super().__init__(attack, learning_rate)

        self.momentum = momentum

        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
        )


