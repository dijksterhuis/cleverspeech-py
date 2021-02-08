import tensorflow as tf
from cleverspeech.utils.Utils import lcomp
from abc import ABC, abstractmethod


class Base(ABC):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
    ):
        """
        Initialise the optimiser.

        :param attack: The current attack graph perform optimisation with.
        :param learning_rate: Learning rate for optimiser (currently Adam)
        """

        self.attack = attack
        self.learning_rate = learning_rate
        self.train = None
        self.variables = None

    @abstractmethod
    def create_optimiser(self):
        pass

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


class AdamOptimiser(Base):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-8,
    ):
        """
        Initialise the optimiser.

        :param attack_graph: The current attack graph perform optimisation with.
        :param learning_rate: Learning rate for optimiser (currently Adam)
        """
        super().__init__(attack, learning_rate)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def create_optimiser(self):

        adv_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
        )

        grad_var = adv_optimizer.compute_gradients(
            self.attack.loss_fn,
            self.attack.graph.opt_vars,
            colocate_gradients_with_ops=True,
        )
        assert None not in lcomp(grad_var, i=0)
        self.train = adv_optimizer.apply_gradients(grad_var)
        self.variables = adv_optimizer.variables()


class AdaGradOptimiser(Base):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
            momementum: float = 0.9,
    ):
        """
        Initialise the optimiser.

        :param attack_graph: The current attack graph perform optimisation with.
        :param learning_rate: Learning rate for optimiser (currently Adam)
        """
        super().__init__(attack, learning_rate)

        self.momentum = momementum

    def create_optimiser(self):

        adv_optimizer = tf.train.AdagradOptimizer(
            learning_rate=self.learning_rate,
        )

        grad_var = adv_optimizer.compute_gradients(
            self.attack.loss_fn,
            self.attack.graph.opt_vars,
            colocate_gradients_with_ops=True,
        )
        assert None not in lcomp(grad_var, i=0)
        self.train = adv_optimizer.apply_gradients(grad_var)
        self.variables = adv_optimizer.variables()


class RMSPropOptimiser(Base):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
            momementum: float = 0.9,
    ):
        """
        Initialise the optimiser.

        :param attack_graph: The current attack graph perform optimisation with.
        :param learning_rate: Learning rate for optimiser (currently Adam)
        """
        super().__init__(attack, learning_rate)

        self.momentum = momementum

    def create_optimiser(self):

        adv_optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
        )

        grad_var = adv_optimizer.compute_gradients(
            self.attack.loss_fn,
            self.attack.graph.opt_vars,
            colocate_gradients_with_ops=True,
        )
        assert None not in lcomp(grad_var, i=0)
        self.train = adv_optimizer.apply_gradients(grad_var)
        self.variables = adv_optimizer.variables()


class MomentumOptimiser(Base):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
            momementum: float = 0.9,
    ):
        """
        Initialise the optimiser.

        :param attack_graph: The current attack graph perform optimisation with.
        :param learning_rate: Learning rate for optimiser (currently Adam)
        """
        super().__init__(attack, learning_rate)

        self.momentum = momementum

    def create_optimiser(self):

        adv_optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
        )

        grad_var = adv_optimizer.compute_gradients(
            self.attack.loss_fn,
            self.attack.graph.opt_vars,
            colocate_gradients_with_ops=True,
        )
        assert None not in lcomp(grad_var, i=0)
        self.train = adv_optimizer.apply_gradients(grad_var)
        self.variables = adv_optimizer.variables()


class CoordinateAdamOptimiser(AdamOptimiser):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-8,
    ):
        """
        Initialise the optimiser.

        :param attack_graph: The current attack graph perform optimisation with.
        :param learning_rate: Learning rate for optimiser (currently Adam)
        """
        super().__init__(attack, learning_rate, beta1, beta2, epsilon)

    def create_optimiser(self):
        adv_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
        )

        optimisers = []

        for variable in self.attack.graph.opt_vars:
            grad_var = adv_optimizer.compute_gradients(
                self.attack.loss_fn,
                variable,
                colocate_gradients_with_ops=True,
            )
            assert None not in lcomp(grad_var, i=0)
            optimisers.append(adv_optimizer.apply_gradients(grad_var))

        self.train = tf.group(optimisers)
        self.variables = adv_optimizer.variables()


class GradientDescentOptimiser(Base):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
    ):
        """
        Initialise the optimiser.

        :param attack: The current attack graph perform optimisation with.
        :param learning_rate: Learning rate for optimiser (currently Adam)
        """

        super().__init__(attack, learning_rate)

    def create_optimiser(self):

        adv_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate,
        )

        grad_var = adv_optimizer.compute_gradients(
            self.attack.loss_fn,
            self.attack.graph.opt_vars,
            colocate_gradients_with_ops=True,
        )
        assert None not in lcomp(grad_var, i=0)
        self.train = adv_optimizer.apply_gradients(grad_var)
        self.variables = adv_optimizer.variables()


class CoordinateGradientDescentOptimiser(Base):
    def __init__(
            self,
            attack,
            learning_rate: float = 10.0,
    ):
        """
        Initialise the optimiser.

        :param attack: The current attack graph perform optimisation with.
        :param learning_rate: Learning rate for optimiser (currently Adam)
        """
        super().__init__(attack, learning_rate)

    def create_optimiser(self):

        adv_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate,
        )

        optimisers = []

        for variable in self.attack.graph.opt_vars:

            grad_var = adv_optimizer.compute_gradients(
                self.attack.loss_fn,
                variable,
                colocate_gradients_with_ops=True,
            )
            assert None not in lcomp(grad_var, i=0)
            optimisers.append(adv_optimizer.apply_gradients(grad_var))

        self.train = tf.group(optimisers)
        self.variables = adv_optimizer.variables()



