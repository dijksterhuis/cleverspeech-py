import tensorflow as tf
from cleverspeech.utils.Utils import lcomp
from abc import ABC, abstractmethod


class AbstractOptimiser(ABC):
    def __init__(self, attack, learning_rate=10.0):
        """
        :param attack: The current attack graph perform optimisation with.
        :param learning_rate: Learning rate for optimiser (currently Adam)
        """

        self.attack = attack
        self.learning_rate = learning_rate

        self.train = None
        self.variables = None
        self.optimizer = None
        self.optimizers = []

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

    @abstractmethod
    def create_optimiser(self):
        pass


class AbstractIndependentOptimiser(AbstractOptimiser):
    def __init__(self, attack, learning_rate=10.0):

        super().__init__(attack, learning_rate)

    def create_optimiser(self):
        """
        Manage the computation of gradients from the loss and the delta variable
        """
        train_ops = []
        self.variables = {}

        for idx, opt in enumerate(self.optimizers):

            grad_var = opt.compute_gradients(
                self.attack.loss_fn,
                [self.attack.graph.opt_vars[idx]],
                colocate_gradients_with_ops=True
            )

            assert None not in lcomp(grad_var, i=0)
            training_op = opt.apply_gradients(grad_var)
            train_ops.append(training_op)

            self.variables[idx] = opt.variables()

        self.train = tf.group(train_ops)


class AbstractBatchwiseOptimiser(AbstractOptimiser):
    def __init__(self, attack, learning_rate=10.0):
        super().__init__(attack, learning_rate)

    def create_optimiser(self):
        """
        Manage the computation of gradients from the loss and the delta variable
        """
        self.variables = {}

        grad_var = self.optimizer.compute_gradients(
            self.attack.loss_fn,
            self.attack.graph.opt_vars,
            colocate_gradients_with_ops=True
        )

        assert None not in lcomp(grad_var, i=0)
        self.variables[0] = self.optimizer.variables()

        self.train = self.optimizer.apply_gradients(grad_var)


class GradientDescentIndependentOptimiser(AbstractIndependentOptimiser):
    def __init__(self, attack, learning_rate=10.0):
        """
        Create gradient descent optimiser.
        """

        super().__init__(attack, learning_rate)

        self.optimizers = []

        for idx in range(self.attack.batch.size):

            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate,
            )

            self.optimizers.append(optimizer)


class AdamIndependentOptimiser(AbstractIndependentOptimiser):
    def __init__(self, attack, learning_rate=10.0, beta1=0.9, beta2=0.999, epsilon=1e-8,):
        """
        Create an Adam optimiser.
        """
        super().__init__(attack, learning_rate)

        self.beta1s = []
        self.beta2s = []
        self.epsilons = []
        self.learning_rates = []
        self.optimizers = []

        for idx in range(self.attack.batch.size):

            optimizer = tf.train.AdamOptimizer(
                learning_rate=float(learning_rate),
                beta1=float(beta1),
                beta2=float(beta2),
                epsilon=float(epsilon),
            )
            self.optimizers.append(optimizer)


class AdaGradIndependentOptimiser(AbstractIndependentOptimiser):
    def __init__(self, attack, learning_rate=10.0, momementum=0.9):
        """
        Create an AdaGrad optimiser.
        """
        super().__init__(attack, learning_rate)

        self.momentum = momementum
        self.optimizers = []

        for idx in range(self.attack.batch.size):
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate,
            )
            self.optimizers.append(optimizer)


class RMSPropIndependentOptimiser(AbstractIndependentOptimiser):
    def __init__(self, attack, learning_rate=10.0, momementum=0.9):
        """
        Create a RMSProp optimiser.
        """

        super().__init__(attack, learning_rate)

        self.momentum = momementum

        for idx in range(self.attack.batch.size):
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate,
            )
            self.optimizers.append(optimizer)


class MomentumIndependentOptimiser(AbstractIndependentOptimiser):
    def __init__(self, attack, learning_rate=10.0, momentum=0.9):
        """
        Create a Momentum optimiser.
        """
        super().__init__(attack, learning_rate)

        self.momentum = momentum

        for idx in range(self.attack.batch.size):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
            )
            self.optimizers.append(optimizer)


class GradientDescentBatchwiseOptimiser(AbstractBatchwiseOptimiser):
    def __init__(self, attack, learning_rate=10.0):
        """
        Create gradient descent optimiser.
        """

        super().__init__(attack, learning_rate)

        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate,
        )


class AdamBatchwiseOptimiser(AbstractBatchwiseOptimiser):
    def __init__(self, attack, learning_rate=10.0, beta1=0.9, beta2=0.999, epsilon=1e-8,):
        """
        Create an Adam optimiser.
        """
        super().__init__(attack, learning_rate)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=float(learning_rate),
            beta1=float(beta1),
            beta2=float(beta2),
            epsilon=float(epsilon),
        )


class AdaGradBatchwiseOptimiser(AbstractBatchwiseOptimiser):
    def __init__(self, attack, learning_rate=10.0, momementum=0.9):
        """
        Create an AdaGrad optimiser.
        """
        super().__init__(attack, learning_rate)

        self.momentum = momementum

        self.optimizer = tf.train.AdagradOptimizer(
            learning_rate=self.learning_rate,
        )


class RMSPropBatchwiseOptimiser(AbstractBatchwiseOptimiser):
    def __init__(self, attack, learning_rate=10.0, momementum=0.9):
        """
        Create a RMSProp optimiser.
        """

        super().__init__(attack, learning_rate)

        self.momentum = momementum

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
        )


class MomentumBatchwiseOptimiser(AbstractBatchwiseOptimiser):
    def __init__(self, attack, learning_rate=10.0, momentum=0.9):
        """
        Create a Momentum optimiser.
        """
        super().__init__(attack, learning_rate)

        self.momentum = momentum

        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
        )
