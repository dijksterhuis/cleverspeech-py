"""
These are tensorflow optimisers with some specific modifications that we need to
get things working.

Batchwise optimisers should only be used with the BatchwiseVariableGraph.
Independent optimisers should only be used with the IndependentVariableGraph.

TODO: Auto handle batch vs. indy depending on the pert. sub graph?

TODO: self.attack.delta_graph.opt_vars -> tf.Graph.TRAINABLE_VARIABLES ???

--------------------------------------------------------------------------------
"""


import tensorflow as tf
from cleverspeech.utils.Utils import lcomp, log, Logger
from abc import ABC, abstractmethod


class AbstractOptimiser(ABC):
    """
    :param attack: The current attack graph perform optimisation with.
    :param learning_rate: Learning rate for optimiser (currently Adam)
    """
    def __init__(self, attack, learning_rate=10.0):

        self.attack = attack
        self.learning_rate = learning_rate

        assert attack.loss_fn is not None

        self.train = None
        self.variables = None
        self.gradients = None
        self.optimizer = None
        self.optimizers = []

    def optimise(self, feed):
        """
        Run the optimisation operation.

        Resets the RNN cell's state before and after optimisation. Means that
        other parts of the attack are not affected by the leftover cell state
        from previous optimisation steps.

        :param feed: attack feed dict with relevant placeholder and data mapping
        """
        self.attack.victim.reset_state()
        self.attack.sess.run(self.train, feed_dict=feed)
        self.attack.victim.reset_state()

    @abstractmethod
    def create_optimiser(self):
        pass


class AbstractIndependentOptimiser(AbstractOptimiser):
    """
    Creates B optimisers for all perturbations in a batch of size B.
    """
    def __init__(self, attack, learning_rate=10.0):

        super().__init__(attack, learning_rate)

    def create_optimiser(self):
        """
        Manage the computation of gradients from the loss and the delta variable
        """
        train_ops = []
        self.variables = {}

        if len(self.optimizers) > 10:
            Logger.warn(
                "Large batches increase time to process ops ...", timings=True
            )

        s = "Will load {n} total train ops ... ".format(n=len(self.optimizers))
        Logger.info(s, timings=True, postfix="\n")

        for idx, opt in enumerate(self.optimizers):

            grad_var = opt.compute_gradients(
                self.attack.loss_fn,
                [self.attack.delta_graph.opt_vars[idx]],
                colocate_gradients_with_ops=True
            )

            assert None not in lcomp(grad_var, i=0)
            training_op = opt.apply_gradients(grad_var)
            train_ops.append(training_op)
            gradients.append(grad_var[0][0])
            self.variables[idx] = opt.variables()

            if (idx + 1) % 10 == 0:
                Logger.info(
                    "{n} train ops loaded ...".format(n=idx + 1), timings=True
                )

        self.train = tf.group(train_ops)
        self.gradients = tf.stack(gradients, axis=0)

        Logger.info(
            "All {n} train ops loaded.".format(n=len(self._train_ops_as_list)),
            timings=True
        )


class AbstractBatchwiseOptimiser(AbstractOptimiser):
    """
    Creates one optimiser for all perturbations in a batch.
    """
    def __init__(self, attack, learning_rate=10.0):
        super().__init__(attack, learning_rate)

    def create_optimiser(self):
        """
        Manage the computation of gradients from the loss and the delta variable
        """
        self.variables = {}

        grad_var = self.optimizer.compute_gradients(
            self.attack.loss_fn,
            self.attack.delta_graph.opt_vars,
            colocate_gradients_with_ops=True
        )

        assert None not in lcomp(grad_var, i=0)
        self.train = self.optimizer.apply_gradients(grad_var)

        self.variables = {0: self.optimizer.variables()}
        self.gradients = grad_var[0]


class GradientDescentIndependentOptimiser(AbstractIndependentOptimiser):
    """
    Gradient descent optimiser.
    """
    def __init__(self, attack, learning_rate=10.0):

        super().__init__(attack, learning_rate)

        self.optimizers = []

        for idx in range(self.attack.batch.size):

            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate,
            )

            self.optimizers.append(optimizer)


class AdamIndependentOptimiser(AbstractIndependentOptimiser):
    """
    Create an Adam optimiser.
    """
    def __init__(self, attack, learning_rate=10.0, beta1=0.9, beta2=0.999, epsilon=1e-8,):
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
    """
    Create an AdaGrad optimiser.
    """
    def __init__(self, attack, learning_rate=10.0, momementum=0.9):

        super().__init__(attack, learning_rate)

        self.momentum = momementum
        self.optimizers = []

        for idx in range(self.attack.batch.size):
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=self.learning_rate,
            )
            self.optimizers.append(optimizer)


class RMSPropIndependentOptimiser(AbstractIndependentOptimiser):
    """
    Create a RMSProp optimiser.
    """
    def __init__(self, attack, learning_rate=10.0, momementum=0.9):

        super().__init__(attack, learning_rate)

        self.momentum = momementum

        for idx in range(self.attack.batch.size):
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate,
            )
            self.optimizers.append(optimizer)


class MomentumIndependentOptimiser(AbstractIndependentOptimiser):
    """
    Create a Momentum optimiser.
    """
    def __init__(self, attack, learning_rate=10.0, momentum=0.9):
        super().__init__(attack, learning_rate)

        self.momentum = momentum

        for idx in range(self.attack.batch.size):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
            )
            self.optimizers.append(optimizer)


class GradientDescentBatchwiseOptimiser(AbstractBatchwiseOptimiser):
    """
    Create gradient descent optimiser.
    """
    def __init__(self, attack, learning_rate=10.0):

        super().__init__(attack, learning_rate)

        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate,
        )


class AdamBatchwiseOptimiser(AbstractBatchwiseOptimiser):
    """
    Create an Adam optimiser.
    """
    def __init__(self, attack, learning_rate=10.0, beta1=0.9, beta2=0.999, epsilon=1e-8,):
        super().__init__(attack, learning_rate)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=float(learning_rate),
            beta1=float(beta1),
            beta2=float(beta2),
            epsilon=float(epsilon),
        )


class AdaGradBatchwiseOptimiser(AbstractBatchwiseOptimiser):
    """
    Create an AdaGrad optimiser.
    """
    def __init__(self, attack, learning_rate=10.0, momementum=0.9):
        super().__init__(attack, learning_rate)

        self.momentum = momementum

        self.optimizer = tf.train.AdagradOptimizer(
            learning_rate=self.learning_rate,
        )


class RMSPropBatchwiseOptimiser(AbstractBatchwiseOptimiser):
    """
    Create a RMSProp optimiser.
    """
    def __init__(self, attack, learning_rate=10.0, momementum=0.9):

        super().__init__(attack, learning_rate)

        self.momentum = momementum

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
        )


class MomentumBatchwiseOptimiser(AbstractBatchwiseOptimiser):
    """
    Create a Momentum optimiser.
    """
    def __init__(self, attack, learning_rate=10.0, momentum=0.9):
        super().__init__(attack, learning_rate)

        self.momentum = momentum

        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
        )
