"""
Constructor classes are interfaces with methods to control how to define an
attack. Most classes interact with this object when accessing other parts of an
attack, e.g. doing inference is usually done by accessing the victim object's
methods here, i.e. attack.victim.inference()

--------------------------------------------------------------------------------
"""


import tensorflow as tf
from abc import ABC, abstractmethod
from cleverspeech.utils.Utils import log


class AbstractAttackConstructor(ABC):
    """
    Allows a modular yet standardised definition of attacks.

    Individual classes are **always** the first arguments of their respective
    `adder` methods and are passed **without being initialised**. i.e. only pass
    in the raw class, not an instance of a class!

    For example, adding an L2 hard constraint would look like this:

    >>> from cleverspeech import graph
    >>> attack = EvasionAttackConstructor()
    >>> attack.add_hard_constraint(
    >>>    graph.Constraints.L2, # the raw, uninitialised class ref
    >>>    r_constant=0.9,
    >>>    update_method="geom"
    >>> )

    This `Constructor` class will handle the initialisation of those classes,
    given the args and kwargs that are also passed to the
    `graph.add_hard_constraint` method. Each instance of the added classes can
    then be accessed as attributes of this `Constructor` class instance.

    For example, this line calls the L2 hard constraint's `analyse()` method:

    >>> attack.hard_constraint.analyse(attack.perturbations)

    This helps keep everything as standardised as possible and means we can use
    an instance of this `Constructor` class as a common interface for all other
    classes that make up an attack.

    Furthermore, if you take a look at the code you'll see that common patterns
    like passing the adversarial examples to a victim model are **always**
    handled by this `Constructor` class, reducing some of the need to remember
    which arguments need to be passed to which classes.

    Generally speaking, these are the rules for all `adder` methods:

    - anything 100% standard for a type of class is handled by the `Constructor`
    - **args** are required for a specific class, but non-standard for all classes of that type
    - **kwargs** are always optional

    TODO: Add logic to check matching VariableGraph and Optimiser types.

    TODO: Add/update additional methods to control an attack e.g. update_bounds

    TODO: Can we set up defaults and add classes only when required?

    ----------------------------------------------------------------------------

    :param sess: a tensorflow session object
    :param batch: the current batch of input data to run the attack with,
        a cleverspeech.data.ingress.batch_generators.batch object

    :param feeds: the input feeds for the tensorflow graph (references between
        the tensorflow graph placeholders and batch data), a
        cleverspeech.data.ingress.Feeds object
    """
    def __init__(self, sess, batch, feeds, bit_depth=2**15):

        self.batch = batch
        self.feeds = feeds
        self.sess = sess
        self.bit_depth = bit_depth

        self.placeholders = None
        self.hard_constraint = None
        self.delta_graph = None
        self.perturbations = None
        self.adversarial_examples = None
        self.victim = None
        self.loss = None
        self.distance_loss = None
        self.adversarial_loss = None
        self.loss_fn = None
        self.optimiser = None
        self.procedure = None
        self.outputs = None

    def add_placeholders(self, placeholders):
        self.placeholders = placeholders(self.batch)
        self.feeds.create_feeds(self.placeholders)

    def add_hard_constraint(self, constraint, *args, **kwargs):
        """
        Override this method to in child classes if needed.

        :param constraint: reference to the **uninitialised**
            cleverspeech.graph.Constraints class
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """

        pass

    @abstractmethod
    def add_perturbation_subgraph(self, graph, *args, **kwargs):
        """
        Must be implemented in child classes -- add a perturbation graph.

        :param graph: an uninitialised reference to a
            cleverspeech.graph.VariableGraphs class, i.e. BatchwiseVariableGraph
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """
        pass

    def add_victim(self, model, *args, **kwargs):
        """
        Add a victim model class, this class will live in the
        cleverSpeech/models directory!

        :param model: an uninitialised class of a VictimModel
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """
        self.victim = model(
            self.sess,
            self.adversarial_examples,
            self.batch,
            *args,
            **kwargs
        )

    def add_loss(self, loss, *args, **kwargs):
        """
        Add a loss class. Multiple losses can be added to create a combined
        objective function.

        TODO: How to track "update-able" loss weights? kwarg to list attribute?

        :param loss: The cleverspeech.graph.Losses class, uninitialised
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """
        # Note that the attribute `loss_fn` is called here
        new_loss = loss(self, *args, **kwargs)
        if self.loss_fn is None:
            self.loss = [new_loss]
            self.loss_fn = new_loss.loss_fn
        else:
            self.loss = self.loss + [new_loss]
            self.loss_fn = self.loss_fn + new_loss.loss_fn

    def create_loss_fn(self):
        """
        Given the loss classes that have been added to the graph, create the
        final loss function by performing a sum over all the loss classes.

        TODO: Add a ref_fn=sum kwarg.

        :return: None
        """
        assert self.loss is not None
        self.loss_fn = sum(l.loss_fn for l in self.loss)

    def add_optimiser(self, optimiser, *args, **kwargs):
        """
        Add an optimiser.

        TODO: Test type of optimiser against type of perturbation sub graph

        :param optimiser: The cleverspeech.graph.Optimisers class, uninitialised
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """
        self.optimiser = optimiser(self, *args, **kwargs)

    def optimise(self, *args, **kwargs):
        """
        Run the optimisation step.

        :param args: any args which are required for the Optimiser's
            optimisation method.
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """
        return self.optimiser.optimise(self.batch, *args, **kwargs)

    def add_procedure(self, procedure, *args, **kwargs):
        """
        Add a cleverspeech.graph.Procedure to run the attack with.

        :param procedure: a cleverspeech.graph.Procedure class, uninitialised
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """
        self.procedure = procedure(self, *args, **kwargs)

    def run(self, *args, **kwargs):
        """
        Start running an attack.

        :param args: any args with which are required to start running attack
        :param kwargs: any optional args with which to start running attack
        :return: True given some condition on when to return results.
        """
        for x in self.procedure.run(*args, **kwargs):
            yield x

    def validate(self):
        """
        Do an initial decoding to verify everything is working
        """
        decodings, probs = self.victim.inference(
            self.batch,
            feed=self.feeds.examples,
            decoder="batch"
        )
        z = zip(self.batch.audios["basenames"], probs, decodings)
        s = ["{}\t{:.3f}\t{}".format(b, p, d) for b, p, d in z]
        log("Initial decodings:", '\n'.join(s), wrap=True)

        s = ["{:.0f}".format(x) for x in self.batch.audios["real_feats"]]
        log("Real Features: ", "\n".join(s), wrap=True)

        s = ["{:.0f}".format(x) for x in self.batch.audios["ds_feats"]]
        log("DS Features: ", "\n".join(s), wrap=True)

        s = ["{:.0f}".format(x) for x in self.batch.audios["n_samples"]]
        log("Real Samples: ", "\n".join(s), wrap=True)


class EvasionAttackConstructor(AbstractAttackConstructor):
    """
    Construct an evasion attack with a hard constraint for security evaluations.
    ----------------------------------------------------------------------------

    :param sess: a tensorflow session object
    :param batch: the current batch of input data to run the attack with,
        a cleverspeech.data.ingress.batch_generators.batch object

    :param feeds: the input feeds for the tensorflow graph (references between
        the tensorflow graph placeholders and batch data), a
        cleverspeech.data.ingress.Feeds object
    """
    def __init__(self, sess, batch, feeds, bit_depth=2**15):

        super().__init__(sess, batch, feeds, bit_depth=bit_depth)

    def add_hard_constraint(self, constraint, *args, **kwargs):
        """
        Add some hard constraint which we will clip the adversarial examples
        with. Often this constraint will be iteratively reduced over time.

        :param constraint: reference to the **uninitialised**
            cleverspeech.graph.Constraints class
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """

        self.hard_constraint = constraint(
            self.sess,
            self.batch,
            *args,
            **kwargs,
            bit_depth=self.bit_depth
        )

    def add_perturbation_subgraph(self, graph, *args, **kwargs):
        """
        Add a perturbation variables to the attack graph.

        :param graph: an uninitialised reference to a
            cleverspeech.graph.VariableGraphs class, i.e. BatchwiseVariableGraph
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """
        self.delta_graph = graph(
            self.sess,
            self.batch,
            *args,
            **kwargs,
        )

        self.perturbations = self.hard_constraint.clip(
            self.delta_graph.final_deltas
        )

        # clip example to valid range
        self.adversarial_examples = tf.clip_by_value(
            self.perturbations + self.placeholders.audios,
            clip_value_min=-self.bit_depth,
            clip_value_max=self.bit_depth - 1
        )


class UnboundedAttackConstructor(AbstractAttackConstructor):
    """
    Construct an unbounded attack with no constraints.
    ----------------------------------------------------------------------------

    :param sess: a tensorflow session object
    :param batch: the current batch of input data to run the attack with,
        a cleverspeech.data.ingress.batch_generators.batch object
    :param feeds: the input feeds for the tensorflow graph (references between
        the tensorflow graph placeholders and batch data), a
        cleverspeech.data.ingress.Feeds object
    """

    def __init__(self, sess, batch, feeds, bit_depth=2 ** 15):

        super().__init__(sess, batch, feeds, bit_depth=bit_depth)

    def add_perturbation_subgraph(self, graph, *args, **kwargs):
        """
        Add a perturbation variables to the attack graph.

        :param graph: an uninitialised reference to a
            cleverspeech.graph.VariableGraphs class, i.e. BatchwiseVariableGraph
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """
        self.delta_graph = graph(
            self.sess,
            self.batch,
            *args,
            **kwargs,
        )

        self.perturbations = self.delta_graph.final_deltas

        # clip example to valid range
        self.adversarial_examples = tf.clip_by_value(
            self.perturbations + self.placeholders.audios,
            clip_value_min=-self.bit_depth,
            clip_value_max=self.bit_depth - 1
        )


class CTCPathSearchConstructor(AbstractAttackConstructor):
    """
    Construct an unbounded attack with no constraints.
    ----------------------------------------------------------------------------

    :param sess: a tensorflow session object
    :param batch: the current batch of input data to run the attack with,
        a cleverspeech.data.ingress.batch_generators.batch object
    :param feeds: not used, only included due to AbstractConstructor
    """

    def __init__(self, sess, batch, feeds, bit_depth=2 ** 15):

        super().__init__(sess, batch, None, bit_depth=bit_depth)
        self.graph = None

    def add_perturbation_subgraph(self, graph, *args, **kwargs):
        """
        Add a perturbation variables to the attack graph.

        :param graph: an uninitialised reference to a
            cleverspeech.graph.VariableGraphs class, i.e. BatchwiseVariableGraph
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """
        self.graph = graph(
            self.sess,
            self.batch,
            *args,
            **kwargs,
        )
