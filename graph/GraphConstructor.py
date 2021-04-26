"""
Constructor classes are interfaces with methods to control how to define an
attack. Most classes interact with this object when accessing other parts of an
attack, e.g. doing inference is usually done by accessing the victim object's
methods here, i.e. attack.victim.inference()

TODO: Rename to AttackGraphConstructors

TODO: Create a new UnboundedAttackGraphConstructor

TODO: Create a new AbstractAttackGraphConstructor

TODO: Can we set up defaults and add classes only when required?

TODO: Rename add_graph method to add_perturbation_sub_graph

TODO: Rename .graph attribute to .perturbations

TODO: Add an add_placeholders method.

--------------------------------------------------------------------------------
"""


import tensorflow as tf

from abc import ABC


class Constructor(ABC):
    """
    Allows a modular yet standardised definition of attacks.

    Individual classes are **always** the first arguments of their respective
    `adder` methods and are passed **without being initialised**. i.e. only pass
    in the raw class, not an instance of a class!

    For example, adding an L2 hard constraint would look like this:

    >>> graph = Constructor()
    >>> graph.add_hard_constraint(
    >>>    cleverspeech.graph.Constraints.L2, # the raw, uninitialised class ref
    >>>    some_arg_for_l2_class,
    >>>    some_other_arg_for_l2_class,
    >>>    some_optional=arg_for_l2_class,
    >>> )

    This `Constructor` class will handle the initialisation of those classes,
    given the args and kwargs that are also passed to the
    `graph.add_hard_constraint` method. Each instance of the added classes can
    then be accessed as attributes of this `Constructor` class instance.

    For example, this line calls the L2 hard constraint's `analyse()` method:

    >>> graph.hard_constraint.analyse(some_perturbation)

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

    TODO: Rename this class to EvasionAttackGraphConstructor

    ----------------------------------------------------------------------------

    :param sess: a tensorflow session object
    :param batch: the current batch of input data to run the attack with,
        a cleverspeech.data.ingress.batch_generators.batch object

    :param feeds: the input feeds for the tensorflow graph (references between
        the tensorflow graph placeholders and batch data), a
        cleverspeech.data.ingress.Feeds object
    """
    def __init__(self, sess, batch, feeds):

        self.batch = batch
        self.feeds = feeds
        self.sess = sess

        self.graph = None
        self.hard_constraint = None
        self.victim = None
        self.loss = None
        self.distance_loss = None
        self.adversarial_loss = None
        self.loss_fn = None
        self.optimiser = None
        self.procedure = None
        self.outputs = None

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
            **kwargs
        )

    def add_graph(self, graph, *args, **kwargs):
        """
        Add a adversarial example variable graph to the attack graph.

        :param graph: an uninitialised reference to a
            cleverspeech.graph.VariableGraphs class, i.e. BatchwiseVariableGraph
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """
        # TODO
        if self.hard_constraint:
            self.graph = graph(
                self.sess,
                self.batch,
                self.hard_constraint,
                *args,
                **kwargs,
            )
        else:
            self.graph = graph(
                self.sess,
                self.batch,
                *args,
                **kwargs,
            )

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
            self.graph.adversarial_examples,
            self.batch,
            *args,
            **kwargs
        )

    def create_loss_fn(self):
        """
        Given the loss classes that have been added to the graph, create the
        final loss function by performing a sum over all the loss classes.

        TODO: Add a ref_fn=sum kwarg.

        :return: None
        """
        assert self.loss is not None
        self.loss_fn = sum(l.loss_fn for l in self.loss)

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

    def add_outputs(self, outputs, *args, **kwargs):
        """
        TODO: This method is not required anymore.

        :param outputs:
        :param args: any args with which are required to initialise the class
        :param kwargs: any optional args with which to initialise the class
        :return: None
        """
        self.outputs = outputs(self, self.batch, *args, **kwargs)

    def run(self, queue, health_check, *args, **kwargs):
        """
        Start running an attack.

        TODO: queue + health_check should be *args, may not want in all cases

        :param queue: a multiprocessing.Queue object to pass data back to the
            separate cleverspeech.data.egress.Writer subprocess.
        :param health_check: a multiprocessing.Pipe object to pass the status of
            and attack back to the cleverspeech.utils.Runtime.AttackSpawner
        :param args: any args with which are required to start running attack
        :param kwargs: any optional args with which to start running attack
        :return:
        """
        self.procedure.run(queue, health_check, *args, **kwargs)

    def update_bound(self, *args, **kwargs):
        """
        Update hard constraint bounds.

        TODO: This method is not used and is very stale.

        :param args: any args with which are required to update a bound
        :param kwargs: any optional args with which to update a bound
        :return: output of the update_bound() method.
        """
        results = self.optimiser.update_bound(*args, **kwargs)
        return results

    def create_feeds(self):
        """
        Run the create_feeds() method on the cleverspeech.data.ingress.Feeds
        object.

        TODO: Run immediately after placeholders have been added.

        :return: None
        """
        self.feeds.create_feeds(self.graph)


class Placeholders(object):
    def __init__(self, batch_size: int, maxlen: int):
        self.audios = tf.placeholder(tf.float32, [batch_size, maxlen], name="new_input")
        self.audio_lengths = tf.placeholder(tf.int32, [batch_size], name='qq_featlens')
        self.targets = tf.placeholder(tf.int32, [batch_size, None], name='qq_targets')
        self.target_lengths = tf.placeholder(tf.int32, [batch_size], name='qq_target_lengths')
