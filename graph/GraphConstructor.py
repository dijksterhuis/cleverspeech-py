"""
Constructor classes are interfaces with methods to control how to define an
attack. Most classes interact with this object when accessing other parts of an
attack, e.g. doing inference is usually done by accessing the victim object's
methods here, i.e. attack.victim.inference()

--------------------------------------------------------------------------------
"""


from cleverspeech.utils.Utils import log


class Feeds:
    examples = None
    attack = None


class Constructor:
    def __init__(self, sess, batch, settings, bit_depth=1.0):

        self.batch = batch
        self.sess = sess
        self.bit_depth = bit_depth
        self.settings = settings

        self.path_search = None
        self.feeds = Feeds()
        self.placeholders = None
        self.box_constraint = None
        self.size_constraint = None
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

    def add_path_search(self, path_search_cls, *args, **kwargs):

        if self.placeholders is not None:
            raise AttributeError(
                "Path search *must* be declared at the start of the graph"
            )

        self.path_search = path_search_cls(self.batch, *args, **kwargs)
        self.path_search.create()

    def add_placeholders(self, placeholders):

        self.placeholders = placeholders(self.batch)

        # TODO: hackity-hack-hacky
        self.feeds.examples = self.placeholders.examples_feed
        self.feeds.attack = self.placeholders.attacks_feed

    def add_box_constraint(self, box_constraint, *args, **kwargs):
        self.box_constraint = box_constraint(*args, **kwargs)

    def add_size_constraint(self, size_constraint, *args, **kwargs):
        self.size_constraint = size_constraint(
            self.sess, self.batch, *args, **kwargs
        )

    def add_perturbation_subgraph(self, graph, *args, **kwargs):
        self.delta_graph = graph(
            self.sess,
            self.batch,
            *args,
            **kwargs,
        )

    def create_adversarial_examples(self):

        self.perturbations = self.delta_graph.deltas

        if self.size_constraint is not None:
            self.perturbations = self.size_constraint.clip(
                self.perturbations
            )

        if self.box_constraint is not None:
            self.perturbations = self.box_constraint.clip(
                self.perturbations
            )
            self.adversarial_examples = self.box_constraint.clip(
                self.perturbations + self.placeholders.audios
            )

        else:
            self.adversarial_examples = self.perturbations + self.placeholders.audios

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

        if self.loss is None:
            self.loss = [new_loss]
        else:
            self.loss.append(new_loss)

    def create_loss_fn(self):
        """
        Given the loss classes that have been added to the graph, create the
        final loss function by performing a sum over all the loss classes.

        TODO: Add a ref_fn=sum kwarg.

        :return: None
        """
        assert self.loss is not None
        self.loss_fn = sum([l.loss_fn for l in self.loss])

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

        probs_rounded = [round(p, 2) for p in probs]

        headers = "BASENAME|FEATS|BATCH_FEATS|LEN|PROBS|DECODING|TARGET_ID|TARGET|N_CHARS"

        z = zip(
            self.batch.audios["basenames"],
            self.batch.audios["real_feats"],
            self.batch.audios["ds_feats"],
            self.batch.audios["n_samples"],
            probs_rounded,
            decodings,
            self.batch.targets["row_ids"],
            self.batch.targets["phrases"],
            self.batch.targets["lengths"],
        )
        s = ["|".join("{}".format(y) for y in x) for x in z]

        log("Running attack for this data:", wrap=False)

        log(
            headers,
            '\n'.join(s),
            wrap=False,
            outdir=self.settings["outdir"],
            fname="batch.psv"
        )

        log("Wrote batch data to psv file", wrap=True)

