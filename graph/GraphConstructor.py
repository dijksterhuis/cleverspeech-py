import tensorflow as tf

from abc import ABC


class Constructor(ABC):
    """
    Allows modular definition and control of attacks
    Inherited by subsequent graphs
    Automatically passes attack graph to loss and optimiser
    """
    def __init__(self, sess, batch):

        self.batch = batch
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

        self.hard_constraint = constraint(
            self.sess,
            self.batch,
            *args,
            **kwargs
        )

    def add_graph(self, graph, *args, **kwargs):
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
        self.victim = model(
            self.sess,
            self.graph.adversarial_examples,
            self.batch,
            *args,
            **kwargs
        )

    def add_distance_loss(self, loss, *args, **kwargs):
        self.distance_loss = loss(self, *args, **kwargs)

    def add_adversarial_loss(self, loss, *args, **kwargs):
        self.adversarial_loss = loss(self, *args, **kwargs)

    def create_loss_fn(self):
        assert self.adversarial_loss is not None

        if self.distance_loss is not None:
            self.loss_fn = self.adversarial_loss.loss_fn + self.distance_loss.loss_fn
        else:
            self.loss_fn = self.adversarial_loss.loss_fn

    # def add_loss(self, loss, *args, **kwargs):
    #     # Note that the attribute `loss_fn` is called here
    #     # This is the nice hack to pass on the attack_graph -- `loss(self, ...)`
    #     new_loss = loss(self, *args, **kwargs)
    #     if self.loss_fn is None:
    #         # tmp change
    #         self.loss = new_loss
    #         self.loss_fn = self.loss.loss_fn
    #     else:
    #         self.loss_fn = new_loss.loss_fn + self.loss_fn

    def add_optimiser(self, optimiser, *args, **kwargs):
        self.optimiser = optimiser(self, *args, **kwargs)

    def optimise(self, *args, **kwargs):
        return self.optimiser.optimise(self.batch, *args, **kwargs)

    def add_procedure(self, procedure, *args, **kwargs):
        self.procedure = procedure(self, *args, **kwargs)

    def add_outputs(self, outputs, *args, **kwargs):
        self.outputs = outputs(self, self.batch, *args, **kwargs)

    def run(self, queue, *args, **kwargs):
        for results in self.procedure.run(*args, **kwargs):
            self.outputs.run(results, queue)

    def update_bound(self, *args, **kwargs):
        results = self.optimiser.update_bound(*args, **kwargs)
        return results


class Placeholders(object):
    def __init__(self, batch_size: int, maxlen: int) -> object:
        self.audios = tf.placeholder(tf.float32, [batch_size, maxlen], name="new_input")
        self.audio_lengths = tf.placeholder(tf.int32, [batch_size], name='qq_featlens')
        self.targets = tf.placeholder(tf.int32, [batch_size, None], name='qq_targets')
        self.target_lengths = tf.placeholder(tf.int32, [batch_size], name='qq_target_lengths')
