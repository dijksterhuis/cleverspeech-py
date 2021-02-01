import tensorflow as tf

from abc import ABC


class Base(ABC):
    def __init__(self, attack, steps: int = 5000, decode_step: int = 10):
        """
        Base class that sets up a wealth of stuff to execute the attack over a
        number of iterations.

        *DO NOT USE IN THE ATTACK GRAPH!* -- define a class on top of this one.
        See examples below.

        """

        assert type(steps) in [float, int]
        assert type(decode_step) in [float, int]
        assert steps > decode_step

        assert attack.optimiser is not None

        self.attack = attack

        self.steps, self.decode_step = steps + 1, decode_step
        self.current_step = 0

    def tf_run(self, tf_variables):
        sess = self.attack.sess
        feed = self.attack.batch.feeds.attack
        return sess.run(tf_variables, feed_dict=feed)

    @staticmethod
    def success_criteria_check(left, right):
        return True if left == right else False

    def update_on_success(self, lefts, rights):
        """
        Update bound conditions if we've been successful.
        """

        deltas = self.tf_run(self.attack.graph.final_deltas)

        z = zip(lefts, rights, deltas)
        for idx, (left, right, delta) in enumerate(z):

            success = self.success_criteria_check(left, right)

            if success:
                self.attack.hard_constraint.update_one(delta, idx)
                yield idx, success
            else:
                yield idx, success

    def decode_step_logic(self):

        # can use either tf or deepspeech decodings ("ds" or "batch")
        # "batch" is prefered as it's what the actual model would use.
        # It does mean switching to CPU every time we want to do
        # inference but there's not a major hit to performance

        decodings, probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.batch.feeds.attack,
            decoder="batch",
            top_five=False,
        )

        targets = self.attack.batch.targets.phrases

        return {
            "step": self.current_step,
            "data": [
                {
                    "idx": idx,
                    "success": success,
                    "decodings": decodings[idx],
                    "target_phrase": targets[idx],
                    "probs": probs
                }
                for idx, success in self.update_on_success(decodings, targets)
            ]
        }

    def run(self):
        """
        Do the actual optimisation.
        """
        attack, g, b = self.attack, self.attack.graph, self.attack.batch

        while self.current_step < self.steps:

            self.current_step += 1
            step = self.current_step

            attack.optimiser.optimise(b.feeds.attack)

            if step % self.decode_step == 0:
                yield self.decode_step_logic()


class UpdateBound(Base):
    def __init__(self, attack, *args, **kwargs):
        """
        Initialise the evaluation procedure.

        :param attack_graph: The current attack graph perform optimisation with.
        """

        super().__init__(attack, *args, **kwargs)

        # We must wait until now to initialise the optimiser so that we can
        # initialise only the attack variables (i.e. not the deepspeech ones).
        start_vars = set(x.name for x in tf.global_variables())

        attack.optimiser.create_optimiser()

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # TODO: New base class that holds vars to be *initialised* rather than opt_vars attributes.
        attack.sess.run(tf.variables_initializer(new_vars + attack.graph.opt_vars))

    def run(self):
        for results in super().run():
            yield results



