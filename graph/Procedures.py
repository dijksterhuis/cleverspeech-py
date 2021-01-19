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

    def update_success(self, decodings, targets):
        """
        Update bound conditions if we've been successful.
        """

        constraint = self.attack.hard_constraint
        new_bounds = list()

        tf_vars = [self.attack.graph.final_deltas, constraint.bounds]
        deltas, bounds = self.tf_run(tf_vars)

        z = zip(decodings, targets, deltas, bounds)
        for idx, (decode, target, delta, bound) in enumerate(z):

            bound = bound[0]
            distance = constraint.analyse(delta)

            if decode == target:
                new_bound = constraint.get_new_bound(bound, distance)
                new_bounds.append([new_bound])
                yield idx, True
            else:
                new_bounds.append([bound])
                yield idx, False

        constraint.update(new_bounds)

    def run(self):
        """
        Do the actual optimisation.
        TODO: Distance metric could be held in AttackGraph or Base?

        :param batch: batch data for the specified model.
        :param steps: total number of steps to run optimisation for.
        :param decode_step: when to check for a successful decoding.
        :return:
            stats for each step of optimisation (loss measurements),
            successful adversarial examples
        """
        attack, g, b = self.attack, self.attack.graph, self.attack.batch

        while self.current_step < self.steps:

            self.current_step += 1
            step = self.current_step

            attack.optimiser.optimise(b.feeds.attack)

            if step % self.decode_step == 0:
                # can use either tf or deepspeech decodings
                # we prefer ds as it's what the actual model would use.
                # It does mean switching to CPU every time we want to do
                # inference but there's not a major hit to performance

                decodings, probs = attack.victim.inference(
                    b,
                    feed=b.feeds.attack,
                    decoder="batch",
                    top_five=False,
                )

                outs = {
                    "step": self.current_step,
                    "data": [],
                }

                targets = b.targets.phrases

                for idx, success in self.update_success(decodings, targets):

                    out = {
                        "idx": idx,
                        "success": success,
                        "decodings": decodings[idx],
                        "target_phrase": b.targets.phrases[idx],
                        "probs": probs
                    }

                    outs["data"].append(out)

                yield outs


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


class StaticBound(Base):
    def update_success(self, decodings, targets):
        """
        *DO NOT* update bound conditions if we've been successful.
        Useful in performing security evaluations at specified distances.
        See `Wild Patterns` -- Biggio et al.
        """

        constraint = self.attack.hard_constraint
        deltas = self.tf_run(self.attack.graph.final_deltas)

        z = zip(
            decodings,
            targets,
            deltas,
        )
        for idx, (decode, target, delta) in enumerate(z):

            distance = constraint.analyse(delta)

            if decode == target:
                success = True
                yield idx, success, distance
            else:
                success = False
                yield idx, success, distance

