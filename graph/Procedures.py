import tensorflow as tf
import numpy as np
from abc import ABC


class AbstractProcedure(ABC):
    def __init__(self, attack, steps: int = 5000, decode_step: int = 10, loss_update_idx=None):
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
        self.update_loss = loss_update_idx
        if self.update_loss is not None:
            assert type(self.update_loss) in [int, float]

    def init_optimiser_variables(self):

        # We must wait to initialise the optimiser so that we can
        # initialise only the attack variables (i.e. not the deepspeech ones).

        self.attack.optimiser.create_optimiser()

        opt_vars = self.attack.optimiser.variables
        opt_vars += self.attack.graph.opt_vars

        self.attack.sess.run(tf.variables_initializer(opt_vars))

    def tf_run(self, tf_variables):
        sess = self.attack.sess
        feed = self.attack.feeds.attack
        return sess.run(tf_variables, feed_dict=feed)

    @staticmethod
    def success_criteria_check(left, right):
        return True if left == right else False

    def run_on_success(self, idx):

        # update the delta hard constraint
        delta = self.tf_run(self.attack.graph.final_deltas)[idx]
        self.attack.hard_constraint.update_one(delta, idx)

        # update any loss weightings
        if self.update_loss is not None:
            self.attack.loss[self.update_loss].update(self.attack.sess, idx)

    def update_on_success(self, lefts, rights):
        """
        Update bound conditions if we've been successful.
        """

        z = zip(lefts, rights)
        for idx, (left, right) in enumerate(z):

            success = self.success_criteria_check(left, right)

            if success:
                self.run_on_success(idx)
                yield idx, success
            else:
                yield idx, success

    def decode_step_logic(self):

        # ==> every x steps apply integer level rounding prior to decoding.
        # this helps the attacks work against the deepspeech native client api
        # which only accepts tf.int16 type inputs. Although it doesn't work 100%
        # of the time so use `bin/classify.py` to get the true success rate.

        # We could do a line search etc. after running optimisation, but doing
        # things during attack optimisation seems to help it find a solution in
        # integer space all by itself (vs. fiddling with the examples after).

        deltas = self.attack.sess.run(self.attack.graph.raw_deltas)
        self.attack.sess.run(
            self.attack.graph.raw_deltas.assign(tf.round(deltas))
        )

        # can use either tf or deepspeech decodings ("ds" or "batch")
        # "batch" is prefered as it's what the actual model would use.
        # It does mean switching to CPU every time we want to do
        # inference but it's not a major hit to performance

        # keep the top 5 scoring decodings and their probabilities as that might
        # be useful come analysis time...

        top_5_decodings, top_5_probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            decoder="batch",
            top_five=True,
        )

        decodings, probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            decoder="batch",
            top_five=False,
        )

        targets = self.attack.batch.targets["phrases"]

        return {
            "step": self.current_step,
            "data": [
                {
                    "idx": idx,
                    "success": success,
                    "decodings": decodings[idx],
                    "target_phrase": targets[idx],
                    "probs": probs[idx],
                    "top_five_decodings": top_5_decodings[idx],
                    "top_five_probs": top_5_probs[idx],
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

            if self.current_step % self.decode_step == 0 or self.current_step == 0:
                yield self.decode_step_logic()

            attack.optimiser.optimise(attack.feeds.attack)

            self.current_step += 1


class Unbounded(AbstractProcedure):
    """
    Updates bounds on a successful decoding.

    Basically just the Base class with a more useful name.
    """
    def __init__(self, attack, *args, **kwargs):

        """
        Initialise the procedure object then initialise the optimiser
        variables => might be additional tf variables to initialise here.
        """
        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()

    def run_on_success(self, idx):
        # Do nothing
        pass


class UpdateOnDecoding(AbstractProcedure):
    """
    Updates bounds on a successful decoding.

    Basically just the Base class with a more useful name.
    """
    def __init__(self, attack, *args, **kwargs):

        """
        Initialise the procedure object then initialise the optimiser
        variables => might be additional tf variables to initialise here.
        """
        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()


class UpdateOnLoss(AbstractProcedure):
    """
    Updates bounds once the loss has reached a certain point.
    """
    def __init__(self, attack, *args, loss_lower_bound=10.0, **kwargs):
        """
        Initialise the procedure object then initialise the optimiser
        variables => might be additional tf variables to initialise here.
        """

        super().__init__(attack, *args, **kwargs)
        self.loss_bound = loss_lower_bound
        self.init_optimiser_variables()

    def run(self):
        for results in super().run():
            yield results

    @staticmethod
    def success_criteria_check(left, right):
        return True if left <= right else False

    def decode_step_logic(self):

        deltas = self.attack.sess.run(self.attack.graph.raw_deltas)
        self.attack.sess.run(
            self.attack.graph.raw_deltas.assign(tf.round(deltas))
        )

        loss = self.tf_run(self.attack.loss_fn)

        top_5_decodings, top_5_probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            decoder="batch",
            top_five=True,
        )

        decodings, probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            decoder="batch",
            top_five=False,
        )

        target_loss = [self.loss_bound for _ in range(self.attack.batch.size)]
        targets = self.attack.batch.targets["phrases"]

        return {
            "step": self.current_step,
            "data": [
                {
                    "idx": idx,
                    "success": success,
                    "decodings": decodings[idx],
                    "target_phrase": targets[idx],
                    "top_five_decodings": top_5_decodings[idx],
                    "top_five_probs": top_5_probs[idx],
                    "probs": probs[idx]
                }
                for idx, success in self.update_on_success(loss, target_loss)
            ]
        }


class UpdateOnProbs(AbstractProcedure):
    """
    Updates bounds once the loss has reached a certain point.
    """
    def __init__(self, attack, *args, probs_diff=10.0, **kwargs):
        """
        Initialise the procedure object then initialise the optimiser
        variables => might be additional tf variables to initialise here.
        """

        super().__init__(attack, *args, **kwargs)
        self.probs_diff = probs_diff
        self.init_optimiser_variables()

    def run(self):
        for results in super().run():
            yield results

    @staticmethod
    def success_criteria_check(left, right):
        return True if left <= right else False

    def decode_step_logic(self):

        deltas = self.attack.sess.run(self.attack.graph.raw_deltas)
        self.attack.sess.run(
            self.attack.graph.raw_deltas.assign(tf.round(deltas))
        )

        top_5_decodings, top_5_probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            decoder="batch",
            top_five=True,
        )

        decodings, probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            decoder="batch",
            top_five=False,
        )

        target_probs = [self.probs_diff for _ in range(self.attack.batch.size)]
        targets = self.attack.batch.targets["phrases"]

        return {
            "step": self.current_step,
            "data": [
                {
                    "idx": idx,
                    "success": success,
                    "decodings": decodings[idx],
                    "target_phrase": targets[idx],
                    "top_five_decodings": top_5_decodings[idx],
                    "top_five_probs": top_5_probs[idx],
                    "probs": probs[idx]
                }
                for idx, success in self.update_on_success(probs, target_probs)
            ]
        }


class HardcoreMode(UpdateOnLoss):
    def __init__(self, attack, *args, loss_lower_bound=1.0, **kwargs):
        super().__init__(
            attack,
            *args,
            loss_lower_bound=loss_lower_bound,
            **kwargs
        )

    def run_on_success(self, idx):

        # update the delta hard constraint
        delta = self.tf_run(self.attack.graph.final_deltas)[idx]
        self.attack.hard_constraint.update_one(delta, idx)

        # update any loss weightings
        if self.update_loss is not None:
            self.attack.loss[self.update_loss].update(self.attack.sess, idx)

    def run(self):
        """
        Do the actual optimisation.
        """
        attack, g, b = self.attack, self.attack.graph, self.attack.batch

        while True:

            if self.current_step % self.decode_step == 0 or self.current_step == 0:
                yield self.decode_step_logic()

            attack.optimiser.optimise(attack.feeds.attack)

            self.current_step += 1
