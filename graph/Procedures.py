"""
Procedures govern **how** an attack will be run, i.e. how to update a bound (if
doing an evasion attack), when to run a decoding step (if at all) and such like.

Generally speaking, Procedures do a lot of the heavy lifting when it comes to
actually doing an attack.

--------------------------------------------------------------------------------
"""

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from cleverspeech.utils.Utils import l_map, log


class AbstractProcedure(ABC):
    def __init__(self, attack, steps: int = 1000, update_step: int = 100):

        self.steps = steps + 1
        self.update_step = update_step

        assert attack.optimiser is not None

        self.attack = attack
        self.current_step = 0

        self.successes = tf.Variable(
            tf.ones(attack.batch.size, dtype=tf.int16),
            trainable=False,
            name="successes"
        )

    def init_optimiser_variables(self):
        """
        We must wait to initialise the optimiser so that we can initialise only
        the attack variables (i.e. not the deepspeech ones).
        """

        log("Initialising graph variables ...", wrap=False)

        self.attack.optimiser.create_optimiser()

        opt_vars = self.attack.delta_graph.opt_vars

        # optimiser.variables is always a {int: list} dictionary
        for val in self.attack.optimiser.variables.values():
            opt_vars += val

        self.attack.sess.run(tf.variables_initializer(opt_vars))
        log("Graph variables initialised.", wrap=True)

    def tf_run(self, tf_variables):
        """
        Helper method to automatically pass in the attack feed dictionary.
        """
        sess = self.attack.sess
        feed = self.attack.feeds.attack
        return sess.run(tf_variables, feed_dict=feed)

    def steps_rule(self):
        """
        Allows MixIns to take control of how long to optimise for.
        e.g. number of iterations, minimum bound reached, one success unbounded
        """
        return self.current_step < self.steps

    @abstractmethod
    def check_for_successful_examples(self):
        """
        Check if we've been successful and run the update steps if we have
        This should be defined in **EVERY** child implementation of this class.
        """
        pass

    def results_hook(self, successes):
        return self.current_step, True, successes

    def pre_optimisation_updates_hook(self, successes):

        if self.attack.size_constraint is not None and any(successes):
            self.attack.size_constraint.update(
                self.tf_run(self.attack.perturbations),
                successes
            )

        for loss in self.attack.loss:
            if loss.updateable is True:
                loss.update_many(successes)

    def post_optimisation_hook(self, successes):
        """
        Should we do any post optimisation + pre-decoding processing?

        N.B. This method is not abstract as it is *not* required to run an
        attack, but feel free to override it.
        """
        pass

    def run(self):
        """
        Do the actual optimisation.
        """

        a = self.attack

        # write out initial step 0 data
        successes = l_map(
            lambda x: x, self.check_for_successful_examples()
        )
        yield self.results_hook(successes)

        log("Initial state written, starting optimisation....")

        while self.steps_rule():

            is_update_step = self.current_step % self.update_step == 0
            is_update_step = is_update_step and not self.current_step == 0

            if is_update_step:

                successes = l_map(
                    lambda x: x, self.check_for_successful_examples()
                )

                bool_success = l_map(
                    lambda x: x[0], successes
                )

                # perform some rounding regardless of success or not.
                # currently uses PGD to round/floor perturbations to integers
                # prior to the results call (otherwise we might tag examples
                # as successful when rounding would make them unsuccessesful)
                self.post_optimisation_hook(bool_success)

                # signal that we've finished optimisation for now **BEFORE**
                # doing any updates (e.g. hard constraint bounds) to attack
                # variables.

                yield self.results_hook(successes)
                # perform updates that will affect the success of perturbations.
                # e.g. CGD clipping etc. These ops must be run *after* results
                # call else we'll never show anything as successful in the logs

                self.pre_optimisation_updates_hook(bool_success)

            # Do the actual optimisation
            a.optimiser.optimise(a.feeds.attack)
            self.current_step += 1


class SuccessOnDecoding(AbstractProcedure):
    """
    Check whether current adversarial decodings match the target phrase,
    yielding True if so, False if not.
    """
    def __init__(self, attack, **kwargs):

        super().__init__(attack, **kwargs)

        self.init_optimiser_variables()

    def check_for_successful_examples(self):

        decodings, _ = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            top_five=False,
        )

        phrases = self.attack.batch.targets["phrases"]

        return l_map(
            lambda x: (x[0] == x[1], x[0], x[1]), zip(decodings, phrases)
        )


class SuccessOnLosses(AbstractProcedure):
    """
    Perform updates when loss reaches a specified threshold.
    """
    def __init__(self, attack, loss_lower_bound=0.1, **kwargs):

        super().__init__(attack, **kwargs)

        self.loss_bound = loss_lower_bound

    def check_for_successful_examples(self):

        loss = self.tf_run(self.attack.loss_fn)
        threshold = [self.loss_bound for _ in range(self.attack.batch.size)]

        return l_map(
            lambda x: x[0] <= x[1], zip(loss, threshold)
        )


class HardcoreMode(SuccessOnDecoding):
    """
    Optimise forever (or until you KeyboardInterrupt).

    Useful for development: leave it running overnight to see how long an
    extreme optimisation case takes to finish.
    """
    def steps_rule(self):
        """
        Keep optimising regardless of any kind of success.
        """
        return True


class SuccessOnDecoderLogProbs(AbstractProcedure):
    """
    Perform updates when decoder log probs reach a specified threshold.
    """
    def __init__(self, attack, probs_diff=10.0, **kwargs):

        super().__init__(attack, **kwargs)

        self.probs_diff = probs_diff

    def check_for_successful_examples(self):

        _, probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            top_five=False,
        )

        loss = self.tf_run(self.attack.loss_fn)
        threshold = [self.probs_diff for _ in range(self.attack.batch.size)]

        return l_map(
            lambda x: x[0] <= x[1], zip(loss, threshold)
        )


class SuccessOnGreedySearchPath(AbstractProcedure):
    """
    Perform updates when loss reaches a specified threshold.
    """
    def __init__(self, attack, **kwargs):

        super().__init__(attack, **kwargs)

    def check_for_successful_examples(self):

        current_argmaxes = tf.argmax(
            self.attack.victim.logits, axis=-1
        )
        target_argmaxes = self.attack.placeholders.targets

        test = tf.reduce_all(
            tf.equal(current_argmaxes, target_argmaxes),
            axis=-1
        )

        return self.attack.sess.run(test)


class SuccessOnDeepSpeechBeamSearchPath(AbstractProcedure):
    """
    Perform updates when loss reaches a specified threshold.
    """
    def __init__(self, attack, loss_lower_bound=0.1, **kwargs):

        super().__init__(attack, **kwargs)

        self.loss_bound = loss_lower_bound

    def check_for_successful_examples(self):

        _, _, token_order, timestep_switches = self.attack.victim.ds_decode_batch_no_lm(
            self.attack.procedure.tf_run(
                self.attack.victim.logits
            ),
            self.attack.batch.audios["ds_feats"],
            top_five=False, with_metadata=True
        )
        ds_decode_align = 28 * np.ones(
            self.attack.victim.logits.shape, dtype=np.int32
        )

        for tok, time in zip(token_order, timestep_switches):
            ds_decode_align[0][time] = tok

        test = tf.reduce_all(
            tf.equal(
                ds_decode_align, tf.argmax(self.attack.victim.logits, axis=-1)
            ), axis=-1
        )

        return self.tf_run(test)
