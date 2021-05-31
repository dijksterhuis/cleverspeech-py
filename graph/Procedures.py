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

from cleverspeech.utils.Utils import l_map


class AbstractProcedure(ABC):
    """
    Base class that sets up a wealth of stuff to execute the attack over a
    number of iterations.
    This class should never be initialised by itself, it should always be
    extended. See examples below.

    :param: attack: an attack graph
    :param: steps: number of iterations to run the attack for
    :param: decode_step: when to stop and check a current decoding

    """
    def __init__(self, attack, steps: int = 5000, update_step: int = 10):

        assert type(steps) in [float, int]
        assert type(update_step) in [float, int]
        assert steps > update_step
        assert attack.optimiser is not None

        self.attack = attack
        self.steps, self.update_step, self.results_step = steps + 1, update_step, update_step
        self.current_step = 0

    def init_optimiser_variables(self):
        """
        We must wait to initialise the optimiser so that we can initialise only
        the attack variables (i.e. not the deepspeech ones).

        This must be called in **EVERY** child classes' __init__() method so we
        can do the CTCAlign* procedures (special case).
        """

        self.attack.optimiser.create_optimiser()

        opt_vars = self.attack.delta_graph.opt_vars

        # optimiser.variables is always a {int: list} dictionary
        for val in self.attack.optimiser.variables.values():
            opt_vars += val

        self.attack.sess.run(tf.variables_initializer(opt_vars))

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

    def do_warm_up(self):
        """
        Should anything else be done before we start?

        N.B. This method is not abstract as it is *not* required to run an
        attack, but feel free to override it.

        For example, you could start with randomised perturbations.
        """
        def random_uniform_func(delta):

            rand_uni = np.random.uniform(
                0.002 * self.attack.bit_depth * -1,
                0.002 * self.attack.bit_depth,
                delta.shape
            )
            return delta + rand_uni

        self.attack.delta_graph.deltas_apply(
            self.attack.sess, random_uniform_func
        )

    def post_optimisation_hook(self):
        """
        Should we do any post optimisation + pre-decoding processing?

        N.B. This method is not abstract as it is *not* required to run an
        attack, but feel free to override it.
        """

        def rounding_func(delta):
            signs = np.sign(delta)
            abs_floor = np.floor(np.abs(delta))
            return signs * abs_floor

        self.attack.delta_graph.deltas_apply(
            self.attack.sess, rounding_func
        )

    @abstractmethod
    def check_for_successful_examples(self):
        """
        Check if we've been successful and run the update steps if we have
        This should be defined in **EVERY** child implementation of this class.
        """
        pass

    def results_hook(self):
        return True

    @abstractmethod
    def post_results_hook(self):
        """
        How should we update the attack for whatever we consider a success?
        This should be defined in **EVERY** child implementation of this class.
        """
        pass

    def run(self):
        """
        Do the actual optimisation.
        """
        a = self.attack

        while self.steps_rule():

            # Do startup stuff.

            if self.current_step == 0:
                self.do_warm_up()

            is_update_step = self.current_step % self.update_step == 0
            is_results_step = self.current_step % self.results_step == 0
            is_zeroth_step = self.current_step == 0
            is_round_step = is_update_step and not is_zeroth_step

            if is_round_step:
                self.post_optimisation_hook()

            if is_results_step:

                # signal that we've finished optimisation for now **BEFORE**
                # doing any updates (e.g. hard constraint bounds) to attack
                # variables.

                yield self.results_hook()

            if is_update_step or is_zeroth_step:

                self.post_results_hook()

            # Do the actual optimisation
            a.optimiser.optimise(a.feeds.attack)
            self.current_step += 1


class Unbounded(AbstractProcedure):
    """
    Never update the constraint or loss weightings.

    Useful to validate that an attack works correctly as all unbounded
    attacks should eventually find successful adversarial examples.
    """
    def __init__(self, attack, *args, **kwargs):

        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()

    def check_for_successful_examples(self):
        """
        Check whether current adversarial decodings match the target phrase,
        yielding True if so, False if not.

        Additionally, if all the decodings match all the target phrases then set
        `self.finished = True` to stop the attack.

        :return: bool
        """

        decodings, _ = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            top_five=False,
        )

        phrases = self.attack.batch.targets["phrases"]

        z1, z2 = zip(decodings, phrases), zip(decodings, phrases)

        for idx, (left, right) in enumerate(z1):
            if left == right:
                yield True

            else:
                yield False

    def post_results_hook(self):
        """
        do nothing
        """
        pass


class UnboundedWithEarlyStopping(Unbounded):
    """
    Never update the constraint or loss weightings.

    Useful to validate that an attack works correctly as all unbounded
    attacks should eventually find successful adversarial examples.
    """

    def __init__(self, attack, *args, **kwargs):
        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()

        self.early_stop_bools = l_map(lambda _: False, range(attack.batch.size))

    def check_for_successful_examples(self):
        """
        Check whether current adversarial decodings match the target phrase,
        yielding True if so, False if not.

        Additionally, if all the decodings match all the target phrases then set
        `self.finished = True` to stop the attack.

        :return: bool
        """
        for idx, res in enumerate(super().check_for_successful_examples()):
            if self.early_stop_bools[idx] is True:
                pass
            else:
                if res is True:
                    self.early_stop_bools[idx] = True

    def steps_rule(self):
        """
        Stop optimising once everything in a batch is successful, or we've hit a
        maximum number of iteration steps.
        """
        return not all(self.early_stop_bools) and self.current_step < self.steps


class HardcoreMode(Unbounded):
    """
    Optimise forever (or until you KeyboardInterrupt).

    Useful for development: leave it running overnight to see how long an
    extreme optimisation case takes to finish.
    """
    def __init__(self, attack, *args, **kwargs):
        super().__init__(
            attack,
            *args,
            **kwargs
        )

    def check_for_successful_examples(self):
        """
        Never stop optimising, but report if decoding is successful.
        """

        decodings, _ = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            top_five=False,
        )

        phrases = self.attack.batch.targets["phrases"]

        z = zip(decodings, phrases)

        for idx, (left, right) in enumerate(z):

            if left == right:
                yield True

            else:
                yield False

    def steps_rule(self):
        """
        Keep optimising regardless of any kind of success.
        """
        return True


class UpdateOnSuccess(AbstractProcedure):
    """
    MixIn to update bounds and loss weightings.

    This class should never be initialised by itself, it should always be
    extended.
    """
    def __init__(self, attack, *args, loss_update_idx = None, **kwargs):

        super().__init__(attack, *args, **kwargs)

        if loss_update_idx is not None:
            assert type(loss_update_idx) in [tuple, list]
            for idx in loss_update_idx:
                assert type(idx) is int

        self.update_loss = loss_update_idx

    def __update_hard_constraint(self, deltas , successes):

        self.attack.hard_constraint.update_many(
            deltas, successes
        )

    def __update_losses(self, successes):
        for loss_idx in self.update_loss:
            loss_to_update = self.attack.loss[loss_idx]
            loss_to_update.update_many(successes)

    def post_results_hook(self):
        """
        Update both hard constraint bound and any loss weightings.
        """

        deltas = self.tf_run(self.attack.perturbations)
        successes = l_map(
            lambda x: x, self.check_for_successful_examples()
        )

        self.__update_hard_constraint(deltas, successes)

        if self.update_loss is not None:
            self.__update_losses(successes)


class UpdateOnDecoding(UpdateOnSuccess):
    """
    Perform updates when decoding matches a target transcription.
    """
    def __init__(self, attack, *args, **kwargs):

        super().__init__(attack, *args, **kwargs)

    def check_for_successful_examples(self):
        """
        Success is when the decoding matches the target phrase.
        """
        decodings, _ = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            top_five=False,
        )

        phrases = self.attack.batch.targets["phrases"]

        z = zip(decodings, phrases)

        for idx, (left, right) in enumerate(z):

            if left == right:
                yield True

            else:
                yield False


class UpdateOnLoss(UpdateOnSuccess):
    """
    Perform updates when loss reaches a specified threshold.
    """

    def __init__(self, attack, *args, loss_lower_bound=0.1, **kwargs):

        super().__init__(attack, *args, **kwargs)

        self.loss_bound = loss_lower_bound

    def check_for_successful_examples(self):
        """
        Success is when the loss reaches a specified threshold.
        """
        loss = self.tf_run(self.attack.loss_fn)
        threshold = [self.loss_bound for _ in range(self.attack.batch.size)]

        z = zip(loss, threshold)

        for idx, (left, right) in enumerate(z):

            if left <= right:
                yield True

            else:
                yield False


class UpdateOnDeepSpeechProbs(UpdateOnSuccess):
    """
    Perform updates when deepspeech decoder log probs reach a specified
    threshold.
    """
    def __init__(self, attack, *args, probs_diff=10.0, **kwargs):

        super().__init__(attack, *args, **kwargs)

        self.probs_diff = probs_diff

    def check_for_successful_examples(self):
        """
        Success is when log likelihood (decoder probabilities) have reached a
        certain threshold.
        """

        _, probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            top_five=False,
        )

        threshold = [self.probs_diff for _ in range(self.attack.batch.size)]

        z = zip(probs, threshold)

        for idx, (left, right) in enumerate(z):

            if left <= right:
                yield True

            else:
                yield False


class SimpleEvasion(UpdateOnDecoding):
    def __init__(self, attack, *args, **kwargs):
        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()


class StandardProcedure(SimpleEvasion):
    pass


class HighConfidenceEvasion(UpdateOnLoss):
    def __init__(self, attack, *args, **kwargs):
        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()


class CTCAlignMixIn(AbstractProcedure, ABC):
    """
    Abstract MixIn class to be used to initialise the CTC alignment search graph
    to find an optimal alignment for the model and target transcription,
    irrelevant of the given example.
    """
    def __init__(self, attack, alignment_graph, *args, **kwargs):

        super().__init__(attack, *args, **kwargs)

        self.alignment_graph = alignment_graph
        self.init_optimiser_variables()

    def init_optimiser_variables(self):

        # We must wait until now to initialise the optimiser so that we can
        # initialise only the attack variables (i.e. not the deepspeech ones).

        self.alignment_graph.optimiser.create_optimiser()
        self.attack.optimiser.create_optimiser()

        opt_vars = [self.alignment_graph.graph.initial_alignments]
        opt_vars += self.alignment_graph.optimiser.variables

        for val in self.attack.optimiser.variables.values():
            opt_vars += val

        opt_vars += self.attack.delta_graph.opt_vars

        self.attack.sess.run(tf.variables_initializer(opt_vars))

    def run(self):
        self.alignment_graph.optimise(self.attack.victim)
        for x in super().run():
            yield x


class CTCAlignUpdateOnDecode(UpdateOnDecoding, CTCAlignMixIn):
    """
    For CTC Alignment Search attacks.

    Update when current decoding matches the target transcription.
    """
    pass


class SimpleCTCAlignEvasion(CTCAlignUpdateOnDecode):
    pass


class StandardCTCAlignProcedure(SimpleCTCAlignEvasion):
    pass


class CTCAlignUnbounded(Unbounded, CTCAlignMixIn):
    """
    For CTC Alignment Search attacks.

    Never update the constraint or loss weightings.

    Useful to validate that an attack works correctly as all unbounded
    attacks should eventually find successful adversarial examples.
    """
    pass


class CTCAlignUpdateOnLoss(UpdateOnLoss, CTCAlignMixIn):
    """
    For CTC Alignment Search attacks.

    Perform updates when loss reaches a specified threshold.
    """
    pass


class HighConfidenceCTCAlignEvasion(CTCAlignUpdateOnLoss):
    pass


class CTCAlignHardcoreMode(HardcoreMode, CTCAlignMixIn):
    """
    For CTC Alignment Search attacks.

    Optimise forever (or until you KeyboardInterrupt).

    Useful for development: leave it running overnight to see how long an
    extreme optimisation case takes to finish.
    """
    pass
