import tensorflow as tf
from abc import ABC, abstractmethod

from cleverspeech.utils.Utils import l_map


class AbstractProcedure(ABC):
    def __init__(self, attack, steps: int = 5000, decode_step: int = 10):
        """
        Base class that sets up a wealth of stuff to execute the attack over a
        number of iterations.

        This class should never be initialised by itself, it should always be
        extended. See examples below.

        """

        assert type(steps) in [float, int]
        assert type(decode_step) in [float, int]
        assert steps > decode_step
        assert attack.optimiser is not None

        self.attack = attack
        self.steps, self.decode_step = steps + 1, decode_step
        self.current_step = 0

    def init_optimiser_variables(self):
        """
        We must wait to initialise the optimiser so that we can initialise only
        the attack variables (i.e. not the deepspeech ones).

        This must be called in **EVERY** child classes' __init__() method so we
        can do the CTCAlign* procedures (special case).
        """

        self.attack.optimiser.create_optimiser()

        opt_vars = self.attack.graph.opt_vars

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
        Should anything else be done before we start? e.g. start with a
        randomised delta?

        N.B. This method is not abstract as it is *not* required to run an
        attack, but feel free to override it.
        """
        pass

    def decode_step_logic(self):
        """
        What should we do at decoding steps?
        """
        # ==> every x steps apply integer level rounding prior to decoding.
        # this helps the attacks work against the deepspeech native client api
        # which only accepts tf.int16 type inputs. Although it doesn't work 100%
        # of the time so use `bin/classify.py` to get the true success rate.

        # We could do a line search etc. after running optimisation, but doing
        # things during attack optimisation seems to help it find a solution in
        # integer space all by itself (vs. fiddling with the examples after).

        a, b = self.attack, self.attack.batch

        def reassign_tf_delta_vars(idx, delta):
            return a.graph.raw_deltas[idx].assign(tf.round(delta))

        deltas = a.sess.run(a.graph.raw_deltas)
        assigns = l_map(
            lambda x: reassign_tf_delta_vars(x[0], x[1]), enumerate(deltas)
        )
        a.sess.run(assigns)

    @abstractmethod
    def check_for_success(self, batched_results):
        """
        Check if we've been successful and run the update steps if we have
        This should be defined in **EVERY** child implementation of this class.
        """
        pass

    def get_current_attack_state(self):
        """
        Get the current values of a bunch of attack graph variables.
        """

        a, b = self.attack, self.attack.batch

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

        graph_losses = [l.loss_fn for l in a.loss]
        losses = a.procedure.tf_run(graph_losses)

        losses_transposed = [
            [
                losses[loss_idx][batch_idx] for loss_idx in range(len(a.loss))
            ] for batch_idx in range(b.size)
        ]

        graph_variables = [
            a.loss_fn,
            a.hard_constraint.bounds,
            a.graph.final_deltas,
            a.graph.adversarial_examples,
            a.graph.opt_vars,
            a.victim.logits,
            tf.transpose(a.victim.raw_logits, [1, 0, 2]),
        ]
        outs = a.procedure.tf_run(graph_variables)

        [
            total_losses,
            bounds_raw,
            deltas,
            adv_audio,
            delta_vars,
            softmax_logits,
            raw_logits,
        ] = outs

        # TODO: Fix nesting here or over in file write subprocess (as is now)?
        initial_tau = self.attack.hard_constraint.initial_taus
        distance_raw = [self.attack.hard_constraint.analyse(d) for d in deltas]
        bound_eps = [x / i_t for x, i_t in zip(bounds_raw, initial_tau)]
        distance_eps = [x / i_t for x, i_t in zip(distance_raw, initial_tau)]

        batched_tokens = [b.targets["tokens"] for _ in range(b.size)]

        batched_results = {
            "step": [self.current_step for _ in range(b.size)],
            "tokens": batched_tokens,
            "losses": losses_transposed,
            "total_loss": total_losses,
            "initial_taus": initial_tau,
            "bounds_raw": bounds_raw,
            "distances_raw": distance_raw,
            "bounds_eps": bound_eps,
            "distances_eps": distance_eps,
            "deltas": deltas,
            "advs": adv_audio,
            "delta_vars": [d for d in delta_vars[0]],
            "softmax_logits": softmax_logits,
            "raw_logits": raw_logits,
            "decodings": decodings,
            "top_five_decodings": top_5_decodings,
            "probs": probs,
            "top_five_probs": top_5_probs,
        }

        targs_batch_exclude = ["tokens"]
        targs = {k: v for k, v in b.targets.items() if k not in targs_batch_exclude}

        audio_batch_exclude = ["max_samples", "max_feats"]
        auds = {k: v for k, v in b.audios.items() if k not in audio_batch_exclude}

        batched_results.update(auds)
        batched_results.update(targs)

        successes = [
            success for success in self.check_for_success(batched_results)
        ]

        batched_results["success"] = successes

        return batched_results

    @abstractmethod
    def do_success_updates(self, idx):
        """
        How should we update the attack for whatever we consider a success?
        This should be defined in **EVERY** child implementation of this class.
        """
        pass

    def run(self, queue, health_check):
        """
        Do the actual optimisation.
        """
        attack, g, b = self.attack, self.attack.graph, self.attack.batch

        while self.steps_rule():

            # Do startup stuff.

            if self.current_step == 0:
                self.do_warm_up()

            # We've performed one step of optimisation. let the parent spawner
            # process know if everything is working. Any exception will have
            # been caught by the attack spawner boilerplate.

            if self.current_step == 1:
                health_check.send(True)

            is_decode_step = self.current_step % self.decode_step == 0
            is_zeroth_step = self.current_step == 0
            is_round_step = is_decode_step and not is_zeroth_step

            if is_round_step:
                self.decode_step_logic()

            if is_decode_step or is_zeroth_step:

                # Generate output data and pass it to the results writer process
                batched_results = self.get_current_attack_state()
                queue.put(batched_results)

                # update graph variables when successful i.e. hard constraint
                self.do_success_updates(batched_results)

            # Do the actual optimisation
            attack.optimiser.optimise(attack.feeds.attack)
            self.current_step += 1


class Unbounded(AbstractProcedure):
    def __init__(self, attack, *args, **kwargs):

        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()

        self.finished = False

    def check_for_success(self, batched_results):
        """
        Stop optimising when the decoding of every example in a batch matches
        the target phrase we want.
        """

        current_n_successes = sum(
            [batched_results["success"] for _ in range(self.attack.batch.size)]
        )
        self.finished = self.attack.batch.size == current_n_successes

        lefts = batched_results["decodings"]
        rights = batched_results["phrases"]

        z = zip(lefts, rights)

        for idx, (left, right) in enumerate(z):

            if left == right:
                yield True

            else:
                yield False

    def steps_rule(self):
        """
        Stop optimising once everything in a batch is successful.
        """
        return self.finished is not True

    def do_success_updates(self, idx):
        """
        if successful do nothing
        """
        pass


class UpdateOnSuccess(AbstractProcedure):
    """
    MixIn to update bounds and loss weightings.

    This class should never be initialised by itself, it should always be
    extended.
    """
    def __init__(self, attack, *args, loss_update_idx = None, **kwargs):

        super().__init__(attack, *args, **kwargs)

        self.init_optimiser_variables()

        if loss_update_idx is not None:
            assert type(loss_update_idx) in [tuple, list]
            for idx in loss_update_idx:
                assert type(idx) is int

        self.update_loss = loss_update_idx

    def update_hard_constraint(self, batched_results):
        self.attack.hard_constraint.update_many(
            batched_results["deltas"], batched_results["success"]
        )

    def update_losses(self, batched_results):
        for loss_idx in self.update_loss:
            loss_to_update = self.attack.loss[loss_idx]
            loss_to_update.update_many(batched_results["success"])

    def do_success_updates(self, batched_results):
        """
        Update both hard constraint bound and any loss weightings.
        """
        self.update_hard_constraint(batched_results)

        if self.update_loss is not None:
            self.update_losses(batched_results)


class UpdateOnDecoding(UpdateOnSuccess):
    def __init__(self, attack, *args, **kwargs):

        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()

    def check_for_success(self, batched_results):
        """
        Success is when the decoding matches the target phrase.
        """
        lefts = batched_results["decodings"]
        rights = batched_results["phrases"]

        z = zip(lefts, rights)

        for idx, (left, right) in enumerate(z):

            if left == right:
                yield True

            else:
                yield False


class UpdateOnLoss(UpdateOnSuccess):
    def __init__(self, attack, *args, loss_lower_bound=10.0, **kwargs):

        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()

        self.loss_bound = loss_lower_bound

    def check_for_success(self, batched_results):
        """
        Success is when the loss reaches a specified threshold.
        """

        lefts = batched_results["total_loss"]
        rights = [self.loss_bound for _ in range(self.attack.batch.size)]

        z = zip(lefts, rights)

        for idx, (left, right) in enumerate(z):

            if left <= right:
                yield True

            else:
                yield False


class UpdateOnDeepSpeechProbs(UpdateOnSuccess):
    def __init__(self, attack, *args, probs_diff=10.0, **kwargs):

        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()

        self.probs_diff = probs_diff

    def check_for_success(self, batched_results):
        """
        Success is when log likelihood (decoder probabilities) have reached a
        certain threshold.
        """

        lefts = batched_results["probs"]
        rights = [self.probs_diff for _ in range(self.attack.batch.size)]

        z = zip(lefts, rights)

        for idx, (left, right) in enumerate(z):

            if left <= right:
                yield True

            else:
                yield False


class HardcoreMode(UpdateOnLoss):
    """
    Updates bounds and loss weightings once loss has reached some extreme value.
    Optimises one example forever (or until you KeyboardInterrupt).

    Useful for development: leave it running overnight to see how long an
    extreme optimisation case takes to finish.
    """
    def __init__(self, attack, *args, loss_lower_bound=1.0, **kwargs):
        super().__init__(
            attack,
            *args,
            loss_lower_bound=loss_lower_bound,
            **kwargs
        )

    def steps_rule(self):
        """
        Keep optimising regardless of any kind of success.
        """
        return True


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

        opt_vars += self.attack.graph.opt_vars

        self.attack.sess.run(tf.variables_initializer(opt_vars))

    def run(self, queue, health_check):
        self.alignment_graph.optimise(self.attack.victim)
        super().run(queue, health_check)


class CTCAlignUpdateOnDecode(UpdateOnDecoding, CTCAlignMixIn):
    pass


class CTCAlignUnbounded(UpdateOnDecoding, CTCAlignMixIn):
    pass


class CTCAlignUpdateOnLoss(UpdateOnLoss, CTCAlignMixIn):
    pass


class CTCAlignHardcoreMode(HardcoreMode, CTCAlignMixIn):
    pass
