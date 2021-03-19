import tensorflow as tf
from abc import ABC, abstractmethod


class AbstractProcedure(ABC):
    def __init__(self, attack, steps: int = 5000, decode_step: int = 10, loss_update_idx=None):
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

        if loss_update_idx is not None:
            assert type(loss_update_idx) in [tuple, list]
            for idx in loss_update_idx:
                assert type(idx) is int

        self.update_loss = loss_update_idx

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

    def steps_rule(self):
        """
        Allows MixIns to take control of how long to optimise for.
        e.g. number of iterations, minimum bound reached, one success unbounded
        """
        return self.current_step < self.steps

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

            # Normal optimisation procedure steps defined by the
            # `decode_step_logic` method. Can be overridden.

            is_decode_step = self.current_step % self.decode_step == 0
            is_zeroth_step = self.current_step == 0

            if is_decode_step or is_zeroth_step:
                # Generate output data and pass it to the results writer process
                batched_results = self.decode_step_logic()

                # update everything if we've been successful
                for idx, success_check in enumerate(batched_results["success"]):
                    if success_check:
                        self.do_success_updates(idx)

                queue.put(batched_results)
                yield batched_results

            # Do the actual optimisation
            attack.optimiser.optimise(attack.feeds.attack)
            self.current_step += 1

    @abstractmethod
    def decode_step_logic(self):
        """
        What should we do at decoding steps?

        The code below is run when child classes run the line:
        > super().decode_step_logic()

        This should be called in any child implementation of this method, unless
        you rewrite the entire method from scratch, which is also possible.
        """
        # ==> every x steps apply integer level rounding prior to decoding.
        # this helps the attacks work against the deepspeech native client api
        # which only accepts tf.int16 type inputs. Although it doesn't work 100%
        # of the time so use `bin/classify.py` to get the true success rate.

        # We could do a line search etc. after running optimisation, but doing
        # things during attack optimisation seems to help it find a solution in
        # integer space all by itself (vs. fiddling with the examples after).

        a, b = self.attack, self.attack.batch

        deltas = a.sess.run(a.graph.raw_deltas)
        a.sess.run(
            a.graph.raw_deltas.assign(tf.round(deltas))
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

        graph_losses = [l.loss_fn for l in a.loss]
        losses = [a.procedure.tf_run(graph_losses)]

        graph_variables = [
            a.loss_fn,
            a.hard_constraint.bounds,
            a.graph.final_deltas,
            a.graph.adversarial_examples,
            a.graph.opt_vars,
            a.victim.logits,
            tf.transpose(a.victim.raw_logits, [1, 0, 2]),
        ]
        outs = losses + a.procedure.tf_run(graph_variables)

        [
            losses,
            total_losses,
            bounds_raw,
            deltas,
            adv_audio,
            delta_vars,
            softmax_logits,
            raw_logits,
        ] = outs

        initial_tau = self.attack.hard_constraint.initial_taus
        distance_raw = self.attack.hard_constraint.analyse(deltas)
        bound_eps = [x / i_t for x, i_t in zip(bounds_raw, initial_tau)]
        distance_eps = [x / i_t for x, i_t in zip(distance_raw, initial_tau)]

        batched_tokens = [b.targets["tokens"] for _ in range(b.size)]

        batched_results = {
            "step": [self.current_step for _ in range(b.size)],
            "tokens": batched_tokens,
            "losses": losses,
            "total_loss": total_losses,
            "initial_taus": initial_tau,
            "bounds_raw": bounds_raw,
            "distances_raw": distance_raw,
            "bounds_eps": bound_eps[:][0],
            "distances_eps": distance_eps[:][0],
            "deltas": deltas,
            "advs": adv_audio,
            "delta_vars": delta_vars,
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

        return batched_results

    def check_for_success(self, lefts, rights):
        """
        Check if we've been successful and run the update steps if we have
        """
        z = zip(lefts, rights)
        for idx, (left, right) in enumerate(z):
            success = self.success_criteria_check(left, right)
            if success:
                yield success
            else:
                yield success

    @staticmethod
    @abstractmethod
    def success_criteria_check(left, right):
        """
        What is success?
        """
        pass

    @abstractmethod
    def do_success_updates(self, idx):
        """
        How should we update the attack for whatever we consider a success?
        """
        pass

    def do_warm_up(self):
        """
        Should anything else be done before we start? e.g. start with a
        randomised delta?

        N.B. This method is not abstract as it is *not* required to run an
        attack, but feel free to override it.
        """
        pass


class Unbounded(AbstractProcedure):
    """
    Don't update anything on a successful decoding, just exit optimisation.
    """
    def __init__(self, attack, *args, **kwargs):

        """
        Initialise the procedure object then initialise the optimiser
        variables => might be additional tf variables to initialise here.
        """
        self.finished = False
        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()

    def decode_step_logic(self):
        batched_results = super().decode_step_logic()

        decodings = batched_results["decodings"]
        targets = self.attack.batch.targets["phrases"]

        batched_results["success"] = [
            success for success in self.check_for_success(decodings, targets)
        ]

        current_n_successes = sum(
            [batched_results["success"] for _ in range(self.attack.batch.size)]
        )
        self.finished = self.attack.batch.size == current_n_successes

        return batched_results

    @staticmethod
    def success_criteria_check(left, right):
        return True if left == right else False

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
    Updates bounds MixIn.

    This class should never be initialised by itself, it should always be
    extended.
    """
    def __init__(self, attack, *args, **kwargs):

        """
        Initialise the procedure object then initialise the optimiser
        variables => might be additional tf variables to initialise here.
        """
        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()

    def do_success_updates(self, idx):

        # update the delta hard constraint
        delta = self.tf_run(self.attack.graph.final_deltas)[idx]
        self.attack.hard_constraint.update_one(delta, idx)

        # update any loss weightings
        if self.update_loss is not None:

            if type(self.update_loss) == int:
                self.attack.loss[self.update_loss].update(self.attack.sess, idx)

            elif type(self.update_loss) in [list, tuple]:
                for idx in self.update_loss:
                    self.attack.loss[idx].update(self.attack.sess, idx)


class UpdateOnDecoding(UpdateOnSuccess):
    """
    Updates bounds and loss weightings on a successful decoding.
    """
    def __init__(self, attack, *args, **kwargs):
        """
        Initialise the procedure object then initialise the optimiser
        variables => might be additional tf variables to initialise here.
        """
        super().__init__(attack, *args, **kwargs)
        self.init_optimiser_variables()

    def decode_step_logic(self):
        batched_results = super().decode_step_logic()

        decodings = batched_results["decodings"]
        targets = self.attack.batch.targets["phrases"]

        batched_results["success"] = [
            success for success in self.check_for_success(decodings, targets)
        ]

        return batched_results

    @staticmethod
    def success_criteria_check(left, right):
        """
        Decodings should match the target phrases exactly.
        :param left: decodings
        :param right: target phrases
        :return: Bool result
        """
        return True if left == right else False


class UpdateOnLoss(UpdateOnSuccess):
    """
    Updates bounds and loss weightings once loss reaches a specified threshold.
    """
    def __init__(self, attack, *args, loss_lower_bound=10.0, **kwargs):
        """
        Initialise the procedure object then initialise the optimiser
        variables => might be additional tf variables to initialise here.
        """
        super().__init__(attack, *args, **kwargs)
        self.loss_bound = loss_lower_bound
        self.init_optimiser_variables()

    def decode_step_logic(self):
        batched_results = super().decode_step_logic()

        loss = batched_results["total_loss"]
        target_loss = [self.loss_bound for _ in range(self.attack.batch.size)]

        batched_results["success"] = [
            success for success in self.check_for_success(loss, target_loss)
        ]

        return batched_results

    @staticmethod
    def success_criteria_check(left, right):
        """
        Loss should be less than/equal to a specified bound.
        :param left: loss
        :param right: target loss
        :return: Bool result
        """
        return True if left <= right else False


class UpdateOnDeepSpeechProbs(UpdateOnSuccess):
    """
    Updates bounds and loss weightings once log likelihood (decoder
    probabilities) has reached a certain point.
    """
    def __init__(self, attack, *args, probs_diff=10.0, **kwargs):
        """
        Initialise the procedure object then initialise the optimiser
        variables => might be additional tf variables to initialise here.
        """

        super().__init__(attack, *args, **kwargs)
        self.probs_diff = probs_diff
        self.init_optimiser_variables()

    def decode_step_logic(self):

        batched_results = super().decode_step_logic()

        probs = batched_results["probs"]
        target_probs = [self.probs_diff for _ in range(self.attack.batch.size)]

        batched_results["success"] = [
            success for success in self.check_for_success(probs, target_probs)
        ]

        return batched_results

    @staticmethod
    def success_criteria_check(left, right):
        """
        Decoder log probabilities should be less than/equal to a specified
        bound.

        Only use this implementation with the DeepSpeech beam search decoder (it
        outputs POSITIVE log probabilities).

        :param left: deepspeech decoder log probabilities (positive)
        :param right: target log probabilities (positive)
        :return: Bool result
        """
        return True if left <= right else False


class UpdateOnTensorflowProbs(UpdateOnDeepSpeechProbs):
    @staticmethod
    def success_criteria_check(left, right):
        """
        Decoder log probabilities should be less than/equal to a specified
        bound.

        Only use this implementation with a TF beam search decoder (it returns
        the raw negative log probabilities).

        :param left: deepspeech decoder log probabilities (positive)
        :param right: target log probabilities (positive)
        :return: Bool result
        """
        return True if left >= right else False


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
        """
        Initialise the evaluation procedure.

        :param attack_graph: The current attack graph perform optimisation with.
        """

        super().__init__(attack, *args, **kwargs)
        self.alignment_graph = alignment_graph
        self.init_optimiser_variables()

    def init_optimiser_variables(self):

        # We must wait until now to initialise the optimiser so that we can
        # initialise only the attack variables (i.e. not the deepspeech ones).

        self.alignment_graph.optimiser.create_optimiser()
        self.attack.optimiser.create_optimiser()

        opt_vars = self.attack.graph.opt_vars
        opt_vars += [self.alignment_graph.graph.initial_alignments]
        opt_vars += self.attack.optimiser.variables
        opt_vars += self.alignment_graph.optimiser.variables

        self.attack.sess.run(tf.variables_initializer(opt_vars))

    def run(self, queue, health_check):
        self.alignment_graph.optimise(self.attack.victim)
        for r in super().run(queue, health_check):
            yield r


class CTCAlignUpdateOnDecode(UpdateOnDecoding, CTCAlignMixIn):
    pass


class CTCAlignUnbounded(UpdateOnDecoding, CTCAlignMixIn):
    pass


class CTCAlignUpdateOnLoss(UpdateOnLoss, CTCAlignMixIn):
    pass


class CTCAlignHardcoreMode(HardcoreMode, CTCAlignMixIn):
    pass
