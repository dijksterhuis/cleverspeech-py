import numpy as np
import tensorflow as tf

from collections import OrderedDict

from cleverspeech.data.egress.metrics.ErrorRates import character_error_rate
from cleverspeech.data.egress.metrics.ErrorRates import word_error_rate
from cleverspeech.data.egress.metrics.DetectionMetrics import lnorm

from cleverspeech.utils.Utils import l_map


def convert_evasion_attack_state_to_dict(attack):
    """
    Get the current values of a bunch of attack graph variables.

    """

    a, b = attack, attack.batch

    # can use either tf or deepspeech decodings ("ds" or "batch")
    # "batch" is prefered as it's what the actual model would use.
    # It does mean switching to CPU every time we want to do
    # inference but it's not a major hit to performance

    # keep the top 5 scoring decodings and their probabilities as that might
    # be useful come analysis time...

    top_5_decodings, top_5_probs = a.victim.inference(
        b,
        feed=a.feeds.attack,
        decoder="batch",
        top_five=True,
    )

    decodings, probs = a.victim.inference(
        b,
        feed=a.feeds.attack,
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
        a.perturbations,
        a.adversarial_examples,
        a.delta_graph.opt_vars,
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
    initial_tau = a.hard_constraint.initial_taus

    distance_raw = l_map(
        a.hard_constraint.analyse, deltas
    )
    bound_eps = l_map(
        lambda x: x[0] / x[1], zip(bounds_raw, initial_tau)
    )
    distance_eps = l_map(
        lambda x: x[0] / x[1], zip(distance_raw, initial_tau)
    )
    batched_tokens = l_map(
        lambda _: b.targets["tokens"], range(b.size)
    )

    batched_results = {
        "step": l_map(lambda _: a.procedure.current_step, range(b.size)),
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

    batched_results["success"] = l_map(
        lambda x: x, a.procedure.check_for_successful_examples()
    )

    return batched_results


def convert_unbounded_attack_state_to_dict(attack):
    """
    Get the current values of a bunch of attack graph variables -- for unbounded
    attacks only.
    """

    a, b = attack, attack.batch

    # can use either tf or deepspeech decodings ("ds" or "batch")
    # "batch" is prefered as it's what the actual model would use.
    # It does mean switching to CPU every time we want to do
    # inference but it's not a major hit to performance

    # keep the top 5 scoring decodings and their probabilities as that might
    # be useful come analysis time...

    top_5_decodings, top_5_probs = a.victim.inference(
        b,
        feed=a.feeds.attack,
        decoder="batch",
        top_five=True,
    )

    decodings, probs = a.victim.inference(
        b,
        feed=a.feeds.attack,
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
        a.perturbations,
        a.adversarial_examples,
        a.delta_graph.opt_vars,
        a.victim.logits,
        tf.transpose(a.victim.raw_logits, [1, 0, 2]),
    ]
    outs = a.procedure.tf_run(graph_variables)

    [
        total_losses,
        deltas,
        adv_audio,
        delta_vars,
        softmax_logits,
        raw_logits,
    ] = outs

    batched_results = {
        "step": l_map(lambda _: a.procedure.current_step, range(b.size)),
        "losses": losses_transposed,
        "total_loss": total_losses,
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

    batched_results["success"] = l_map(
        lambda x: x, a.procedure.check_for_successful_examples()
    )

    return batched_results


class EvasionResults:

    def __init__(self, extra_logging_keys=None):
        self.extra_logging_keys = extra_logging_keys

        self.logging_keys = [
            "step",
            "basenames",
            "success",
            "total_loss",
            "bounds_eps",
            "distances_eps",
            "probs",
        ]
        if self.extra_logging_keys is not None:
            self.logging_keys += self.extra_logging_keys

    @staticmethod
    def get_argmax_alignment(tokens, logits):
        argmax_alignment = [tokens[i] for i in np.argmax(logits, axis=1)]
        argmax_alignment = "".join(argmax_alignment)
        return argmax_alignment

    # noinspection PyMethodMayBeStatic
    def custom_success_modifications(self, db_output):
        """
        Extend this class with custom modifications, e.g. to modify experiment
        specific data per example.

        :param db_output: current output data to be written to disk
        :return: modified output data to be written to disk
        """
        return db_output

    # noinspection PyMethodMayBeStatic
    def custom_logging_modifications(self, log_output):
        """
        Extend this class with custom modifications, as above but for logging
        only.

        :param log_output: current log data
        :return: modified output log data
        """
        return log_output

    @staticmethod
    def step_logging(step_results, delimiter="|"):
        s = ""
        for k, v in step_results.items():
            if type(v) in (float, np.float32, np.float64):
                s += "{k}: {v:.4f}{d}".format(k=k, v=v, d=delimiter)
            elif type(v) in [int, np.int8, np.int16, np.int32, np.int64]:
                s += "{k}: {v:.0f}{d}".format(k=k, v=v, d=delimiter)
            else:
                try:
                    s += "{k}: {v}{d}".format(k=k, v=v, d=delimiter)
                except TypeError:
                    pass
        return s

    @staticmethod
    def convert_to_example_wise(batched_results):

        assert type(batched_results) is dict

        d = {idx: {} for idx in range(len(batched_results["step"]))}

        for k, v in batched_results.items():
            for idx in d.keys():
                try:
                    d[idx][k] = batched_results[k][idx]
                except IndexError:
                    print(idx, k, batched_results[k])
                    raise

        return d

    @staticmethod
    def fix_nestings(y):
        return [x[0] for x in y]

    def transpose(self, batched_results):

        batched_results["initial_taus"] = self.fix_nestings(
            batched_results["initial_taus"]
        )
        batched_results["bounds_raw"] = self.fix_nestings(
            batched_results["bounds_raw"]
        )
        batched_results["distances_raw"] = self.fix_nestings(
            batched_results["distances_raw"]
        )
        batched_results["bounds_eps"] = self.fix_nestings(
            batched_results["bounds_eps"]
        )
        batched_results["distances_eps"] = self.fix_nestings(
            batched_results["distances_eps"]
        )

        return self.convert_to_example_wise(batched_results)

    def gen(self, batched_results):

        example_wise_results = self.transpose(batched_results)

        for example_idx, example_data in example_wise_results.items():

            logging_data = [(k, example_data[k]) for k in self.logging_keys]

            log_result = OrderedDict(logging_data)
            log_result = self.custom_logging_modifications(log_result)

            # always add custom checks after everything else
            # to make log output human readable.

            all_losses = [
                ("loss{}".format(i), l) for i, l in enumerate(example_data["losses"])
            ]

            # convert spaces to "=" so we can tell what's what in logs
            example_data["decodings"] = example_data["decodings"].replace(" ", "=")
            example_data["phrases"] = example_data["phrases"].replace(" ", "=")

            cer = character_error_rate(
                example_data["decodings"], example_data["phrases"]
            )
            wer = word_error_rate(
                example_data["decodings"], example_data["phrases"]
            )
            log_updates = [
                *all_losses,
                ("cer", cer),
                ("wer", wer),
                ("targ", example_data["phrases"]),
                ("decode", example_data["decodings"]),
            ]

            log_result.update(
                log_updates
            )

            # -- Log how we're doing to file on disk
            # if you want to monitor progress live then you'll have to do:
            # `tail -f ./path/to/outdir/log.txt`.
            step_logs = self.step_logging(log_result)

            if example_data["success"] is True:

                db_output = OrderedDict(example_data)
                db_output = self.custom_success_modifications(db_output)

                yield step_logs, db_output

            else:
                yield step_logs, None


class UnboundedResults:

    def __init__(self, extra_logging_keys=None):
        self.extra_logging_keys = extra_logging_keys

        self.logging_keys = [
            "step",
            "basenames",
            "success",
            "total_loss",
            "probs",
        ]
        if self.extra_logging_keys is not None:
            self.logging_keys += self.extra_logging_keys

    @staticmethod
    def get_argmax_alignment(tokens, logits):
        argmax_alignment = [tokens[i] for i in np.argmax(logits, axis=1)]
        argmax_alignment = "".join(argmax_alignment)
        return argmax_alignment

    # noinspection PyMethodMayBeStatic
    def custom_success_modifications(self, db_output):
        """
        Extend this class with custom modifications, e.g. to modify experiment
        specific data per example.

        :param db_output: current output data to be written to disk
        :return: modified output data to be written to disk
        """
        return db_output

    # noinspection PyMethodMayBeStatic
    def custom_logging_modifications(self, log_output):
        """
        Extend this class with custom modifications, as above but for logging
        only.

        :param log_output: current log data
        :return: modified output log data
        """
        return log_output

    @staticmethod
    def step_logging(step_results, delimiter="|"):
        s = ""
        for k, v in step_results.items():
            if type(v) in (float, np.float32, np.float64):
                s += "{k}: {v:.4f}{d}".format(k=k, v=v, d=delimiter)
            elif type(v) in [int, np.int8, np.int16, np.int32, np.int64]:
                s += "{k}: {v:.0f}{d}".format(k=k, v=v, d=delimiter)
            else:
                try:
                    s += "{k}: {v}{d}".format(k=k, v=v, d=delimiter)
                except TypeError:
                    pass
        return s

    @staticmethod
    def convert_to_example_wise(batched_results):

        assert type(batched_results) is dict

        d = {idx: {} for idx in range(len(batched_results["step"]))}

        for k, v in batched_results.items():
            for idx in d.keys():
                try:
                    d[idx][k] = batched_results[k][idx]
                except IndexError:
                    print(idx, k, batched_results[k])
                    raise

        return d

    @staticmethod
    def fix_nestings(y):
        return [x[0] for x in y]

    def transpose(self, batched_results):
        return self.convert_to_example_wise(batched_results)

    def gen(self, batched_results):

        example_wise_results = self.transpose(batched_results)

        for example_idx, example_data in example_wise_results.items():

            logging_data = [(k, example_data[k]) for k in self.logging_keys]

            log_result = OrderedDict(logging_data)
            log_result = self.custom_logging_modifications(log_result)

            # always add custom checks after everything else
            # to make log output human readable.

            all_losses = [
                ("loss{}".format(i), l) for i, l in enumerate(example_data["losses"])
            ]

            example_data["linfs"] = [
                lnorm(x, norm=np.inf) for x in example_data["deltas"]
            ]
            example_data["l2s"] = [
                lnorm(x, norm=2) for x in example_data["deltas"]
            ]

            # convert spaces to "=" so we can tell what's what in logs
            example_data["decodings"] = example_data["decodings"].replace(" ", "=")
            example_data["phrases"] = example_data["phrases"].replace(" ", "=")

            cer = character_error_rate(
                example_data["decodings"], example_data["phrases"]
            )
            wer = word_error_rate(
                example_data["decodings"], example_data["phrases"]
            )
            log_updates = [
                *all_losses,
                ("cer", cer),
                ("wer", wer),
                ("targ", example_data["phrases"]),
                ("decode", example_data["decodings"]),
                ("l2", example_data["l2s"]),
                ("linf", example_data["linfs"]),
            ]

            log_result.update(
                log_updates
            )

            # -- Log how we're doing to file on disk
            # if you want to monitor progress live then you'll have to do:
            # `tail -f ./path/to/outdir/log.txt`.
            step_logs = self.step_logging(log_result)

            if example_data["success"] is True:

                db_output = OrderedDict(example_data)
                db_output = self.custom_success_modifications(db_output)

                yield step_logs, db_output

            else:
                yield step_logs, None

