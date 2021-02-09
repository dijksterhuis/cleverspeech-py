import tensorflow as tf
import numpy as np

from abc import ABC
from collections import OrderedDict

from cleverspeech.data.Results import SingleJsonDB, step_logging, success_logging
from cleverspeech.utils.Utils import log, dump_wavs
from cleverspeech.eval.ErrorRates import character_error_rate
from cleverspeech.eval.ErrorRates import word_error_rate
from cleverspeech.eval.OnlineProcessing import get_perceptual_stats


class Base(ABC):

    def __init__(self, attack, batch, outdir):
        self.attack = attack
        self.batch = batch
        self.outdir = outdir

        self.example_db = None
        self.__initialise_db()

    def __initialise_db(self):
        self.example_db = SingleJsonDB(self.outdir)

    def put_db(self, example):
        self.example_db.open(
            example['basename'].rstrip(".wav")
        ).put(example)

    @staticmethod
    def get_argmax_alignment(tokens, logits):
        argmax_alignment = [tokens[i] for i in np.argmax(logits, axis=1)]
        argmax_alignment = "".join(argmax_alignment)
        return argmax_alignment

    def run(self, outs, queue):

        # -- get batch'd variables from the tensorflow graph

        a, b = self.attack, self.batch

        graph_losses = [l.loss_fn for l in a.loss]
        losses = a.procedure.tf_run(graph_losses)

        graph_variables = [
            a.loss_fn,
            a.hard_constraint.bounds,
            a.graph.final_deltas,
            a.graph.adversarial_examples,
            a.graph.opt_vars,
            a.victim.logits,
            tf.transpose(a.victim.raw_logits, [1, 0, 2])
        ]

        tf_outs = a.procedure.tf_run(graph_variables)

        [
            total_losses,
            bounds,
            deltas,
            adv_audio,
            delta_vars,
            softmax_logits,
            raw_logits,
        ] = tf_outs

        for out in outs["data"]:

            # -- get stuff from the outs dict passed in from the Procedure
            idx = out["idx"]
            decoding = out["decodings"]
            target_phrase = out["target_phrase"]
            probs = out["probs"][idx]
            success = out["success"]

            # -- get useful stuff for logging to stdout / file
            basename = b.audios.basenames[idx]
            delta = deltas[idx]

            initial_tau = a.hard_constraint.initial_taus[idx][0]
            distance_raw = a.hard_constraint.analyse(delta)
            bound_raw = bounds[idx][0]
            bound_epsilon = bound_raw / initial_tau
            distance_epsilon = distance_raw / initial_tau

            total_loss = total_losses[idx]
            all_losses = [
                ("loss{}".format(i), l[idx]) for i, l in enumerate(losses)
            ]

            target = target_phrase.replace(' ', '=')
            decoding = decoding.replace(' ', '=')
            cer = character_error_rate(decoding, target)
            wer = word_error_rate(decoding, target)

            # -- Use an OrderDict to preserve the order of items.
            # Otherwise we get weird print issues in containers.
            log_result = OrderedDict(
                [
                    ("step", outs["step"]),
                    ("file", basename),
                    ("success", int(success)),
                    ("tot_loss", total_loss),
                    *all_losses,
                    ("eps_b", bound_epsilon),
                    ("eps_d", distance_epsilon),
                    ("cer", cer),
                    ("wer", wer),
                    ("loglike", probs),
                    ("targ", target),
                    ("decode", decoding),
                ]
            )

            # -- Log how we're doing to file on disk
            # if you want to monitor progress live then you'll have to do:
            # `tail -f ./path/to/outdir/log.txt`.
            step_logs = step_logging(log_result)

            log(
                step_logs,
                wrap=False,
                outdir=self.outdir,
                stdout=False
            )

            if success is True:

                # save time by only doing this stuff when needed
                actual_length = b.audios.actual_lengths[idx]
                original = b.audios.audio[idx]
                advex = adv_audio[idx]
                d_var = delta_vars[0][idx]
                raw_logs = raw_logits[idx]
                smax_logs = softmax_logits[idx]
                argmax_alignment = self.get_argmax_alignment(
                    b.targets.tokens,
                    raw_logs,
                )

                db_output = OrderedDict(
                    [
                        ("step", outs["step"]),
                        ("actual_length", actual_length),
                        ("target", target),
                        ("success", success),
                        ("original", original),
                        ("delta", delta),
                        ("advex", advex),
                        ("top_five_decode", decoding),
                        ("top_five_loglikes", probs),
                        ("delta_variables", d_var),
                        ("argmax_decoding", argmax_alignment),
                        ("raw_logits", raw_logs),
                        ("smax_logits", smax_logs),
                        ("initial_tau", initial_tau),
                        ("distance_eps", distance_epsilon),
                        ("bound_eps", bound_epsilon),
                        ("distance_raw", distance_raw),
                        ("bound_raw", bound_raw),
                    ]
                )
                db_output.update(log_result)

                # -- Calculate some SNR / Lp norm values etc.
                # snr_stats = get_perceptual_stats(db_output)
                # db_output.update(snr_stats)

                # -- Log success to stdout as that's nice to know about.
                # Turning this off for now as it's redundant info that bloats
                # the jenkins logs.

                # log(success_logging(db_output), wrap=False)

                queue.put(db_output)

        return outs["step"]

