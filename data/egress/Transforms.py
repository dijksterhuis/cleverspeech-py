import numpy as np

from abc import ABC
from collections import OrderedDict

from cleverspeech.data.egress.eval.ErrorRates import character_error_rate
from cleverspeech.data.egress.eval.ErrorRates import word_error_rate


class Standard(ABC):

    def __init__(self, extra_logging_keys=None):
        self.extra_logging_keys = extra_logging_keys

    @staticmethod
    def get_argmax_alignment(tokens, logits):
        argmax_alignment = [tokens[i] for i in np.argmax(logits, axis=1)]
        argmax_alignment = "".join(argmax_alignment)
        return argmax_alignment

    def custom_success_modifications(self, db_output):
        """
        Extend this class with custom modifications, e.g. to modify experiment
        specific data per example.

        :param db_output: current output data to be written to disk
        :return: modified output data to be written to disk
        """
        return db_output

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
    def success_logging(example_result):

        s = """\rFound new example: {f}\tBound: {b:.3f}\tDistance: {d:.3f}\tLoss: {l:.3f}\tLoglike: {p:.3f}\tDecoding: {o}\tTarget: {t}\t""".format(
            f=example_result['basename'],
            b=example_result["bound"],
            d=example_result["distance"],
            l=example_result["total loss"],
            p=example_result["loglike"],
            o=example_result["decoding"],
            t=example_result["target"],
        )

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

    def run(self, batched_results):

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

        example_wise_results = self.convert_to_example_wise(batched_results)

        for example_idx, example_data in example_wise_results.items():

            logging_keys = [
                "step",
                "basenames",
                "success",
                "total_loss",
                "bounds_eps",
                "distances_eps",
                "probs",
            ]

            if self.extra_logging_keys is not None:
                logging_keys += self.extra_logging_keys

            logging_data = [(k, example_data[k]) for k in logging_keys]

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
                db_output["alignment_argmax"] = self.get_argmax_alignment(
                    example_data["tokens"], example_data["softmax_logits"]
                )

                db_output = self.custom_success_modifications(db_output)

                yield step_logs, db_output

            else:
                yield step_logs, None

