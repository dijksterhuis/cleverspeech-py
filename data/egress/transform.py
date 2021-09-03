import numpy as np

from collections import OrderedDict

from cleverspeech.data.metrics.transcription_error import character_error_rate
from cleverspeech.data.metrics.transcription_error import word_error_rate
from cleverspeech.data.metrics.dsp_metrics import lnorm
from cleverspeech.data.metrics.dsp_metrics import peak_to_peak
from cleverspeech.utils.Utils import log


def fix_nesting(y):
    return [x[0] for x in y]


def fix_evasion_nestings(batched_results):

    batched_results["initial_taus"] = fix_nesting(
        batched_results["initial_taus"]
    )
    batched_results["bounds_raw"] = fix_nesting(
        batched_results["bounds_raw"]
    )

    return batched_results


def get_argmax_alignment(tokens, logits):
    argmax_alignment = [tokens[i] for i in np.argmax(logits, axis=1)]
    argmax_alignment = "".join(argmax_alignment)
    return argmax_alignment


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


def transpose(batched_results):
    assert type(batched_results) is dict

    d = {idx: {} for idx in range(len(batched_results["step"]))}

    for k, v in batched_results.items():
        for idx in d.keys():
            try:
                d[idx][k] = batched_results[k][idx]
            except IndexError:
                print(idx, k, batched_results[k])
                raise

            except KeyError:
                print(idx, k, batched_results[k])
                raise

    return d


def metadata_transforms(batched_results):

    examples = transpose(fix_evasion_nestings(batched_results))

    for idx, example in examples.items():

        example["argmax"] = get_argmax_alignment(
            example["tokens"], example["raw_logits"]
        )

        example["l0"] = lnorm(example["deltas"], norm=0)
        example["l1"] = lnorm(example["deltas"], norm=1)
        example["l2"] = lnorm(example["deltas"], norm=2)
        example["linf"] = lnorm(example["deltas"], norm=np.inf)
        example["p2p"] = peak_to_peak(example["deltas"])

        # convert spaces to "=" so we can tell what's what in logs
        example["decodings"] = example["decodings"].replace(" ", "=")
        example["phrases"] = example["phrases"].replace(" ", "=")

        yield OrderedDict(example)


def logging_transforms(example_data, additional_logging_keys=None):

    logging_keys = [
        "step",
        "basenames",
        "success",
        "total_loss",
        "l0",
        "l2",
        "linf",
        "p2p",
        "probs",
    ]

    if additional_logging_keys is not None:
        logging_keys.append(additional_logging_keys)

    logging_data = [(k, example_data[k]) for k in logging_keys]

    log_result = OrderedDict(logging_data)

    # always add custom checks after everything else
    # to make log output human readable.

    all_losses = [
        ("loss{}".format(i), l) for i, l in
        enumerate(example_data["losses"])
    ]

    cer = character_error_rate(
        example_data["decodings"], example_data["phrases"]
    )
    wer = word_error_rate(
        example_data["decodings"], example_data["phrases"]
    )
    log_updates = [
        *all_losses,
        # ("grads", sum(np.abs(example_data["gradients"]))),
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
    step_logs = step_logging(log_result)

    return step_logs


def transforms_gen(results, settings):
    for example_data in metadata_transforms(results):
        log(
            logging_transforms(example_data),
            wrap=False,
            outdir=settings["outdir"],
            stdout=False,
            timings=True,
        )
        yield example_data
