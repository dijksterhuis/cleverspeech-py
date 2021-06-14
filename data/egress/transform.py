import numpy as np

from collections import OrderedDict

from cleverspeech.data.egress.metrics.ErrorRates import character_error_rate
from cleverspeech.data.egress.metrics.ErrorRates import word_error_rate
from cleverspeech.data.egress.metrics.DetectionMetrics import lnorm
from cleverspeech.data.egress.metrics.DetectionMetrics import peak_to_peak
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
    batched_results["distances_raw"] = fix_nesting(
        batched_results["distances_raw"]
    )
    batched_results["bounds_eps"] = fix_nesting(
        batched_results["bounds_eps"]
    )
    batched_results["distances_eps"] = fix_nesting(
        batched_results["distances_eps"]
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


def evasion_transforms(batch_results):

    evasion_examples = transpose(fix_evasion_nestings(batch_results))

    for example_idx, example_data in evasion_examples.items():

        example_data["argmax"] = get_argmax_alignment(
            example_data["tokens"], example_data["raw_logits"]
        )
        # convert spaces to "=" so we can tell what's what in logs
        example_data["decodings"] = example_data["decodings"].replace(" ", "=")
        example_data["phrases"] = example_data["phrases"].replace(" ", "=")

        yield OrderedDict(example_data)


def unbounded_transforms(batched_results):

    unbounded_examples = transpose(batched_results)

    for example_idx, example_data in unbounded_examples.items():

        example_data["argmax"] = get_argmax_alignment(
            example_data["tokens"], example_data["raw_logits"]
        )

        example_data["p2p"] = peak_to_peak(example_data["deltas"])
        example_data["linfs"] = lnorm(example_data["deltas"], norm=np.inf)
        example_data["l2s"] = lnorm(example_data["deltas"], norm=2)

        # convert spaces to "=" so we can tell what's what in logs
        example_data["decodings"] = example_data["decodings"].replace(" ", "=")
        example_data["phrases"] = example_data["phrases"].replace(" ", "=")

        yield OrderedDict(example_data)


def evasion_logging(example_data, additional_logging_keys=None):

    logging_keys = [
        "step",
        "basenames",
        "success",
        "total_loss",
        "bounds_eps",
        "distances_eps",
        "probs",
    ]

    if additional_logging_keys is not None:
        logging_keys.append(additional_logging_keys)

    logging_data = [(k, example_data[k]) for k in logging_keys]
    log_result = OrderedDict(logging_data)

    all_losses = [
        ("loss{}".format(i), l) for i, l in enumerate(example_data["losses"])
    ]
    cer = character_error_rate(example_data["decodings"], example_data["phrases"])
    wer = word_error_rate(example_data["decodings"], example_data["phrases"])

    log_updates = [
        *all_losses,
        ("grads", sum(example_data["gradients"])),
        ("cer", cer),
        ("wer", wer),
        ("targ", example_data["phrases"]),
        ("decode", example_data["decodings"]),
    ]

    log_result.update(log_updates)

    # -- Log how we're doing to file on disk
    # if you want to monitor progress live then you'll have to do:
    # `tail -f ./path/to/outdir/log.txt`.
    step_logs = step_logging(log_result)

    return step_logs


def unbounded_logging(example_data, additional_logging_keys=None):

    logging_keys = [
        "step",
        "basenames",
        "success",
        "total_loss",
        "linfs",
        "l2s",
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
        ("grads", sum(np.abs(example_data["gradients"]))),
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


def evasion_gen(results, settings):
    for example_data in evasion_transforms(results):
        log(
            evasion_logging(example_data),
            wrap=False,
            outdir=settings["outdir"],
            stdout=False,
            timings=True,
        )
        yield example_data


def unbounded_gen(results, settings):
    for example_data in unbounded_transforms(results):
        log(
            unbounded_logging(example_data),
            wrap=False,
            outdir=settings["outdir"],
            stdout=False,
            timings=True,
        )
        yield example_data