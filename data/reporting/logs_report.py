#!/usr/bin/env python3
import os
import sys
import itertools
import numpy as np

from matplotlib import pyplot as plt


plt.style.use(["science", "muted"])


def custom_bool(x):
    return x == "True"


LOG_PIPE_INDICES = {
    "step": (0, int),
    "basenames": (1, str),
    "success": (2, custom_bool),
    "total_loss": (3, float),
    "l0": (4, float),
    "l2": (5, float),
    "linf": (6, float),
    "p2p": (7, float),
    "probs": (8, float),
    "cer": (-5, float),
    "wer": (-4, float),
    "targ": (-3, str),
    "decode": (-2, str),
}

DELIMITER = "|"


def read_plain_text_log_file(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    return data


def remove_timestamps(row):
    return row.split("\t")[1]


def parse_delimiters(row):
    return row.split(DELIMITER)


def get_column(row, log_column):

    log_idx, conversion_fn = LOG_PIPE_INDICES[log_column]

    str_data = row[log_idx].split(": ")[1]

    return conversion_fn(str_data)


def extract_data_generator(log_data):

    for row in log_data:

        row = parse_delimiters(remove_timestamps(row))

        parsed_row_data = {
            k: get_column(row, k) for k in list(LOG_PIPE_INDICES.keys())
        }

        yield parsed_row_data


def find_first_and_last_success_and_zeroth_step(data):

    first_successful_steps = dict()
    last_successful_steps = dict()
    zeroth_step = dict()

    for basename, step_dict in data.items():
        for step, data in step_dict.items():

            if step == 0:
                zeroth_step[basename] = step_dict

            if data["success"] is True:
                if basename not in first_successful_steps.keys():
                    first_successful_steps[basename] = step_dict
                else:
                    last_successful_steps[basename] = step_dict
            else:
                pass

    return first_successful_steps, last_successful_steps, zeroth_step


def aggregate_per_step(data, idx, ref_fn=np.mean):

    for step, row in data.items():
        # print(row)
        losses = [r[idx] for r in row]
        yield step, ref_fn(losses)


def plot_per_measure(data, title, y_max=None, x_label=None, y_label=None, stddevs=None):

    x = np.asarray(list(data.keys()))
    y = np.asarray(list(data.values()))

    plt.plot(x, y, ls="-", marker="")
    plt.title(title)

    if stddevs is not None:
        s = np.asarray(list(stddevs.values()))

        plt.fill_between(
            x,
            (y + s),
            (y - s),
            alpha=.45
        )

    y_min = np.floor(min(y)) if min(y) < 0 else 0

    if y_max is not None:
        plt.ylim(y_min, y_max)
    else:
        plt.ylim(y_min)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    plt.grid()


def main(log_file_path, outpath):
    #
    # plt.ion()
    # plt.show()
    #
    # while True:

    raw_log_data = read_plain_text_log_file(log_file_path)

    extracted = [
        row for row in extract_data_generator(raw_log_data)
    ]

    basename_ordered = dict()
    step_ordered = dict()

    for idx, row in enumerate(extracted):

        bname, step = row["basenames"], row["step"]
        basename_ordered[bname] = {step: row}
        row["l2_n"] = row["l2"] / row["l0"] if row["l0"] > 0 else row["l2"]

        if step not in step_ordered.keys():
            step_ordered[step] = [row]
        else:
            step_ordered[step].append(row)

    first, last, zeroth = find_first_and_last_success_and_zeroth_step(
        basename_ordered
    )

    # =============> success rates

    success_rates = np.zeros(len(list(step_ordered.keys())), dtype=np.float32)
    mean_wers = np.zeros(len(list(step_ordered.keys())), dtype=np.float32)
    mean_cers = np.zeros(len(list(step_ordered.keys())), dtype=np.float32)

    cumulative_success_rates = np.zeros(
        len(list(step_ordered.keys())), dtype=np.float32
    )
    prev = [0 for _ in range(len(list(basename_ordered.keys())))]

    for idx, res_data in enumerate(step_ordered.values()):

        successes = [1 if r["success"] else 0 for r in res_data]
        success_rate = sum(successes) / len(list(basename_ordered.keys()))
        success_rates[idx] = success_rate

        prev = [1 if x == 1 or y == 1 else 0 for x, y in zip(prev, successes)]
        cumulative_success_rate = sum(prev) / len(list(basename_ordered.keys()))
        cumulative_success_rates[idx] = cumulative_success_rate

        mean_wer = np.mean([r["wer"] for r in res_data])
        mean_cer = np.mean([r["cer"] for r in res_data])

        mean_wers[idx] = mean_wer
        mean_cers[idx] = mean_cer

    plt.figure(1, tight_layout=True, figsize=(12, 9))
    plt.subplot(2, 2, 1)

    xes = np.asarray(list(step_ordered.keys()))

    plt.plot(xes, cumulative_success_rates, ls="-", marker="")
    plt.ylim([0, 1])
    plt.title("Cumulative Success Rate Per Step")

    plt.subplot(2, 2, 2)

    xes = np.asarray(list(step_ordered.keys()))

    plt.plot(xes, success_rates, ls="-", marker="")
    plt.ylim([0, 1])
    plt.title("Success Rate Per Step")

    plt.subplot(2, 2, 3)

    xes = np.asarray(list(step_ordered.keys()))
    plt.plot(xes, mean_wers, ls="-", marker="")
    plt.ylim([0, np.ceil(max(mean_wers))])
    plt.title("Mean Word-Error-Rate Rate Per Step")

    plt.subplot(2, 2, 4)

    xes = np.asarray(list(step_ordered.keys()))
    plt.plot(xes, mean_cers, ls="-", marker="")
    plt.ylim([0, np.ceil(max(mean_cers)/10)*10])
    plt.title("Mean Character-Error-Rate Rate Per Step")

    # =============> loss vs. ell norms

    def mean_normed(x):
        return np.mean(x) / len(x)


    loss_aggregates = {
        "mean": np.mean, "max": np.max, "min": np.min, "std": np.std
    }

    aggd_loss_per_step = {
        ref: {
            k: v for k, v in
        aggregate_per_step(step_ordered, "total_loss", ref_fn=ref_fn)
        } for ref, ref_fn in loss_aggregates.items()
    }

    lnorm_aggreagates = {
        "mean": np.mean, "max": np.max, "min": np.min, "std": np.std
    }

    aggd_l0_per_step = {
        ref: {
            k: v for k, v in
        aggregate_per_step(step_ordered, "l0", ref_fn=ref_fn)
        } for ref, ref_fn in lnorm_aggreagates.items()
    }
    aggd_l2_normed_per_step = {
        ref: {
            k: v for k, v in
            aggregate_per_step(step_ordered, "l2_n", ref_fn=ref_fn)
        } for ref, ref_fn in lnorm_aggreagates.items()
    }
    aggd_l2_per_step = {
        ref: {
            k: v for k, v in
            aggregate_per_step(step_ordered, "l2", ref_fn=ref_fn)
        } for ref, ref_fn in lnorm_aggreagates.items()
    }
    aggd_linf_per_step = {
        ref: {
            k: v for k, v in
            aggregate_per_step(step_ordered, "linf", ref_fn=ref_fn)
        } for ref, ref_fn in lnorm_aggreagates.items()
    }
    aggd_p2p_per_step = {
        ref: {
            k: v for k, v in
            aggregate_per_step(step_ordered, "p2p", ref_fn=ref_fn)
        } for ref, ref_fn in lnorm_aggreagates.items()
    }


    results = {
        "loss": aggd_loss_per_step,
        "l0": aggd_l0_per_step,
        # "l2_n": aggd_l2_normed_per_step,
        "l2": aggd_l2_per_step,
        "linf": aggd_linf_per_step,
        "p2p": aggd_p2p_per_step,
    }

    res_labels_map = {
        "loss": "Loss",
        "l0": "$\|\delta\|_0$",
        "l2": "$\|\delta\|_2$",
        # "l2_n": "$\\frac{{\|\delta\|_2}}{{N}}$",
        "linf": "$\|\delta\|_\infty$",
        "p2p": "Peak-to-Peak",
    }

    plottable_refs = ["mean", "max", "min"]

    prod = itertools.product(plottable_refs, list(results.keys()))

    plt.figure(2, tight_layout=True, figsize=(18, 12))

    for i, (ref, res) in enumerate(prod):

        maxmax = np.max(list(results[res]["max"].values()))
        denom = 10 ** np.floor(np.log10(maxmax))
        ceiled_maxmax = np.ceil(maxmax / denom) * denom

        plt.subplot(len(plottable_refs), len(list(results.keys())), i + 1)

        plot_per_measure(
            results[res][ref],
            "{ref} {res} per Step".format(
                ref=ref.capitalize(), res=res_labels_map[res].capitalize()
            ),
            x_label="Steps",
            y_label="{ref} {res}".format(
                ref=ref.capitalize(), res=res_labels_map[res]
            ),
            stddevs=None, # results[res]["std"] if ref == "mean" else None,
            y_max=ceiled_maxmax
        )
    plt.savefig(
        os.path.join(outpath, "measure_per_steps.png")
    )
    # plt.figure(2, tight_layout=True)
    # print(success_rate)
    # plt.barh(np.asarray(success_rate), width=0.5)
    # plt.title("Success Rate")
    # plt.savefig(
    #     os.path.join(outpath, "success_rate.png")
    # )
    #     plt.pause(10)
        # plt.figure(1)
        # plt.clf()
        # plt.figure(2)
        # plt.clf()


if __name__ == '__main__':
    outdir = sys.argv[1]
    log_file_paths = sys.argv[2:]
    if len(log_file_paths) > 1:
        for log_file_path in log_file_paths:
            main(log_file_path, outdir)
        plt.legend([l.replace("_", "-") for l in log_file_paths])
    elif len(log_file_paths) == 1:
        main(log_file_paths[0], outdir)

    else:
        pass
    plt.show()

