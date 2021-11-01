import os
import sys
import random
import argparse

import numpy as np
import tensorflow as tf


def args(attack_run, additional_args: dict = None):
    from cleverspeech.graph.Paths import ALL_PATHS

    # choices = list(experiments.keys())

    standard_non_required_args = {
        "gpu_device": [int, 0, False, None],
        "batch_size": [int, 1, False, None],
        "skip_n_batch": [int, 0, False, None],
        "learning_rate": [float, 10 / 2**15, False, None],
        "nsteps": [int, 10000, False, None],
        "decode_step": [int, 100, False, None],
        "restart_step": [int, 2500, False, None],
        "constraint_update": [str, "geom", False, ["geom", "lin", "log"]],
        "rescale": [float, 0.9, False, None],
        "audio_indir": [str, "./samples/all/", False, None],
        "outdir": [str, "./adv/", False, None],
        "targets_path": [str, "./samples/cv-valid-test.csv", False, None],
        "max_examples": [int, 100, False, None],
        "max_targets": [int, 2000, False, None],
        "max_audio_file_bytes": [int, 120000, False, None],
        "pgd_rounding": [int, 0, False, [0, 1]],
        "delta_randomiser": [float, 0.0, False, None],
        "beam_width": [int, 500, False, None],
        # 4568 is the random seed used by DeepSpeech
        "random_seed": [int, 4568, False, None],
        "align": [
            str, list(ALL_PATHS.keys())[0], False, ALL_PATHS.keys()
        ],
        "align_repeat_factor": [float, None, False, None],
        "decoder": [
            str, "batch", False, [
                "batch",
                "greedy",
                "ds",
                "batch_no_lm",
                "greedy_no_lm",
                "tf_beam",
                "tf_greedy",
                "hotfix_greedy",
            ]
        ],
        "writer": [str, "local_latest", False, [
            "local_latest", "local_all", "s3_latest", "s3_all"
        ]],
    }

    if additional_args is not None:
        assert type(additional_args) is dict
        full_args = additional_args
        full_args.update(standard_non_required_args)

    else:
        full_args = standard_non_required_args

    parser = argparse.ArgumentParser()

    parser.add_argument('--dry_run', dest='dry_run', action='store_true')
    parser.set_defaults(dry_run=False)

    # parser.add_argument(
    #     "experiment",
    #     nargs=1,
    #     choices=choices,
    # )

    for k, v in full_args.items():

        arg_type, arg_default, arg_required, arg_choices = v

        if arg_choices is None:

            parser.add_argument(
                "--" + k,
                nargs=1,
                type=arg_type,
                default=[arg_default],
                required=arg_required,
            )

        else:
            parser.add_argument(
                "--" + k,
                nargs=1,
                type=arg_type,
                default=[arg_default],
                required=arg_required,
                choices=arg_choices
            )

    arguments = parser.parse_args()

    if arguments.align_repeat_factor[0] is not None:
        assert arguments.align[0] == "custom"

    if arguments.align[0] == "custom":
        assert arguments.align_repeat_factor[0] is not None

    # Disable tensorflow looking at tf.app.flags.FLAGS but we get to keep
    # the args in the args variable. Otherwise we get the following exception
    # `absl.flags._exceptions.UnrecognizedFlagError: Unknown command line flag`
    # because tf detects the flags in `DeepSpeech.Utils.Config` and gets
    # confused when it finds a duplicate key like `batch_size`
    while len(sys.argv) > 1:
        sys.argv.pop()

    # experiment = arguments.experiment[0]
    # log("Running new experiment: {}".format(experiment))

    def update_master_settings(d, k, v):
        if v is not None:
            d.update({k: v})

    master_settings = {
        "dry_run": arguments.dry_run
    }

    for k in standard_non_required_args.keys():
        update_master_settings(
            master_settings, k, arguments.__getattribute__(k)[0]
        )

    if additional_args is not None:
        for k in additional_args.keys():
            update_master_settings(
                master_settings, k, arguments.__getattribute__(k)[0]
            )

    master_settings["unique_run_id"] = random.getrandbits(32)

    if arguments.random_seed[0] is not None:
        seed = arguments.random_seed[0]
    else:
        seed = 420

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)  # also set per batch in ../runtime/Execution.py

    attack_run(master_settings)
