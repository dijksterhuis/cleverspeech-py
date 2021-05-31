import sys
import random
import argparse

import numpy as np
import tensorflow as tf

from cleverspeech.utils.Utils import log


def args(attack_run, additional_args: dict = None):

    # choices = list(experiments.keys())

    standard_non_required_args = {
        "gpu_device": [int, 0, False, None],
        "batch_size": [int, 1, False, None],
        "learning_rate": [float, 10, False, None],
        "nsteps": [int, 10000, False, None],
        "decode_step": [int, 100, False, None],
        "constraint_update": [str, "geom", False, ["geom", "lin", "log"]],
        "rescale": [float, 0.9, False, None],
        "audio_indir": [str, "./samples/all/", False, None],
        "outdir": [str, "./adv/", False, None],
        "targets_path": [str, "./samples/cv-valid-test.csv", False, None],
        "max_examples": [int, 100, False, None],
        "max_targets": [int, 2000, False, None],
        "max_audio_file_bytes": [int, 120000, False, None],
        "beam_width": [int, 500, False, None],
        "random_seed": [int, 420, False, None],
        "decoder": [
            str, "batch", False, [
                "batch",
                "greedy",
                "ds",
                "batch_no_lm",
                "greedy_no_lm",
                "tf",
                "tf_greedy",
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

    master_settings = {}

    for k in standard_non_required_args.keys():
        update_master_settings(
            master_settings, k, arguments.__getattribute__(k)[0]
        )

    if additional_args is not None:
        for k in additional_args.keys():
            update_master_settings(
                master_settings, k, arguments.__getattribute__(k)[0]
            )

    if arguments.random_seed[0] is not None:
        seed = arguments.random_seed[0]
    else:
        seed = 420

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    attack_run(master_settings)
