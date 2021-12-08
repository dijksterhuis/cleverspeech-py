import argparse
import random
import sys

import numpy as np
import tensorflow as tf


def args(attack_run, additional_args: dict = None):
    from cleverspeech.graph.Paths import ALL_PATHS

    # choices = list(experiments.keys())

    standard_non_required_args = {
        "data_loader": [
            str,
            "mcv7-sentences",
            False,
            ["mcv7-sentences", "mcv7-singlewords", "mcv1-sentences", "csv", "json"],
            "Choose a dataset loader. [!!] csv and json require additional args to be added to a script."
        ],
        "--batch_size": [int, 1, False, None, "How many examples to generate simultaneously."],
        "--skip_n_batch": [int, 0, False, None, "Whether to skip a certain number of batches (in case of previous failures)."],
        "--learning_rate": [float, 10 / 2**15, False, None, "Learning rate/ Update step size for optimisation"],
        "--nsteps": [int, 10000, False, None, "How many iterations of optimisation updates."],
        "--decode_step": [int, 100, False, None, "Which step to pause and perform updates based on decoding results."],
        "--gpu_device": [int, 0, False, None, "Which GPU device ID number to execute on."],
        "--restart_step": [int, 2500, False, None, "[!!] Only used with graph.Procedures.*WithRandomRestarts types. Which step to perform random restarting."],
        "--constraint_update": [str, "geom", False, ["geom", "lin", "log"], "[!!] Only used when a graph.constraints.size class is applied. Method to update the upper bound on size."],
        "--rescale": [float, 0.9, False, None, "[!!] Only used when a graph.constraints.size class is applied. Constant for updating the upper bound on size."],
        "--outdir": [str, "./adv/", False, None, "Directory to write out results files to (will be created if doesn't exist)."],
        "--max_examples": [int, 100, False, None, "Maximum number of examples to generate. If max_examples > batch_size then run multiple batches. If max_examples < batch_size then alter batch_size = max_examples"],
        "--max_targets": [int, 2000, False, None, "Maximum number of transcriptions to search through for suitable targets."],
        "--max_audio_file_bytes": [int, 120000, False, None, "Only generate examples for original audio files less than this size in bytes"],
        "--pgd_rounding": [int, 0, False, [0, 1], "[!!] Depreciated."],
        "--delta_randomiser": [float, 0.0, False, None, "Initialise perturbations with random uniform with this absolute maximum amplitude."],
        "--beam_width": [int, 500, False, None, "How many beams in the decoder. Greedy search will ignore this flag."],
        # 4568 is the random seed used by DeepSpeech
        "--random_seed": [int, 4568, False, None, "Set the random seed for all future ops."],
        "--align": [
            str, list(ALL_PATHS.keys())[0], False, ALL_PATHS.keys(), "[!!] Only used when a graph.Paths class is in the graph. Select a method to generate a target alignment."
        ],
        "--decoder": [
            str, "batch", False, [
                "batch",
                "greedy",
                "ds",
                "batch_no_lm",
                "greedy_no_lm",
                "tf_beam",
                "tf_greedy",
                "hotfix_greedy",
            ],
            "Choose a decoder class."
        ],
        "--writer": [str, "local_latest", False, [
            "local_latest", "local_all", "s3_latest", "s3_all"
        ], "Where and how should results files be written out."],
    }

    full_args = standard_non_required_args

    if additional_args is not None:
        assert type(additional_args) is dict
        full_args.update(additional_args)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dry_run',
        dest='dry_run',
        action='store_true',
        help="Load everything but don't actually perform optimisation.",
    )
    parser.add_argument(
        '--enable_jit',
        dest='enable_jit',
        action='store_true',
        help="Enable JIT compilation (doesn't have a significant positive impact on performance)."

    )
    parser.add_argument(
        '--use_resource_variables',
        dest='use_resource_variables',
        action='store_true',
        help="Use TF V2 style resource variables."
    )
    parser.add_argument(
        '--no_step_logs',
        dest='no_step_logs',
        action='store_true',
        help="Don't log any stats to disk, only write the results files."
    )
    parser.set_defaults(dry_run=False)
    parser.set_defaults(enable_jit=False)
    parser.set_defaults(use_resource_variables=False)
    parser.set_defaults(no_step_logs=False)

    # parser.add_argument(
    #     "experiment",
    #     nargs=1,
    #     choices=choices,
    # )

    for k, v in full_args.items():
        if len(v) == 4:
            arg_type, arg_default, arg_required, arg_choices = v
            arg_help = None
        else:
            arg_type, arg_default, arg_required, arg_choices, arg_help = v

        if "--" not in k and arg_choices is None:
            parser.add_argument(
                k,
                nargs=1,
                type=arg_type,
                default=[arg_default],
                help=arg_help,
            )

        elif "--" not in k and arg_choices is not None:
            parser.add_argument(
                k,
                nargs=1,
                type=arg_type,
                default=[arg_default],
                choices=arg_choices,
                help=arg_help,
            )

        elif arg_choices is None:

            parser.add_argument(
                k,
                nargs=1,
                type=arg_type,
                default=[arg_default],
                required=arg_required,
                help=arg_help,
            )

        else:
            parser.add_argument(
                k,
                nargs=1,
                type=arg_type,
                default=[arg_default],
                required=arg_required,
                choices=arg_choices,
                help=arg_help,
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

    master_settings = {
        "dry_run": arguments.dry_run,
        "enable_jit": arguments.enable_jit,
        "use_resource_variables": arguments.use_resource_variables,
        "no_step_logs": arguments.no_step_logs,
    }

    for k in standard_non_required_args.keys():
        if "--" in k:
            k = k.lstrip("--")
        update_master_settings(
            master_settings, k, arguments.__getattribute__(k)[0]
        )

    if additional_args is not None:
        for k in additional_args.keys():
            if "--" in k:
                k = k.lstrip("--")
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
