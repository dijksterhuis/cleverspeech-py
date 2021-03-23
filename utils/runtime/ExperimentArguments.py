import sys
import random
import argparse

import numpy as np
import tensorflow as tf

from cleverspeech.utils.Utils import log


def args(experiments):

    choices = list(experiments.keys())

    standard_non_required_args = {
        "gpu_device": int,
        "batch_size": int,
        "learning_rate": float,
        "nsteps": int,
        "decode_step": int,
        "constraint_update": str,
        "rescale": float,
        "max_spawns": int,
        "audio_indir": str,
        "outdir": str,
        "targets_path": str,
        "spawn_delay": int,
        "max_examples": int,
        "random_seed": int
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs=1,
        choices=choices,
    )

    for k, v in standard_non_required_args.items():
        parser.add_argument(
            "--" + k,
            nargs=1,
            type=v,
            default=[None],
            required=False
        )

    args = parser.parse_args()

    # Disable tensorflow looking at tf.app.flags.FLAGS but we get to keep
    # the args in the args variable. Otherwise we get the following exception
    # `absl.flags._exceptions.UnrecognizedFlagError: Unknown command line flag`
    # because tf detects the flags in `DeepSpeech.Utils.Config` and gets
    # confused when it finds a duplicate key like `batch_size`
    while len(sys.argv) > 1:
        sys.argv.pop()

    experiment = args.experiment[0]
    log("Running new experiment: {}".format(experiment))

    def update_master_settings(d, k, v):
        if v is not None:
            d.update({k: v})

    master_settings = {}

    # TODO -- how to automate the args.x part of this.

    update_master_settings(
        master_settings, "gpu_device", args.gpu_device[0]
    )
    update_master_settings(
        master_settings, "batch_size", args.batch_size[0]
    )
    update_master_settings(
        master_settings, "learning_rate", args.learning_rate[0]
    )
    update_master_settings(
        master_settings, "nsteps", args.nsteps[0]
    )
    update_master_settings(
        master_settings, "decode_step", args.decode_step[0]
    )
    update_master_settings(
        master_settings, "constraint_update", args.constraint_update[0]
    )
    update_master_settings(
        master_settings, "rescale", args.rescale[0]
    )
    update_master_settings(
        master_settings, "max_spawns", args.max_spawns[0]
    )
    update_master_settings(
        master_settings, "audio_indir", args.audio_indir[0]
    )
    update_master_settings(
        master_settings, "outdir", args.outdir[0]
    )
    update_master_settings(
        master_settings, "targets_path", args.targets_path[0]
    )
    update_master_settings(
        master_settings, "spawn_delay", args.spawn_delay[0]
    )
    update_master_settings(
        master_settings, "max_examples", args.max_examples[0]
    )
    update_master_settings(
        master_settings, "random_seed", args.random_seed[0]
    )

    if args.random_seed[0] is not None:
        seed = args.random_seed[0]
    else:
        seed = 420

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    experiments[experiment](master_settings)

    exit(0)
