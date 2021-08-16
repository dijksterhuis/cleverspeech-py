#!/usr/bin/env python3

"""
Script to discover suitable batch sizes for your GPU.
A Titan Xp with ~12GiB can generate 150 adversarial examples simultaneously.
"""

import os
import traceback
import multiprocessing as mp

from tensorflow import errors as tf_errors

from cleverspeech import data
from cleverspeech import graph
from cleverspeech import models

from cleverspeech.config.ExperimentArguments import args
from cleverspeech.runtime.TensorflowRuntime import TFRuntime
from cleverspeech.utils.Utils import log


def custom_executor(settings, batch, attack_fn):

    tf_runtime = TFRuntime(settings["gpu_device"])
    with tf_runtime.session as sess, tf_runtime.device as tf_device:

        attack = attack_fn(sess, batch, settings)

        def do_nothing():
            return None

        for _ in attack.run():
            do_nothing()

        return True


def custom_manager(settings, attack_fn, batch_gen):

    for b_id, batch in batch_gen:

        attack_process = mp.Process(
            target=custom_executor,
            args=(settings, batch, attack_fn)
        )

        try:
            attack_process.start()
            while attack_process.is_alive():
                os_mem = os.popen('free -t').readlines()[-1].split()[1:]
                tot_m, used_m, free_m = map(int, os_mem)
                assert free_m > 0.1 * tot_m

        except tf_errors.ResourceExhaustedError as e:
            log("Attack failed due to GPU memory constraints.")
            attack_process.terminate()
            raise

        except AssertionError as e:
            log("Attack failed due to CPU memory constraints.")
            attack_process.terminate()
            raise

        except Exception as e:

            log("Attack failed due to some code problem.")

            tb = "".join(traceback.format_exception(None, e, e.__traceback__))

            s = "Something broke! Attack failed to run for these examples:\n"
            s += '\n'.join(batch.audios["basenames"])
            s += "\n\nError Traceback:\n{e}".format(e=tb)

            log(s, wrap=True)
            attack_process.terminate()
            raise


def ctc_only_box_constraint_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.BoxConstraintOnly,
        random_scale=settings["delta_randomiser"]
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )
    attack.add_loss(
        graph.Losses.BEAM_SEARCH_ADV_LOSSES[settings["loss"]],
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.SuccessOnDecoding,
        steps=settings["nsteps"],
        update_step=settings["decode_step"]
    )

    return attack


def cw_only_box_constraint_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
        graph.Paths.ALL_PATHS[settings["align"]]
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.BoxConstraintOnly,
        random_scale=settings["delta_randomiser"],
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )
    attack.add_loss(
        graph.Losses.CWMaxMin,
        use_softmax=False,
        k=settings["kappa"]
    )
    attack.create_loss_fn()
    attack.create_loss_fn()
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.SuccessOnDecoding,
        steps=settings["nsteps"],
        update_step=settings["decode_step"]
    )

    return attack


ATTACK_GRAPHS= {
    "ctc": ctc_only_box_constraint_graph,
    "cw": cw_only_box_constraint_graph,
}


def attack_run(master_settings):

    everything_is_okay = True

    batch_size = 1

    while everything_is_okay:

        master_settings["batch_size"] = batch_size
        master_settings["max_examples"] = batch_size
        master_settings["nsteps"] = 10
        master_settings["decode_step"] = 5

        batch_gen = data.ingress.mcv_v1.BatchIterator(master_settings)

        attack_graph = ATTACK_GRAPHS[master_settings["graph"]]

        if batch_size >= 1024:
            batch_size = 1024
            everything_is_okay = False

        try:

            log("testing for batch size: {}".format(batch_size), wrap=True)
            custom_manager(
                master_settings,
                attack_graph,
                batch_gen,
            )

        except tf_errors.ResourceExhaustedError as e:
            everything_is_okay = False
            batch_size = batch_size // 2

        except AssertionError as e:
            everything_is_okay = False
            batch_size = batch_size // 2

        else:
            batch_size *= 2

    log("biggest batch size: {}".format(batch_size), wrap=True)
    log("Finished all runs.")


def main():
    log("", wrap=True)

    extra_args = {
        'graph': [str, "ctc", False, ATTACK_GRAPHS.keys()],
    }

    args(attack_run, additional_args=extra_args)


if __name__ == '__main__':
    main()
