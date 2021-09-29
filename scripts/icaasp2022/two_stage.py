#!/usr/bin/env python3
import os

from copy import deepcopy

from cleverspeech import data
from cleverspeech import graph
from cleverspeech import models

from cleverspeech.config.arguments_parser import args
from cleverspeech.runtime.Execution import default_manager
from cleverspeech.utils.Utils import log


def cgd_cw_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.ClippedGradientDescent,
        random_scale=settings["delta_randomiser"],
        constraint_cls=graph.Constraints.L2,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder=settings["decoder"],
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.CWMaxMin,
        weight_settings=(1.0e3, 0.5),
        updateable=True,
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"],
    )
    attack.add_procedure(
        graph.Procedures.SuccessOnDecoding,
        steps=settings["nsteps"],
        update_step=settings["decode_step"],
    )

    return attack


def attack_run(master_settings):

    run = master_settings["set"]
    settings = deepcopy(master_settings)

    log("Starting {} run.".format(run))

    settings["attack_graph"] = run
    settings["decoder"] = "tf_greedy"
    settings["learning_rate"] = 10.0
    settings["rescale"] = 0.8
    # settings["max_examples"] = 50
    # settings["max_targets"] = 3992
    settings["batch_size"] = 50
    settings["decode_step"] = 25
    # settings["delta_randomiser"] = 0.01  set this manually using cli arguments
    settings["nsteps"] = 15000
    settings["constraint_update"] = "geom"
    settings["writer"] = "local_latest"
    settings["outdir"] = os.path.join(settings["outdir"], run)

    audios = data.ingress.two_stage.Audios(
        settings["audio_indir"],
    )

    transcriptions = data.ingress.two_stage.Targets(
        settings["targets_path"],
    )

    batch_gen = data.ingress.two_stage.BatchIterator(
        settings, audios, transcriptions
    )

    default_manager(
        settings,
        ATTACK_GRAPHS[run],
        batch_gen,
    )
    log("Finished {} run.".format(run))


ATTACK_GRAPHS = {
  "cw-twostage": cgd_cw_graph,
}


def main():

    log("", wrap=True)
    extra_args = {
        "set": [str, "cw-twostage", False, list(ATTACK_GRAPHS.keys())]
    }
    args(attack_run, additional_args=extra_args)


if __name__ == '__main__':
    main()
