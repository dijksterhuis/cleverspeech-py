#!/usr/bin/env python3
import os
import random

from copy import deepcopy

from cleverspeech import data
from cleverspeech import graph
from cleverspeech import models

from cleverspeech.config.arguments_parser import args
from cleverspeech.runtime.Execution import default_manager
from cleverspeech.utils.Utils import log


def cgd_ctc_greedy_search_graph(sess, batch, settings):

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
        graph.losses.adversarial.AlignmentFree.CTCLoss,
        updateable=False,
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


def cgd_cw_dense_align_greedy_search_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
      graph.Paths.Dense
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


def cgd_cw_patchstart_align_greedy_search_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
      graph.Paths.StartPatch
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
    graph = run.split("_")[0]
    dataset = run.split("_")[1]
    settings = deepcopy(master_settings)

    log("Starting {} run.".format(run))

    settings["attack_graph"] = graph
    settings["data"] = dataset
    settings["decoder"] = "tf_greedy"
    # settings["learning_rate"] = 10.0
    # settings["delta_randomiser"] = 0.01
    settings["rescale"] = 0.8
    settings["batch_size"] = 50
    settings["decode_step"] = 25
    settings["nsteps"] = 15000
    settings["constraint_update"] = "geom"
    settings["writer"] = "local_latest"
    settings["outdir"] = os.path.join(
        master_settings["outdir"], os.path.join(dataset, graph)
    )

    # initial run without any random warm up

    audios = data.ingress.from_csv.Audios(
        settings["csv_file_path"],
        filter_term=settings["set"],
    )

    transcriptions = data.ingress.from_csv.Targets(
        settings["csv_file_path"]
    )

    batch_gen = data.ingress.from_csv.BatchIterator(
        settings, audios, transcriptions
    )
    log("Minimal settings run", wrap=False)
    settings["learning_rate"] = 10.0
    settings["delta_randomiser"] = 0.0
    settings["outdir"] = os.path.join(
        master_settings["outdir"], os.path.join(dataset, graph)
    )
    settings["outdir"] = os.path.join(
        settings["outdir"], os.path.join(
            "lr{}".format(settings["learning_rate"]),
            "r{}".format(settings["delta_randomiser"]),
        )
    )
    log(
        "Settings:\nLearning rate: {}\nDelta init: {}\nOutdir {}".format(
            settings["learning_rate"],
            settings["delta_randomiser"],
            settings["outdir"],
        ),
        wrap=True
    )

    default_manager(
        settings,
        ATTACK_GRAPHS[run],
        batch_gen,
    )

    # a bunch of runs with any random warm up

    for idx in range(9):

        audios = data.ingress.from_csv.Audios(
            settings["csv_file_path"],
            filter_term=settings["set"],
        )

        transcriptions = data.ingress.from_csv.Targets(
            settings["csv_file_path"]
        )

        batch_gen = data.ingress.from_csv.BatchIterator(
            settings, audios, transcriptions
        )

        log("Randomised settings run: {}".format(idx), wrap=False)
        settings["learning_rate"] = random.randint(10, 1000)
        settings["delta_randomiser"] = round(random.random(), 4)
        settings["outdir"] = os.path.join(
            master_settings["outdir"], os.path.join(dataset, graph)
        )
        settings["outdir"] = os.path.join(
            settings["outdir"], os.path.join(
                "lr{}".format(settings["learning_rate"]),
                "r{}".format(settings["delta_randomiser"]),
            )
        )
        log(
            "Settings:\nLearning rate: {}\nDelta init: {}\nOutdir {}".format(
                settings["learning_rate"],
                settings["delta_randomiser"],
                settings["outdir"],
            ),
            wrap=True
        )

        default_manager(
            settings,
            ATTACK_GRAPHS[run],
            batch_gen,
        )

    log("Finished all {} runs.".format(run))


ATTACK_GRAPHS = {
  "cw-dense_trimmed": cgd_cw_dense_align_greedy_search_graph,
  "cw-patchstart_trimmed": cgd_cw_patchstart_align_greedy_search_graph,
  "cw-patchstart_untrimmed": cgd_cw_patchstart_align_greedy_search_graph,
  "ctc-greedy_silence": cgd_ctc_greedy_search_graph,
  "ctc-greedy_trimmed": cgd_ctc_greedy_search_graph,
}


def main():

    log("", wrap=True)
    extra_args = {
        "set": [str, "ctc-greedy", False, list(ATTACK_GRAPHS.keys())],
        "csv_file_path":
            [
                str,
                "./cleverspeech/scripts/icaasp2020/failures.csv",
                False,
                None
            ],
    }
    args(attack_run, additional_args=extra_args)


if __name__ == '__main__':
    main()
