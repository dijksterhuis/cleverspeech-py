#!/usr/bin/env python3
import os

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
        update_method="geom",
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder="tf_greedy",
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


def cgd_cw_gradient_path_greedy_search_graph(sess, batch, settings):

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
        update_method="geom",
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder="tf_greedy",
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentFree.CWMaxMin,
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


def cgd_cw_sparse_align_greedy_search_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
      graph.Paths.Sparse
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.ClippedGradientDescent,
        random_scale=settings["delta_randomiser"],
        constraint_cls=graph.Constraints.L2,
        r_constant=settings["rescale"],
        update_method="geom",
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder="tf_greedy",
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


def cgd_cw_mid_align_greedy_search_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
      graph.Paths.Mid
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.ClippedGradientDescent,
        random_scale=settings["delta_randomiser"],
        constraint_cls=graph.Constraints.L2,
        r_constant=settings["rescale"],
        update_method="geom",
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder="tf_greedy",
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
        update_method="geom",
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder="tf_greedy",
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
        update_method="geom",
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder="tf_greedy",
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


def cgd_cw_patchend_align_greedy_search_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
      graph.Paths.EndPatch
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.ClippedGradientDescent,
        random_scale=settings["delta_randomiser"],
        constraint_cls=graph.Constraints.L2,
        r_constant=settings["rescale"],
        update_method="geom",
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder="tf_greedy",
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


def cgd_cw_patchmid_align_greedy_search_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
      graph.Paths.MidPatch
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.ClippedGradientDescent,
        random_scale=settings["delta_randomiser"],
        constraint_cls=graph.Constraints.L2,
        r_constant=settings["rescale"],
        update_method="geom",
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder="tf_greedy",
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
    settings["outdir"] = os.path.join(settings["outdir"], run)

    audios = data.ingress.mcv_v1.Audios(
        settings["audio_indir"],
        settings["max_examples"],
        filter_term=".wav",
        max_file_size=settings["max_audio_file_bytes"],
        file_size_sort="shuffle"
    )

    transcriptions = data.ingress.mcv_v1.Targets(
        settings["targets_path"],
        settings["max_targets"],
    )

    batch_gen = data.ingress.mcv_v1.BatchIterator(
        settings, audios, transcriptions
    )

    default_manager(
        settings,
        ATTACK_GRAPHS[run],
        batch_gen,
    )
    log("Finished {} run.".format(run))


ATTACK_GRAPHS = {
  "cw-sparse": cgd_cw_sparse_align_greedy_search_graph,
  "cw-mid": cgd_cw_mid_align_greedy_search_graph,
  "cw-dense": cgd_cw_dense_align_greedy_search_graph,
  "cw-grad": cgd_cw_gradient_path_greedy_search_graph,
  "cw-patchstart": cgd_cw_patchstart_align_greedy_search_graph,
  "cw-patchmid": cgd_cw_patchmid_align_greedy_search_graph,
  "cw-patchend": cgd_cw_patchend_align_greedy_search_graph,
  "ctc-greedy": cgd_ctc_greedy_search_graph,
}


def main():

    log("", wrap=True)
    extra_args = {
        "set": [str, "ctc-greedy", False, list(ATTACK_GRAPHS.keys())]
    }
    args(attack_run, additional_args=extra_args)


if __name__ == '__main__':
    main()