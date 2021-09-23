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


def cgd_ctc_beam_search_graph(sess, batch, settings):

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
        decoder="batch_no_lm",
        beam_width=500,
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


def cgd_cw_gradient_path_beam_search_graph(sess, batch, settings):

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


def cgd_cw_ctcalign_greedy_search_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
      graph.Paths.CTC
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


def cgd_cw_monosparse_greedy_search_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
      graph.Paths.RandomMonotonicSparse
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


def cgd_logprobs_gradient_path_beam_search_graph(sess, batch, settings):

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
        decoder="batch_no_lm",
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentFree.SumLogProbsForward,
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


def cgd_logprobs_sparse_align_beam_search_graph(sess, batch, settings):

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
        decoder="batch_no_lm",
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.SumLogProbsForward,
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


def cgd_logprobs_mid_align_beam_search_graph(sess, batch, settings):

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
        decoder="batch_no_lm",
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.SumLogProbsForward,
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


def cgd_logprobs_dense_align_beam_search_graph(sess, batch, settings):

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
        decoder="batch_no_lm",
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.SumLogProbsForward,
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


def cgd_logprobs_patchstart_align_beam_search_graph(sess, batch, settings):

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
        decoder="batch_no_lm",
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.SumLogProbsForward,
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


def cgd_logprobs_patchend_align_beam_search_graph(sess, batch, settings):

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
        decoder="batch_no_lm",
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.SumLogProbsForward,
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


def cgd_logprobs_patchmid_align_beam_search_graph(sess, batch, settings):

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
        decoder="batch_no_lm",
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.SumLogProbsForward,
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


def cgd_logprobs_ctcalign_beam_search_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
      graph.Paths.CTC
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
        decoder="batch_no_lm",
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.SumLogProbsForward,
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


def cgd_logprobs_monosparse_beam_search_graph(sess, batch, settings):

    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
      graph.Paths.CTC
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
        decoder="batch_no_lm",
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.SumLogProbsForward,
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

    exp_graphs = ATTACK_GRAPHS[master_settings["set"]]

    for run in exp_graphs.keys():

        settings = deepcopy(master_settings)

        log("Starting {} run.".format(run))

        settings["attack_graph"] = run
        settings["outdir"] = os.path.join(settings["outdir"], run)

        audios = data.ingress.mcv_v1.Audios(
            settings["audio_indir"],
            settings["max_examples"],
            filter_term=".wav",
            max_file_size=settings["max_audio_file_bytes"],
            file_size_sort="shuffle,"
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
            exp_graphs[run],
            batch_gen,
        )
        log("Finished {} run.".format(run))


ATTACK_GRAPHS = {
  0: {
    "cw-sparse": cgd_cw_sparse_align_greedy_search_graph,
    "cw-mid": cgd_cw_mid_align_greedy_search_graph,
    "cw-dense": cgd_cw_dense_align_greedy_search_graph,
    "cw-ctcalign": cgd_cw_ctcalign_greedy_search_graph,
    "cw-monosparse": cgd_cw_monosparse_greedy_search_graph,
  },
  1: {
    "cw-grad": cgd_cw_gradient_path_greedy_search_graph,
    "cw-patchstart": cgd_cw_patchstart_align_greedy_search_graph,
    "cw-patchmid": cgd_cw_patchmid_align_greedy_search_graph,
    "cw-patchend": cgd_cw_patchend_align_greedy_search_graph,
  },
  2: {
    "lprobs-sparse": cgd_logprobs_sparse_align_beam_search_graph,
    "lprobs-mid": cgd_logprobs_mid_align_beam_search_graph,
    "lprobs-dense": cgd_logprobs_dense_align_beam_search_graph,
    "lprobs-ctcalign": cgd_logprobs_ctcalign_beam_search_graph,
    "lprobs-monosparse": cgd_logprobs_ctcalign_beam_search_graph,
  },
  3: {
    "lprobs-grad": cgd_logprobs_gradient_path_beam_search_graph,
    "lprobs-patchstart": cgd_logprobs_patchstart_align_beam_search_graph,
    "lprobs-patchmid": cgd_logprobs_patchmid_align_beam_search_graph,
    "lprobs-patchend": cgd_logprobs_patchend_align_beam_search_graph,
  },
  4: {
    "ctc-greedy": cgd_ctc_greedy_search_graph,
    # "ctc-beam": cgd_ctc_beam_search_graph,
  }
}


def main():

    log("", wrap=True)
    extra_args = {
        "set": [int, 0, False, list(ATTACK_GRAPHS.keys())]
    }
    args(attack_run, additional_args=extra_args)


if __name__ == '__main__':
    main()
