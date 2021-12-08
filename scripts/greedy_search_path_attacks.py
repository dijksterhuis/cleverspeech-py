#!/usr/bin/env python3
import os
from cleverspeech import data
from cleverspeech import graph
from cleverspeech import models

from cleverspeech.config.arguments_parser import args
from cleverspeech.runtime.Execution import default_manager
from cleverspeech.utils.Utils import log


def only_box_constraint_graph(sess, batch, settings):
    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
        graph.Paths.ALL_PATHS[settings["align"]]
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_box_constraint(
        graph.constraints.box.ClippedBoxConstraint
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.IndependentVariables,
        random_scale=settings["delta_randomiser"]
    )
    attack.create_adversarial_examples()
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )
    if settings["loss"] in KAPPA_LOSSES:
        attack.add_loss(
            graph.losses.adversarial.AlignmentBased.GREEDY[settings["loss"]],
            use_softmax=settings["use_softmax"],
            k=settings["kappa"]
        )
    else:
        attack.add_loss(
            graph.losses.adversarial.AlignmentBased.GREEDY[settings["loss"]],
            use_softmax=settings["use_softmax"],
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


def clipped_gradient_descent_graph(sess, batch, settings):
    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
        graph.Paths.ALL_PATHS[settings["align"]]
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_box_constraint(
        graph.constraints.box.ClippedBoxConstraint
    )
    attack.add_size_constraint(
        graph.constraints.size.L2,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.IndependentVariables,
        random_scale=settings["delta_randomiser"]
    )
    attack.create_adversarial_examples()
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )

    if settings["loss"] in KAPPA_LOSSES:
        attack.add_loss(
            graph.losses.adversarial.AlignmentBased.GREEDY[settings["loss"]],
            use_softmax=settings["use_softmax"],
            k=settings["kappa"],
            weight_settings=(1.0e3, 0.5),
            updateable=True,
        )
    else:
        attack.add_loss(
            graph.losses.adversarial.AlignmentBased.GREEDY[settings["loss"]],
            use_softmax=settings["use_softmax"],
            weight_settings=(1.0e3, 0.5),
            updateable=True,
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


def clipped_linf_with_l2_loss(sess, batch, settings):
    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
        graph.Paths.ALL_PATHS[settings["align"]]
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_box_constraint(
        graph.constraints.box.ClippedBoxConstraint
    )
    attack.add_size_constraint(
        graph.constraints.size.Linf,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.IndependentVariables,
        random_scale=settings["delta_randomiser"]
    )
    attack.create_adversarial_examples()
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )

    if settings["loss"] in KAPPA_LOSSES:
        attack.add_loss(
            graph.losses.adversarial.AlignmentBased.GREEDY[settings["loss"]],
            use_softmax=settings["use_softmax"],
            k=settings["kappa"],
            weight_settings=(1.0e3, 0.5),
            updateable=True,
        )
    else:
        attack.add_loss(
            graph.losses.adversarial.AlignmentBased.GREEDY[settings["loss"]],
            use_softmax=settings["use_softmax"],
            weight_settings=(1.0e3, 0.5),
            updateable=True,
        )
    attack.add_loss(
        graph.losses.Distance.L2CarliniLoss,
        weight_settings=(1.0e-3, 1),
        updateable=True,
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


def clipped_l2_with_linf_loss(sess, batch, settings):
    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
        graph.Paths.ALL_PATHS[settings["align"]]
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_box_constraint(
        graph.constraints.box.ClippedBoxConstraint
    )
    attack.add_size_constraint(
        graph.constraints.size.Linf,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.IndependentVariables,
        random_scale=settings["delta_randomiser"]
    )
    attack.create_adversarial_examples()
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )

    if settings["loss"] in KAPPA_LOSSES:
        attack.add_loss(
            graph.losses.adversarial.AlignmentBased.GREEDY[settings["loss"]],
            use_softmax=settings["use_softmax"],
            k=settings["kappa"],
            weight_settings=(1.0e3, 0.5),
            updateable=True,
        )
    else:
        attack.add_loss(
            graph.losses.adversarial.AlignmentBased.GREEDY[settings["loss"]],
            use_softmax=settings["use_softmax"],
            weight_settings=(1.0e3, 0.5),
            updateable=True,
        )
    attack.add_loss(
        graph.losses.Distance.LinfLoss,
        weight_settings=(1.0e-3, 1),
        updateable=False,
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.SuccessOnDecoding,
        steps=settings["nsteps"],
        update_step=settings["decode_step"],
    )

    return attack


def psycho_graph(sess, batch, settings):
    attack = graph.GraphConstructor.Constructor(
        sess, batch, settings
    )
    attack.add_path_search(
        graph.Paths.ALL_PATHS[settings["align"]]
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_box_constraint(
        graph.constraints.box.ClippedBoxConstraint
    )
    attack.add_size_constraint(
        graph.constraints.size.Linf,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_frequency_masker(
        graph.Maskers.PsychoacousticMasking,
    )
    attack.add_perturbation_subgraph(
        graph.Perturbations.IndependentVariables,
        random_scale=settings["delta_randomiser"]
    )
    attack.create_adversarial_examples()
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )
    if settings["loss"] in KAPPA_LOSSES:
        attack.add_loss(
            graph.losses.adversarial.AlignmentBased.GREEDY[settings["loss"]],
            use_softmax=settings["use_softmax"],
            k=settings["kappa"],
            weight_settings=(1.0e3, 0.75),
            updateable=True,
        )
    else:
        attack.add_loss(
            graph.losses.adversarial.AlignmentBased.GREEDY[settings["loss"]],
            use_softmax=settings["use_softmax"],
            weight_settings=(1.0e3, 0.75),
            updateable=True,
        )
    attack.add_loss(
        graph.losses.Distance.PsychoacousticFrequencyMaskingLoss,
        updateable=False
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.SuccessOnDecodingWithRestartsAndEarlyStop,
        steps=settings["nsteps"],
        update_step=settings["decode_step"],
        restart_step=settings["restart_step"],
    )

    return attack


def attack_run(master_settings):
    """
    Use Carlini & Wagner's improved loss function form the original audio paper,
    but reintroduce kappa from the image attack as we're looking to perform
    targeted maximum-confidence evasion attacks --- i.e. not just find minimum
    perturbations.

    :param master_settings: a dictionary of arguments to run the attack, as
    defined by command line arguments. Will override the settings dictionary
    defined below.

    :return: None
    """

    master_settings["use_softmax"] = bool(master_settings["use_softmax"])

    if master_settings["loss"] in KAPPA_LOSSES:
        assert master_settings["kappa"] is not None
        if master_settings["use_softmax"] is True:
            assert 0 <= master_settings["kappa"] < 1
        else:
            assert master_settings["kappa"] >= 0
    else:
        master_settings["kappa"] = None

    batch_gen = data.ingress.helpers.create_mcv_batch_gen_fn(master_settings)

    default_manager(
        master_settings,
        ATTACK_GRAPHS[master_settings["attack_graph"]],
        batch_gen,
    )


ATTACK_GRAPHS = {
    "box": only_box_constraint_graph,
    "cgd": clipped_gradient_descent_graph,
    "linf_l2": clipped_linf_with_l2_loss,
    "l2_linf": clipped_l2_with_linf_loss,
    "psycho": psycho_graph,
}

KAPPA_LOSSES = ["cw", "weightedmaxmin", "adaptivekappa"]


def main():

    extra_args = {
        "--attack_graph": [
            str,
            "box",
            False,
            ATTACK_GRAPHS.keys(),
            "Choose the attack graph for this script -- default 'box' constraint graph."
        ],
        "loss": [
            str,
            "cw",
            True,
            graph.losses.adversarial.AlignmentFree.GRADIENT_PATHS.keys(),
            "Choose the loss function for this script.",
        ],
        "--kappa": [float, None, False, None, "Select a value for kappa constant (for CW flavour losses)"],
        '--use_softmax': [int, 0, False, [0, 1], "0 = use raw activations, 1 = use softmax vectors, default 0"],
    }

    args(attack_run, additional_args=extra_args)


if __name__ == '__main__':
    main()
