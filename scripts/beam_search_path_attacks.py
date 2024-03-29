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
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.NON_GREEDY[settings["loss"]],
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.SuccessOnDecodingWithRestarts,
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
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.NON_GREEDY[settings["loss"]],
        weight_settings=(1.0e3, 0.5),
        updateable=True,
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.SuccessOnDecodingWithRestarts,
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
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.NON_GREEDY[settings["loss"]],
        weight_settings=(1.0e3, 0.5),
        updateable=True,
    )
    attack.add_loss(
        graph.losses.Distance.L2CarliniLoss,
        weight_settings=(1.0e-3, 1),  # (1e-2, 1.5)
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
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.NON_GREEDY[settings["loss"]],
        weight_settings=(1.0e3, 0.5),
        updateable=True,
    )
    attack.add_loss(
        graph.losses.Distance.LinfLoss,
        weight_settings=(1.0e-3, 1),  # (1e-2, 1.5)
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
        update_step=settings["decode_step"],
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

    batch_gen = data.ingress.helpers.create_mcv_batch_gen_fn(master_settings)

    default_manager(
        master_settings,
        ATTACK_GRAPHS[master_settings["attack_graph"]],
        batch_gen,
    )
    log("Finished run.")


ATTACK_GRAPHS = {
    "box": only_box_constraint_graph,
    "cgd": clipped_gradient_descent_graph,
    "linf_l2": clipped_linf_with_l2_loss,
    "l2_linf": clipped_l2_with_linf_loss,
}


def main():

    extra_args = {
        "--attack_graph": [str, "box", False, ATTACK_GRAPHS.keys()],
        "loss": [str, None, True, graph.losses.adversarial.AlignmentBased.NON_GREEDY.keys()],
    }

    args(attack_run, additional_args=extra_args)


if __name__ == '__main__':
    main()




