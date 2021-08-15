#!/usr/bin/env python3
import os
from cleverspeech import data
from cleverspeech import graph
from cleverspeech import models

from cleverspeech.config.ExperimentArguments import args
from cleverspeech.runtime.Execution import default_manager
from cleverspeech.utils.Utils import log


def only_box_constraint_graph(sess, batch, settings):

    attack = graph.AttackConstructors.UnboundedAttackConstructor(
        sess, batch
    )
    attack.add_path_search(
        graph.Paths.ALL_PATHS[settings["align"]]
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_perturbation_subgraph(
        graph.PerturbationSubGraphs.Independent
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )
    if settings["loss"] in KAPPA_LOSSES:
        attack.add_loss(
            graph.Losses.GREEDY_SEARCH_ADV_LOSSES[settings["loss"]],
            use_softmax=settings["use_softmax"],
            k=settings["kappa"]
        )
    else:
        attack.add_loss(
            graph.Losses.GREEDY_SEARCH_ADV_LOSSES[settings["loss"]],
            use_softmax=settings["use_softmax"],
        )
    attack.create_loss_fn()
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.Unbounded,
        steps=settings["nsteps"],
        update_step=settings["decode_step"],
        apply_pgd_rounding=settings["pgd_rounding"],
        apply_warm_up=settings["random_warm_up"],
    )

    return attack


def clipped_gradient_descent_graph(sess, batch, settings):

    attack = graph.AttackConstructors.EvasionAttackConstructor(
        sess, batch
    )
    attack.add_path_search(
        graph.Paths.ALL_PATHS[settings["align"]]
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_hard_constraint(
        graph.Constraints.L2,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_perturbation_subgraph(
        graph.PerturbationSubGraphs.Independent
    )
    attack.add_victim(
        models.DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )
    if settings["loss"] in KAPPA_LOSSES:
        attack.add_loss(
            graph.Losses.GREEDY_SEARCH_ADV_LOSSES[settings["loss"]],
            use_softmax=settings["use_softmax"],
            k=settings["kappa"]
        )
    else:
        attack.add_loss(
            graph.Losses.GREEDY_SEARCH_ADV_LOSSES[settings["loss"]],
            use_softmax=settings["use_softmax"],
        )
    attack.create_loss_fn()
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.EvasionCGD,
        steps=settings["nsteps"],
        update_step=settings["decode_step"],
        apply_pgd_rounding=settings["pgd_rounding"],
        apply_warm_up=settings["random_warm_up"],
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

    attack_type = master_settings["attack_graph"]
    align = master_settings["align"]
    loss = master_settings["loss"]
    decoder = master_settings["decoder"]
    kappa = master_settings["kappa"]
    outdir = master_settings["outdir"]

    outdir = os.path.join(outdir, attack_type)
    outdir = os.path.join(outdir, "{}/".format(loss))
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))
    outdir = os.path.join(outdir, "k{}/".format(kappa))

    master_settings["outdir"] = outdir
    master_settings["attack type"] = attack_type

    batch_gen = data.ingress.mcv_v1.BatchIterator(master_settings)

    default_manager(
        master_settings,
        ATTACK_GRAPHS[master_settings["attack_graph"]],
        batch_gen,
    )
    log("Finished run.")


ATTACK_GRAPHS = {
    "box": only_box_constraint_graph,
    "cgd": clipped_gradient_descent_graph,
}

KAPPA_LOSSES = ["cw", "cw-toks", "weightedmaxmin", "adaptivekappa"]


def main():
    log("", wrap=True)

    extra_args = {
        "attack_graph": [str, "box", True, ATTACK_GRAPHS.keys()],
        "loss": [str, None, True, graph.Losses.GREEDY_SEARCH_ADV_LOSSES.keys()],
        "kappa": [float, None, False, None],
        'use_softmax': [int, 0, False, [0, 1]],
    }

    args(attack_run, additional_args=extra_args)


if __name__ == '__main__':
    main()