#!/usr/bin/env python3
import os

from cleverspeech import data
from cleverspeech import graph
from cleverspeech import models

from cleverspeech.config.ExperimentArguments import args
from cleverspeech.runtime.Execution import default_manager
from cleverspeech.utils.Utils import log


def create_unbounded_graph(sess, batch, settings):

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
    attack.add_loss(
        graph.Losses.BEAM_SEARCH_ADV_LOSSES[settings["loss"]],
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


def create_cgd_evasion_graph(sess, batch, settings):

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
    attack.add_loss(
        graph.Losses.BEAM_SEARCH_ADV_LOSSES[settings["loss"]],
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

    attack_type = master_settings["attack_graph"]
    align = master_settings["align"]
    loss = master_settings["loss"]
    decoder = master_settings["decoder"]
    outdir = master_settings["outdir"]

    outdir = os.path.join(outdir, attack_type)
    outdir = os.path.join(outdir, "{}/".format(loss))
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))

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
    "unbounded": create_unbounded_graph,
    "cgd_evasion": create_cgd_evasion_graph,
}


def main():
    log("", wrap=True)

    extra_args = {
        "attack_graph": [str, "unbounded", False, ATTACK_GRAPHS.keys()],
        "loss": [str, None, True, graph.Losses.BEAM_SEARCH_ADV_LOSSES.keys()]
    }

    args(attack_run, additional_args=extra_args)


if __name__ == '__main__':
    main()




