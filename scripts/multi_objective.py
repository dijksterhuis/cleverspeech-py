#!/usr/bin/env python3
import os

from cleverspeech import data
from cleverspeech import graph
from cleverspeech import models

from cleverspeech.config.arguments_parser import args
from cleverspeech.runtime.Execution import default_manager
from cleverspeech.utils.Utils import log
from tensorflow import (
    equal as tf_equal,
    minimum as tf_minimum,
    greater as tf_greater,
    less as tf_less,
    add as tf_add,
    div as tf_div,
    multiply as tf_mult,
    where as tf_where,
    zeros_like as tf_zeros_like,
    ones_like as tf_ones_like
)

def linf_constraint_graph(sess, batch, settings):

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

    def ref_fn(losses):
        a, b = losses
        return tf_where(tf_equal(a, tf_zeros_like(a)), b, a)

    def ref_fn_b(*losses):
        a, b = losses
        return tf_where(tf_less(a, tf_ones_like(a)), b, a)

    def ref_fn2(losses):
        a, b = losses
        args = tf_less(a, tf_zeros_like(a)), tf_div(tf_ones_like(a), 1e-5), tf_div(1.0, a)
        return tf_add(a, tf_minimum(tf_where(*args), 10.0) * b)

    def ref_fn2_unclamped(losses):
        a, b = losses
        args = tf_less(a, tf_zeros_like(a)), tf_div(tf_ones_like(a), 1e-5), tf_div(1.0, a)
        return tf_add(a, tf_where(*args) * b)

    def ref_fn3(losses):
        a, b = losses
        return tf_add(a, tf_div(b, a + 1e-5))

    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.GREEDY[settings["loss1"]],
        updateable=True,
        weight_settings=(1.0, 0.5),
    )
    attack.add_loss(
        graph.losses.adversarial.AlignmentBased.NON_GREEDY[settings["loss2"]],
        updateable=True,
        weight_settings=(1.0, 0.5),
    )
    attack.create_loss_fn(ref_fn=ref_fn_b)
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


def attack_run(master_settings):
    batch_gen = data.ingress.helpers.create_batch_gen_fn(master_settings)

    default_manager(
        master_settings,
        ATTACK_GRAPHS[master_settings["attack_graph"]],
        batch_gen,
    )
    log("Finished run.")


ATTACK_GRAPHS = {
    "linf": linf_constraint_graph,
}


def main():
    log("", wrap=True)

    extra_args = {
        "attack_graph": [str, "linf", False, ATTACK_GRAPHS.keys()],
        "loss1": [str, None, True, graph.losses.adversarial.AlignmentBased.GREEDY.keys()],
        "loss2": [str, None, True, graph.losses.adversarial.AlignmentBased.NON_GREEDY.keys()],
    }

    args(attack_run, additional_args=extra_args)


if __name__ == '__main__':
    main()




