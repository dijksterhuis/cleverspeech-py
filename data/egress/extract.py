import tensorflow as tf

from cleverspeech.utils.Utils import l_map, log


def convert_to_batch_from_one(x, size):
    return l_map(lambda _: x, range(size))


def get_decodings(attack):

    if attack.victim.decoder in ["batch", "ds", "batch_no_lm"] and attack.victim.beam_width >= 5:

        top_5_decodings, top_5_probs = attack.victim.inference(
            attack.batch,
            feed=attack.feeds.attack,
            top_five=True,
        )

        best_decodings = [
            top_5_decodings[idx][0] for idx in range(attack.batch.size)
        ]
        best_probs = [
            top_5_probs[idx][0] for idx in range(attack.batch.size)
        ]

        result = {
            "decodings": best_decodings,
            "probs": best_probs,
            "top_five_decodings": top_5_decodings,
            "top_five_probs": top_5_probs,
        }

    else:

        decodings, probs = attack.victim.inference(
            attack.batch,
            feed=attack.feeds.attack,
        )

        result = {
            "decodings": decodings,
            "probs": probs,
        }

    return result


def get_batched_losses(attack):
    graph_losses = [l.loss_fn for l in attack.loss]
    losses = attack.procedure.tf_run(graph_losses)

    losses_transposed = [
        [
            losses[loss_idx][batch_idx] for loss_idx in range(len(attack.loss))
        ] for batch_idx in range(attack.batch.size)
    ]

    return losses_transposed


def get_tf_graph_variables(tf_graph_variables: list, tf_run):
    return tf_run(tf_graph_variables)


def get_target_batch(batch):
    excludes = ["tokens"]
    targs = {k: v for k, v in batch.targets.items() if k not in excludes}
    return targs


def get_audio_batch(batch):
    excludes = ["max_samples", "max_feats", "ground_truth"]
    auds = {k: v for k, v in batch.audios.items() if k not in excludes}
    return auds


def get_success_bools(batch, decodings):
    return l_map(
        lambda x: x[0] == x[1], zip(decodings, batch.targets["phrases"])
    )


def get_constraint_raw_distances(attack, deltas):
    return l_map(
        attack.delta_graph.hard_constraint.analyse, deltas
    )


def convert_to_epsilon(value, reference):
    return l_map(
        lambda x: x[0] / x[1], zip(value, reference)
    )


def get_attack_state(attack, successes):
    """
    Get the current values of a bunch of attack graph variables -- for unbounded
    attacks only.
    """

    batched_steps = convert_to_batch_from_one(
        attack.procedure.current_step,
        attack.batch.size
    )
    batched_tokens = convert_to_batch_from_one(
        attack.batch.targets["tokens"],
        attack.batch.size
    )

    # decodings = get_decodings(attack)

    each_graph_loss_transposed = get_batched_losses(attack)

    tf_graph_variables = [
        attack.loss_fn,
        attack.perturbations,
        attack.adversarial_examples,
        attack.victim.logits,
        tf.transpose(attack.victim.raw_logits, [1, 0, 2]),
        # attack.optimiser.gradients,
    ]

    if attack.size_constraint is not None:

        initial_taus = attack.size_constraint.initial_taus
        bounds_raw = attack.size_constraint.bounds

        tf_graph_variables.append(bounds_raw)

        np_vars = get_tf_graph_variables(
            tf_graph_variables, attack.procedure.tf_run
        )

        [
            total_losses,
            deltas,
            adv_audio,
            softmax_logits,
            raw_logits,
            # gradients,
            bounds_raw,
        ] = np_vars

    else:

        initial_taus = [[None] for _ in range(attack.batch.size)]
        bounds_raw = [[None] for _ in range(attack.batch.size)]

        np_vars = get_tf_graph_variables(
            tf_graph_variables, attack.procedure.tf_run
        )

        [
            total_losses,
            deltas,
            adv_audio,
            softmax_logits,
            raw_logits,
            # gradients,
        ] = np_vars

    batched_results = {
        "step": batched_steps,
        "tokens": batched_tokens,
        "losses": each_graph_loss_transposed,
        "total_loss": total_losses,
        "deltas": deltas,
        "advs": adv_audio,
        # "gradients": gradients,
        "softmax_logits": softmax_logits,
        "raw_logits": raw_logits,
        # "success": get_success_bools(attack.batch, decodings["decodings"]),
        "bounds_raw": bounds_raw,
        "initial_taus": initial_taus,
    }
    batched_results.update(get_audio_batch(attack.batch))
    batched_results.update(get_target_batch(attack.batch))
    # batched_results.update(decodings)
    batched_results.update({"successes": successes})

    return batched_results

