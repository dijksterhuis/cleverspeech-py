import tensorflow as tf

from cleverspeech.utils.Utils import l_map, log


def convert_to_batch_from_one(x, size):
    return l_map(lambda _: x, range(size))


def get_decoding_from_one_decoder(attack, decoder_type="batch"):

    top_5_decodings, top_5_probs = attack.victim.inference(
        attack.batch,
        feed=attack.feeds.attack,
        decoder=decoder_type,
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

    return result


def get_decodings_from_all_decoders(attack):
    decoders = ["greedy", "batch"]

    results = [
        get_decoding_from_one_decoder(attack, decoder_type=decoder) for decoder in decoders
    ]

    return results


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
    excludes = ["max_samples", "max_feats"]
    auds = {k: v for k, v in batch.audios.items() if k not in excludes}
    return auds


def get_success_bools(attack):
    return l_map(
        lambda x: x, attack.procedure.check_for_successful_examples()
    )


def get_constraint_raw_distances(attack, deltas):
    return l_map(
        attack.hard_constraint.analyse, deltas
    )


def convert_to_epsilon(value, reference):
    return l_map(
        lambda x: x[0] / x[1], zip(value, reference)
    )


def get_unbounded_attack_state(attack):
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

    decodings = get_decoding_from_one_decoder(attack, decoder_type="batch")

    each_graph_loss_transposed = get_batched_losses(attack)

    tf_graph_variables = [
        attack.loss_fn,
        attack.perturbations,
        attack.adversarial_examples,
        attack.delta_graph.opt_vars,
        attack.victim.logits,
        tf.transpose(attack.victim.raw_logits, [1, 0, 2]),
        attack.optimiser.gradients,
    ]

    [
        total_losses,
        deltas,
        adv_audio,
        delta_vars,
        softmax_logits,
        raw_logits,
        gradients,
    ] = get_tf_graph_variables(tf_graph_variables, attack.procedure.tf_run)

    batched_results = {
        "step": batched_steps,
        "tokens": batched_tokens,
        "losses": each_graph_loss_transposed,
        "total_loss": total_losses,
        "deltas": deltas,
        "advs": adv_audio,
        "gradients": gradients,
        "delta_vars": [d for d in delta_vars[0]],
        "softmax_logits": softmax_logits,
        "raw_logits": raw_logits,
        "success": get_success_bools(attack)
    }
    batched_results.update(get_audio_batch(attack.batch))
    batched_results.update(get_target_batch(attack.batch))
    batched_results.update(decodings)

    return batched_results


def get_evasion_attack_state(attack):
    """
    Get the current values of a bunch of attack graph variables.

    """

    batched_results = get_unbounded_attack_state(attack)

    tf_graph_variables = [
        attack.hard_constraint.bounds,
        attack.perturbations,
    ]

    [
        bounds_raw,
        deltas,
    ] = get_tf_graph_variables(tf_graph_variables, attack.procedure.tf_run)

    # TODO: Fix nesting here or over in file write subprocess (as is now)?
    initial_tau = attack.hard_constraint.initial_taus
    distance_raw = get_constraint_raw_distances(attack, deltas)
    bound_eps = convert_to_epsilon(bounds_raw, initial_tau)
    distance_eps = convert_to_epsilon(distance_raw, initial_tau)

    additional_results = {
        "initial_taus": initial_tau,
        "bounds_raw": bounds_raw,
        "distances_raw": distance_raw,
        "bounds_eps": bound_eps,
        "distances_eps": distance_eps,
    }

    batched_results.update(additional_results)

    return batched_results

