import numpy as np
from collections import OrderedDict
from cleverspeech.data.egress.eval import DetectionMetrics
from cleverspeech.data.egress.eval import NoiseWeightings


def fix_padding(delta, original, advex):

    padding = delta.size - original.size
    if padding > 0:
        delta = delta[0:original.size]
        advex = advex[0:original.size]
    else:
        pass

    return delta, original, advex


def create_stats_dictionary(delta, original, adv):

    return OrderedDict(
        [
            ("snr_energy_db", DetectionMetrics.snr_energy_db(delta, original)),
            ("snr_power_db", DetectionMetrics.snr_power_db(delta, original)),
            ("snr_seg", DetectionMetrics.snr_segmented(delta, original)),
            # ("snr_energy", DetectionMetrics.snr_energy(adv, original)),
            # ("snr_power", DetectionMetrics.snr_power(adv, original)),
            # ("snr_seg", DetectionMetrics.snr_segmented(adv, original)),
            ("rms_amp_db_delta", DetectionMetrics.rms_amplitude_db(delta)),
            ("rms_amp_db_adv", DetectionMetrics.rms_amplitude_db(adv)),
            ("rms_amp_db_orig", DetectionMetrics.rms_amplitude_db(original)),
            # ("rms_amp_raw_delta", rms_amplitude_db(delta)),
            # ("rms_amp_raw_adv", rms_amplitude_db(adv)),
            # ("rms_amp_raw_orig", rms_amplitude_db(original)),
            ("energy_db_delta", DetectionMetrics.energy_db(delta)),
            ("energy_db_adv", DetectionMetrics.energy_db(adv)),
            ("energy_db_orig", DetectionMetrics.energy_db(original)),
            # ("energy_raw_delta", energy_db(delta)),
            # ("energy_raw_adv", energy_db(adv)),
            # ("energy_raw_orig", energy_db(original)),
            ("power_db_delta", DetectionMetrics.power_db(delta)),
            ("power_db_adv", DetectionMetrics.power_db(adv)),
            ("power_db_orig", DetectionMetrics.power_db(original)),
            # ("power_raw_delta", power_db(delta)),
            # ("power_raw_adv", power_db(adv)),
            # ("power_raw_orig", power_db(original)),
        ]
    )


def get_perceptual_stats(example_results):

    delta = example_results["deltas"]
    original = example_results["audio"]
    adv = example_results["advs"]

    # -- de-pad the delta and advex audio
    delta, original, adv = fix_padding(
        delta,
        original.astype(np.float32),
        adv
    )

    # -- Apply noise weightings
    a_delta = NoiseWeightings.a_weighting(delta)
    a_original = NoiseWeightings.a_weighting(original)
    a_adv = NoiseWeightings.a_weighting(adv)
    itu_delta = NoiseWeightings.itu_weighting(delta)
    itu_original = NoiseWeightings.itu_weighting(original)
    itu_adv = NoiseWeightings.itu_weighting(adv)

    # -- Generate the actual stats.
    # Note that many of these stats actually give us the same results. Left in
    # as it's useful to sanity check results.

    plain_stats = create_stats_dictionary(
        delta, original, adv
    )
    a_weighted_stats = create_stats_dictionary(
        a_delta, a_original, a_adv
    )
    itu_weighted_stats = create_stats_dictionary(
        itu_delta, itu_original, itu_adv
    )

    stats = OrderedDict(
        [
            ("samples", original.size),
            ("l1_delta", DetectionMetrics.lnorm(delta, norm=1)),
            ("l2_delta", DetectionMetrics.lnorm(delta, norm=2)),
            ("linf_delta", DetectionMetrics.lnorm(delta, norm=np.inf)),
            ("NO_WEIGHT", plain_stats),
            ("A", a_weighted_stats),
            ("ITU", itu_weighted_stats),
        ]
    )
    return stats
