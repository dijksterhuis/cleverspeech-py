import os
import sys

import numpy as np

from collections import OrderedDict
from cleverspeech.utils.Utils import log, load_wavs

from cleverspeech.data.egress.eval.DetectionMetrics import lnorm
from cleverspeech.data.egress.eval.DetectionMetrics import power_db
from cleverspeech.data.egress.eval.DetectionMetrics import energy_db
from cleverspeech.data.egress.eval.DetectionMetrics import rms_amplitude_db
from cleverspeech.data.egress.eval.DetectionMetrics import snr_power_db
from cleverspeech.data.egress.eval.DetectionMetrics import snr_power
from cleverspeech.data.egress.eval.DetectionMetrics import snr_energy_db
from cleverspeech.data.egress.eval.DetectionMetrics import snr_energy
from cleverspeech.data.egress.eval.DetectionMetrics import snr_segmented

from cleverspeech.data.egress.eval.NoiseWeightings import a_weighting
from cleverspeech.data.egress.eval.NoiseWeightings import itu_weighting


def get_fps(indir):
    for f in os.listdir(indir):
        yield os.path.join(indir, f)


def fix_padding(delta, original, advex):

    padding = delta.size - original.size
    if padding > 0:
        delta = delta[0:original.size]
        advex = advex[0:original.size]
    else:
        pass

    return delta, original, advex


def normalise_16_bit(array):
    if max(array) <= 1.0 or min(array) >= -1.0:
        array[array > 0] = array[array > 0] * (2 ** 15 - 1)
        array[array < 0] = array[array < 0] * (2 ** 15)
    return array


def type_check_str(x):
    if type(x) is np.float:
        x = float(x)
    elif type(x) is np.int:
        x = int(x)
    return str(x)


def batch_generate_statistic_file(indir):

    assert os.path.exists(indir)

    deltas_file_paths = [fp for fp in get_fps(indir) if "delta.wav" in fp]
    origs_file_paths = [fp for fp in get_fps(indir) if "original.wav" in fp]
    advexs_file_paths = [fp for fp in get_fps(indir) if "advex.wav" in fp]

    deltas_file_paths.sort()
    origs_file_paths.sort()
    advexs_file_paths.sort()

    deltas = load_wavs(deltas_file_paths, "float32")
    advexs = load_wavs(advexs_file_paths, "float32")
    origs = load_wavs(origs_file_paths, "float32")

    gen = enumerate(
        zip(
            deltas_file_paths,
            origs_file_paths,
            advexs_file_paths,
            deltas,
            origs,
            advexs
        )
    )

    stats_out_filepath = os.path.join(indir, "stats.csv")

    for i, (delta_fp, orig_fp, adv_fp, delta, original, adv) in gen:

        # delta and adv are both padded to the maximum length of an example in
        # batch -- so we resize them back to equal the original.
        delta, original, adv = fix_padding(delta, original.astype(np.float32), adv)

        # We need to return to the 16-bit integer range.
        # Otherwise our results cannot be compared to previous work.
        delta = normalise_16_bit(delta)
        original = normalise_16_bit(original)
        adv = normalise_16_bit(adv)

        # Apply noise weightings
        a_delta = a_weighting(delta)
        a_original = a_weighting(original)
        a_adv = a_weighting(adv)
        itu_delta = itu_weighting(delta)
        itu_original = itu_weighting(original)
        itu_adv = itu_weighting(adv)

        # Generate the actual stats. Note that many of these stats actually give
        # use the same results -- it's useful to confirm everything is working.
        stats = OrderedDict(
            [
                ("filepath_adv", adv_fp),
                ("samples", original.size),
                # ("l1", LNorm(1).analyse(d)), TODO
                ("l1_delta", lnorm(delta, norm=1)),
                ("l2_delta", lnorm(delta, norm=2)),
                ("linf_delta", lnorm(delta, norm=np.inf)),
                ("snr_energy_plain", snr_energy(adv, original)),
                ("snr_energy_A", snr_energy(a_adv, a_original)),
                ("snr_energy_ITU", snr_energy(itu_adv, itu_original)),
                ("snr_energy_db_plain", snr_energy_db(adv, original)),
                ("snr_energy_db_A", snr_energy_db(a_adv, a_original)),
                ("snr_energy_db_ITU", snr_energy(itu_adv, itu_original)),
                ("snr_power_plain", snr_power(adv, original)),
                ("snr_power_A", snr_power(a_adv, a_original)),
                ("snr_power_ITU", snr_power(itu_adv, itu_original)),
                ("snr_power_db_plain", snr_power_db(adv, original)),
                ("snr_power_db_A", snr_power_db(a_adv, a_original)),
                ("snr_power_db_ITU", snr_power_db(itu_adv, itu_original)),
                ("snr_seg_plain", snr_segmented(adv, original)),
                ("snr_seg_A", snr_segmented(a_adv, a_original)),
                ("snr_seg_ITU", snr_segmented(itu_adv, itu_original)),
                ("rms_amp_db_delta_plain", rms_amplitude_db(delta)),
                ("rms_amp_db_delta_A", rms_amplitude_db(a_delta)),
                ("rms_amp_db_delta_ITU", rms_amplitude_db(itu_delta)),
                ("rms_amp_db_adv_plain", rms_amplitude_db(adv)),
                ("rms_amp_db_adv_A", rms_amplitude_db(a_adv)),
                ("rms_amp_db_adv_ITU", rms_amplitude_db(itu_adv)),
                ("rms_amp_db_orig_plain", rms_amplitude_db(original)),
                ("rms_amp_db_orig_A", rms_amplitude_db(a_original)),
                ("rms_amp_db_orig_ITU", rms_amplitude_db(itu_original)),
                ("rms_amp_raw_delta_plain", rms_amplitude_db(delta)),
                ("rms_amp_raw_delta_A", rms_amplitude_db(a_delta)),
                ("rms_amp_raw_delta_ITU", rms_amplitude_db(itu_delta)),
                ("rms_amp_raw_adv_plain", rms_amplitude_db(adv)),
                ("rms_amp_raw_adv_A", rms_amplitude_db(a_adv)),
                ("rms_amp_raw_adv_ITU", rms_amplitude_db(itu_adv)),
                ("rms_amp_raw_orig_plain", rms_amplitude_db(original)),
                ("rms_amp_raw_orig_A", rms_amplitude_db(a_original)),
                ("rms_amp_raw_orig_ITU", rms_amplitude_db(itu_original)),
                ("energy_db_delta_plain", energy_db(delta)),
                ("energy_db_delta_A", energy_db(a_delta)),
                ("energy_db_delta_ITU", energy_db(itu_delta)),
                ("energy_db_adv_plain", energy_db(adv)),
                ("energy_db_adv_A", energy_db(a_adv)),
                ("energy_db_adv_ITU", energy_db(itu_adv)),
                ("energy_db_orig_plain", energy_db(original)),
                ("energy_db_orig_A", energy_db(a_original)),
                ("energy_db_orig_ITU", energy_db(itu_original)),
                ("energy_raw_delta_plain", energy_db(delta)),
                ("energy_raw_delta_A", energy_db(a_delta)),
                ("energy_raw_delta_ITU", energy_db(itu_delta)),
                ("energy_raw_adv_plain", energy_db(adv)),
                ("energy_raw_adv_A", energy_db(a_adv)),
                ("energy_raw_adv_ITU", energy_db(itu_adv)),
                ("energy_raw_orig_plain", energy_db(original)),
                ("energy_raw_orig_A", energy_db(a_original)),
                ("energy_raw_orig_ITU", energy_db(itu_original)),
                ("power_db_delta_plain", power_db(delta)),
                ("power_db_delta_A", power_db(a_delta)),
                ("power_db_delta_ITU", power_db(itu_delta)),
                ("power_db_adv_plain", power_db(adv)),
                ("power_db_adv_A", power_db(a_adv)),
                ("power_db_adv_ITU", power_db(itu_adv)),
                ("power_db_orig_plain", power_db(original)),
                ("power_db_orig_A", power_db(a_original)),
                ("power_db_orig_ITU", power_db(itu_original)),
                ("power_raw_delta_plain", power_db(delta)),
                ("power_raw_delta_A", power_db(a_delta)),
                ("power_raw_delta_ITU", power_db(itu_delta)),
                ("power_raw_adv_plain", power_db(adv)),
                ("power_raw_adv_A", power_db(a_adv)),
                ("power_raw_adv_ITU", power_db(itu_adv)),
                ("power_raw_orig_plain", power_db(original)),
                ("power_raw_orig_A", power_db(a_original)),
                ("power_raw_orig_ITU", power_db(itu_original))
            ]
        )

        headers = ",".join([x for x in stats.keys()])
        if not os.path.exists(stats_out_filepath):
            with open(stats_out_filepath, 'a+') as f:
                f.write(headers + '\n')
            log("Wrote headers.", wrap=False)

        values = ",".join([type_check_str(x) for x in stats.values()])
        with open(stats_out_filepath, 'a+') as f:
            f.write(values + '\n')

        s = "\rWrote statistics for {a} of {b}.".format(a=i+1, b=len(deltas))
        sys.stdout.write(s)
        sys.stdout.flush()

    log("\nFinished writing statistics.", wrap=False)


if __name__ == '__main__':
    args = sys.argv[1]
    batch_generate_statistic_file(args)

