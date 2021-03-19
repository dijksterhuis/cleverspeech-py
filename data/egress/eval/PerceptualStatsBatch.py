import os
import sys

import numpy as np

from collections import OrderedDict
from cleverspeech.utils.Utils import log
from cleverspeech.utils import WavFile

from cleverspeech.data.egress.eval import DetectionMetrics
from cleverspeech.data.egress.eval import NoiseWeightings


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

    deltas = [WavFile.load(fp, "float32") for fp in deltas_file_paths]
    advexs = [WavFile.load(fp, "float32") for fp in advexs_file_paths]
    origs = [WavFile.load(fp, "float32") for fp in origs_file_paths]

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
        a_delta = NoiseWeightings.a_weighting(delta)
        a_original = NoiseWeightings.a_weighting(original)
        a_adv = NoiseWeightings.a_weighting(adv)
        itu_delta = NoiseWeightings.itu_weighting(delta)
        itu_original = NoiseWeightings.itu_weighting(original)
        itu_adv = NoiseWeightings.itu_weighting(adv)

        # Generate the actual stats. Note that many of these stats actually give
        # use the same results -- it's useful to confirm everything is working.
        stats = OrderedDict(
            [
                # ==> misc metadata
                ("raw_model_success", True),
                ("filepath_adv", adv_fp),
                ("samples", original.size),
                # ==> Lp norm metrics
                # ("l1", LNorm(1).analyse(d)), TODO
                ("l1_delta", DetectionMetrics.lnorm(delta, norm=1)),
                ("l2_delta", DetectionMetrics.lnorm(delta, norm=2)),
                ("linf_delta", DetectionMetrics.lnorm(delta, norm=np.inf)),
                # ==> snr from the energy in DBfs
                ("snr_energy_db_plain", DetectionMetrics.snr_energy_db(delta, original)),
                ("snr_energy_db_A", DetectionMetrics.snr_energy_db(a_delta, a_original)),
                ("snr_energy_db_ITU", DetectionMetrics.snr_energy_db(itu_delta, itu_original)),
                # ==> energy snr from raw 16 bit int values
                ("snr_energy_plain", DetectionMetrics.snr_energy(delta, original)),
                ("snr_energy_A", DetectionMetrics.snr_energy(a_delta, a_original)),
                ("snr_energy_ITU", DetectionMetrics.snr_energy(itu_delta, itu_original)),
                # ==> power snr in DBfs
                ("snr_power_db_plain", DetectionMetrics.snr_power_db(delta, original)),
                ("snr_power_db_A", DetectionMetrics.snr_power_db(a_delta, a_original)),
                ("snr_power_db_ITU", DetectionMetrics.snr_power_db(itu_delta, itu_original)),
                # ==> power snr from raw 16 bit int values
                ("snr_power_plain", DetectionMetrics.snr_power(delta, original)),
                ("snr_power_A", DetectionMetrics.snr_power(a_delta, a_original)),
                ("snr_power_ITU",DetectionMetrics.snr_power(itu_delta, itu_original)),
                # ==> segmented snr from lea's paper
                ("snr_seg_plain", DetectionMetrics.snr_segmented(delta, original)),
                ("snr_seg_A", DetectionMetrics.snr_segmented(a_delta, a_original)),
                ("snr_seg_ITU", DetectionMetrics.snr_segmented(itu_delta, itu_original)),
                # ==> RMS Amplitude in DBfs
                ("rms_amp_db_delta_plain", DetectionMetrics.rms_amplitude_db(delta)),
                ("rms_amp_db_delta_A", DetectionMetrics.rms_amplitude_db(a_delta)),
                ("rms_amp_db_delta_ITU", DetectionMetrics.rms_amplitude_db(itu_delta)),
                ("rms_amp_db_adv_plain", DetectionMetrics.rms_amplitude_db(adv)),
                ("rms_amp_db_adv_A", DetectionMetrics.rms_amplitude_db(a_adv)),
                ("rms_amp_db_adv_ITU", DetectionMetrics.rms_amplitude_db(itu_adv)),
                ("rms_amp_db_orig_plain", DetectionMetrics.rms_amplitude_db(original)),
                ("rms_amp_db_orig_A", DetectionMetrics.rms_amplitude_db(a_original)),
                ("rms_amp_db_orig_ITU", DetectionMetrics.rms_amplitude_db(itu_original)),
                # ==> RMS Amplitude from raw 16 ints
                ("rms_amp_raw_delta_plain", DetectionMetrics.rms_amplitude_db(delta)),
                ("rms_amp_raw_delta_A", DetectionMetrics.rms_amplitude_db(a_delta)),
                ("rms_amp_raw_delta_ITU", DetectionMetrics.rms_amplitude_db(itu_delta)),
                ("rms_amp_raw_adv_plain", DetectionMetrics.rms_amplitude_db(adv)),
                ("rms_amp_raw_adv_A", DetectionMetrics.rms_amplitude_db(a_adv)),
                ("rms_amp_raw_adv_ITU", DetectionMetrics.rms_amplitude_db(itu_adv)),
                ("rms_amp_raw_orig_plain", DetectionMetrics.rms_amplitude_db(original)),
                ("rms_amp_raw_orig_A", DetectionMetrics.rms_amplitude_db(a_original)),
                ("rms_amp_raw_orig_ITU", DetectionMetrics.rms_amplitude_db(itu_original)),
                # ==> Energy in Decibels
                ("energy_db_delta_plain", DetectionMetrics.energy_db(delta)),
                ("energy_db_delta_A", DetectionMetrics.energy_db(a_delta)),
                ("energy_db_delta_ITU", DetectionMetrics.energy_db(itu_delta)),
                ("energy_db_adv_plain", DetectionMetrics.energy_db(adv)),
                ("energy_db_adv_A", DetectionMetrics.energy_db(a_adv)),
                ("energy_db_adv_ITU", DetectionMetrics.energy_db(itu_adv)),
                ("energy_db_orig_plain", DetectionMetrics.energy_db(original)),
                ("energy_db_orig_A", DetectionMetrics.energy_db(a_original)),
                ("energy_db_orig_ITU", DetectionMetrics.energy_db(itu_original)),
                # ==> Energy from raw 16 bit ints
                ("energy_raw_delta_plain", DetectionMetrics.energy_db(delta)),
                ("energy_raw_delta_A", DetectionMetrics.energy_db(a_delta)),
                ("energy_raw_delta_ITU", DetectionMetrics.energy_db(itu_delta)),
                ("energy_raw_adv_plain", DetectionMetrics.energy_db(adv)),
                ("energy_raw_adv_A", DetectionMetrics.energy_db(a_adv)),
                ("energy_raw_adv_ITU", DetectionMetrics.energy_db(itu_adv)),
                ("energy_raw_orig_plain", DetectionMetrics.energy_db(original)),
                ("energy_raw_orig_A", DetectionMetrics.energy_db(a_original)),
                ("energy_raw_orig_ITU", DetectionMetrics.energy_db(itu_original)),
                # ==> Signal power in DBfs
                ("power_db_delta_plain", DetectionMetrics.power_db(delta)),
                ("power_db_delta_A", DetectionMetrics.power_db(a_delta)),
                ("power_db_delta_ITU", DetectionMetrics.power_db(itu_delta)),
                ("power_db_adv_plain", DetectionMetrics.power_db(adv)),
                ("power_db_adv_A", DetectionMetrics.power_db(a_adv)),
                ("power_db_adv_ITU", DetectionMetrics.power_db(itu_adv)),
                ("power_db_orig_plain", DetectionMetrics.power_db(original)),
                ("power_db_orig_A", DetectionMetrics.power_db(a_original)),
                ("power_db_orig_ITU", DetectionMetrics.power_db(itu_original)),
                # ==> Signal power from raw 16 bit ints
                ("power_raw_delta_plain", DetectionMetrics.power_db(delta)),
                ("power_raw_delta_A", DetectionMetrics.power_db(a_delta)),
                ("power_raw_delta_ITU", DetectionMetrics.power_db(itu_delta)),
                ("power_raw_adv_plain", DetectionMetrics.power_db(adv)),
                ("power_raw_adv_A", DetectionMetrics.power_db(a_adv)),
                ("power_raw_adv_ITU", DetectionMetrics.power_db(itu_adv)),
                ("power_raw_orig_plain", DetectionMetrics.power_db(original)),
                ("power_raw_orig_A", DetectionMetrics.power_db(a_original)),
                ("power_raw_orig_ITU", DetectionMetrics.power_db(itu_original))
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

        # TODO: Generate spectrograms whilst all these audio files are loaded.
        # TODO: Run through against normal deepspeech (16-bit int wav files).

    log("\nFinished writing statistics.", wrap=False)


if __name__ == '__main__':
    args = sys.argv[1]
    batch_generate_statistic_file(args)

