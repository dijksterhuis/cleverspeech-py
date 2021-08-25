import os
import sys
import csv
import json
import pandas as pd
import numpy as np

from collections import OrderedDict
from cleverspeech.utils.Utils import log

from cleverspeech.data.egress.metrics import DetectionMetrics
from cleverspeech.data.egress.metrics import NoiseWeightings

from deepspeech import Model



def generate_stats_file(indir, outdir):

    stats_out_filepath = os.path.join(outdir, "stats.csv")

    example_json_results_file_paths = [
        fp for fp in get_fps(indir) if "sample" in fp and ".json" in fp
    ]
    s = "Found {n} results files in director {d}... Processing now.".format(
        n=len(example_json_results_file_paths), d=indir
    )
    log(s)

    # ds = start_deepspeech_package_model()

    # settings = load_settings_file(indir)

    for idx, json_file_path in enumerate(example_json_results_file_paths):

        # ==== EXTRACT

        metadata = get_file_metadata(json_file_path)

        with open(json_file_path, "r") as in_f:
            data = json.load(in_f)[0]

        log("Loaded: {}".format(json_file_path), wrap=False, timings=True)

        # ==== TRANSFORM

        audio_data, weights, audio_file_names = preprocess_audio_data(
            data["deltas"],
            data["audio"],
            data["advs"],
        )

        # client_decodings = OrderedDict(
        #     [
        #         (k, ds.stt(audio_data[k]["none"], 16000).replace(" ", "=")) for k in audio_file_names
        #     ]
        # )

        decodings = OrderedDict(
            [
                # ("client", client_decodings),
                ("cleverspeech",
                    OrderedDict(
                        [
                            ("decoding", data["decodings"][0]),
                            ("target", data["phrases"][0]),
                            ("log_probs", data["probs"][0]),
                        ]
                    )
                )
            ]
        )

        misc_data = OrderedDict(
            [
                ("step", data["step"][0]),
                ("loss", data["total_loss"][0]),
                ("n_samples", data["n_samples"][0]),
                ("real_feats", data["real_feats"][0]),
                ("bounds", OrderedDict(
                        [
                            ("raw", data["bounds_raw"][0]),
                            # ("eps", data["bounds_eps"][0]),
                            ("initital", data["initial_taus"][0]),
                        ]
                    )
                 ),
                # ("distances", OrderedDict(
                #     [
                #         ("raw", data["distances_raw"][0]),
                #         # ("eps", data["distances_eps"][0]),
                #     ]
                # )
                #  ),
                ("decode", decodings),
            ]
        )

        def calc_lnorm(d, norm_int, weight):
            for audio_file in audio_file_names:

                value = DetectionMetrics.lnorm(
                    d[audio_file][weight],
                    norm=norm_int
                )
                yield audio_file, value

        lnorm_keys = OrderedDict([
            ("l1", 1),
             ("l2", 2),
             ("linf", np.inf),
        ])

        l_norms = OrderedDict([
            (norm_str,
                OrderedDict([
                    (weight,
                        OrderedDict([
                            (file_k, v) for file_k, v in calc_lnorm(
                                audio_data,
                                norm_int,
                                weight
                            )
                        ])
                     ) for weight in weights
                ])
            ) for norm_str, norm_int in lnorm_keys.items()
        ])

        snr_analysis_fns = OrderedDict([
            ("snr_energy_db", DetectionMetrics.snr_energy_db),
            # ("snr_energy", DetectionMetrics.snr_energy),
            ("snr_pow_db", DetectionMetrics.snr_power_db),
            # ("snr_pow", DetectionMetrics.snr_power),
            ("snr_seg_db", DetectionMetrics.snr_segmented),
        ])

        snr_stats = OrderedDict([
            (snr_key,
                OrderedDict([
                    (
                        weight,
                        snr_fn(
                            audio_data["deltas"][weight],
                            audio_data["originals"][weight]
                        )
                    ) for weight in weights
                ])
             ) for snr_key, snr_fn in snr_analysis_fns.items()
        ])

        dsp_analysis_fns = OrderedDict([
            ("rms_amp_db", DetectionMetrics.rms_amplitude_db),
            # ("rms_amp", DetectionMetrics.rms_amplitude),
            ("energy_db", DetectionMetrics.energy_db),
            # ("energy", DetectionMetrics.energy),
            ("power_db", DetectionMetrics.power_db),
            # ("power", DetectionMetrics.power),
        ])

        dsp_stats = OrderedDict([
            (dsp_key, OrderedDict([
                (weight, OrderedDict([
                    (audio_file, dsp_fn(audio_data[audio_file][weight])) for audio_file in audio_file_names
                ])) for weight in weights
            ])) for dsp_key, dsp_fn in dsp_analysis_fns.items()
        ])

        stats = metadata
        stats.update(misc_data)
        stats.update(l_norms)
        stats.update(snr_stats)
        stats.update(dsp_stats)

        # ==== LOAD

        write_to_csv(stats, stats_out_filepath)

        s = "Wrote statistics for {f_in} to {f_out} | {a} of {b}.".format(
            f_in=json_file_path,
            f_out=stats_out_filepath,
            a=idx + 1,
            b=len(example_json_results_file_paths)
        )
        log(s, wrap=False, timings=True)


if __name__ == '__main__':
    file_path = sys.argv[1]

    csv_data = pd.read_csv(file_path)

    print(csv_data.mean())

