import os
import sys
import csv
import json

import numpy as np

from collections import OrderedDict
from cleverspeech.utils.Utils import log
from cleverspeech.utils import WavFile

from cleverspeech.data.egress.eval import DetectionMetrics
from cleverspeech.data.egress.eval import NoiseWeightings

from deepspeech import Model


def get_fps(indir):
    for f in os.listdir(indir):
        yield os.path.join(indir, f)


def start_deepspeech_package_model():
    beam_width = 500
    lm_alpha = 0.75
    lm_beta = 1.85
    n_features = 26
    n_context = 9

    # TODO: Need to include output_graph.pb in the docker image builds

    cleverspeech_path = "/home/dijksterhuis/0-dox/1-dev/3-attacks/cleverSpeech/"
    deepspeech_files_path = os.path.join(
        cleverspeech_path, "models/DeepSpeech_v041/data/models/"
    )

    alphabet_file_path = os.path.join(
        deepspeech_files_path, "alphabet.txt"
    )
    model_chkpoint_file_path = os.path.join(
        deepspeech_files_path, "output_graph.pb"
    )
    lm_file_path = os.path.join(
        deepspeech_files_path, "lm.binary"
    )
    trie_file_path  = os.path.join(
        deepspeech_files_path, "trie"
    )

    ds = Model(
        model_chkpoint_file_path,
        n_features,
        n_context,
        alphabet_file_path,
        beam_width
    )

    ds.enableDecoderWithLM(
        alphabet_file_path,
        lm_file_path,
        trie_file_path,
        lm_alpha,
        lm_beta
    )

    return ds


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


def get_file_metadata(json_file_path):

    created_time = os.path.getctime(os.path.abspath(json_file_path))
    modified_time = os.path.getmtime(os.path.abspath(json_file_path))

    directory_split = os.path.abspath(json_file_path).split("/")
    result_dir_index = directory_split.index('adv')

    exp = directory_split[result_dir_index+1]
    params = directory_split[result_dir_index + 2: -1]

    hyper_params = {
        "param_{}".format(idx): param for idx, param in enumerate(params)
    }

    metadata = {
        "exp": exp,
        "hyperparams": hyper_params,
        "created": created_time,
        "modified": modified_time,
        "sample": os.path.basename(json_file_path).rstrip(".json")
    }
    return metadata


def load_settings_file(indir):
    with open(os.path.join(indir, "settings.json"), "r") as in_f:
        return json.load(in_f)[0]


def preprocess_audio_data(delta, original, adv):

    # make sure we're using numpy arrays
    delta = np.asarray(delta, dtype=np.float32)
    original = np.asarray(original, dtype=np.float32)
    adv = np.asarray(adv, dtype=np.float32)

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

    # return *everything* as 16 bit int arrays
    outs = {
        "deltas": {
            "itu": itu_delta.astype(np.int16),
            "a": a_delta.astype(np.int16),
            "none": delta.astype(np.int16),
        },
        "advs": {
            "itu": itu_adv.astype(np.int16),
            "a": a_adv.astype(np.int16),
            "none": adv.astype(np.int16),
        },
        "originals": {
            "itu": itu_original.astype(np.int16),
            "a": a_original.astype(np.int16),
            "none": original.astype(np.int16),
        },
    }

    weights = ["itu", "a", "none"]
    audio_file_names = ["deltas", "advs", "originals"]

    return outs, weights, audio_file_names


def flatten_dict(d: dict) -> dict:
    """
    Flatten hierarchical dicts into a dict of path tuples -> deep values.

    https://stackoverflow.com/a/66784565/5945794
    """
    out = {}

    def _flatten_into(into, pairs, prefix=()):
        for key, value in pairs:
            p_key = prefix + (key,)
            if isinstance(value, list):
                print(value)
                _flatten_into(into, enumerate(list), p_key)
            elif isinstance(value, dict):
                _flatten_into(into, value.items(), p_key)
            else:
                out[p_key] = value

    _flatten_into(out, d.items())
    return out


def write_to_csv(stats_data, out_file_path):

    flat_row = flatten_dict(stats_data)

    headers = [".".join(map(str, key)) for key in flat_row.keys()]
    row_data = [str(flat_row.get(key, "")) for key in flat_row.keys()]

    if not os.path.exists(out_file_path):
        write_headers = True
    else:
        write_headers = False

    with open(out_file_path, 'w+') as outfile:
        writer = csv.writer(outfile)

        if write_headers:
            writer.writerow(headers)
            log("Wrote headers.", wrap=False)

        writer.writerow(row_data)

    return True


def write_to_json(stats_data, out_file_path):
    with open(out_file_path, "w+") as outfile:
        json.dump([stats_data], outfile, indent=2)

    return True


def generate_stats_file(indir):

    stats_out_filepath = os.path.join(indir, "stats.csv")

    example_json_results_file_paths = [
        fp for fp in get_fps(indir) if "sample" in fp and ".json" in fp
    ]
    s = "Found {n} results files in director {d}... Processing now.".format(
        n=len(example_json_results_file_paths), d=indir
    )
    log(s)

    ds = start_deepspeech_package_model()

    # settings = load_settings_file(indir)

    for idx, json_file_path in enumerate(example_json_results_file_paths):

        # ==== EXTRACT

        metadata = get_file_metadata(json_file_path)

        with open(json_file_path, "r") as in_f:
            data = json.load(in_f)[0]

        # ==== TRANSFORM

        audio_data, weights, audio_file_names = preprocess_audio_data(
            data["deltas"],
            data["audio"],
            data["advs"],
        )

        client_decodings = {
            k: ds.stt(audio_data[k]["none"], 16000).replace(" ", "=") for k in audio_file_names
        }

        decodings = {
            "client": client_decodings,
            "cleverspeech": {
                "decoding": data["decodings"][0],
                "target": data["phrases"][0],
                "log_probs": data["probs"][0],
            }
        }

        misc_data = {
            "step": data["step"][0],
            "loss": data["total_loss"][0],
            "n_samples": data["n_samples"][0],
            "real_feats": data["real_feats"][0],
            "bounds": {
                "raw": data["bounds_raw"][0],
                "eps": data["bounds_eps"][0],
                "initial": data["initial_taus"][0],
            },
            "distances": {
                "raw": data["distances_raw"][0],
                "eps": data["distances_eps"][0],
            },
            "decode": decodings

        }

        def calc_lnorm(d, norm_int, weight):
            for audio_file in audio_file_names:

                value = DetectionMetrics.lnorm(
                    d[audio_file][weight],
                    norm=norm_int
                )
                yield audio_file, value

        lnorm_keys = {
            "l1": 1,
            "l2": 2,
            "linf": np.inf,
        }

        l_norms = {
            norm_str:
                {
                    weight: {
                        file_k: v for file_k, v in calc_lnorm(audio_data, norm_int, weight)
                    } for weight in weights
                } for norm_str, norm_int in lnorm_keys.items()
        }

        snr_analysis_fns = {
            "snr_energy_db": DetectionMetrics.snr_energy_db,
            "snr_energy": DetectionMetrics.snr_energy,
            "snr_pow_db": DetectionMetrics.snr_power_db,
            "snr_pow": DetectionMetrics.snr_power,
            "snr_seg_db": DetectionMetrics.snr_segmented,
        }

        snr_stats = {
            snr_key: {
                weight: snr_fn(
                    audio_data["deltas"][weight], audio_data["originals"][weight]
                ) for weight in weights
            } for snr_key, snr_fn in snr_analysis_fns.items()
        }

        dsp_analysis_fns = {
            "rms_amp_db": DetectionMetrics.rms_amplitude_db,
            "rms_amp": DetectionMetrics.rms_amplitude,
            "energy_db": DetectionMetrics.energy_db,
            "energy": DetectionMetrics.energy,
            "power_db": DetectionMetrics.power_db,
            "power": DetectionMetrics.power,
        }

        dsp_stats = {
            dsp_key: {
                weight: {
                    audio_file : dsp_fn(audio_data[audio_file][weight]) for audio_file in audio_file_names
                } for weight in weights
            } for dsp_key, dsp_fn in dsp_analysis_fns.items()
        }

        stats = metadata
        stats.update(misc_data)
        stats.update(l_norms)
        stats.update(snr_stats)
        stats.update(dsp_stats)

        # ==== LOAD

        write_to_csv(stats, stats_out_filepath)

        s = "\rWrote statistics for {f_in} to {f_out} | {a} of {b}.".format(
            f_in=json_file_path,
            f_out=stats_out_filepath,
            a=idx + 1,
            b=len(example_json_results_file_paths)
        )
        sys.stdout.write(s)
        sys.stdout.flush()


if __name__ == '__main__':
    args = sys.argv[1]
    generate_stats_file(args)

