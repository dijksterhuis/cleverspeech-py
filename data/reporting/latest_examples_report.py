import os
import sys
import csv
import json
import random
import itertools
import numpy as np
import pandas as pd
import progressbar as pbar
import multiprocessing as mp

from collections import OrderedDict
from cleverspeech.utils.Utils import log, l_map

from cleverspeech.dsp import metrics
from cleverspeech.dsp import weighting_curves

from deepspeech import Model


def get_fps(indir):
    for f in os.listdir(indir):
        yield os.path.join(indir, f)


def start_deepspeech_package_model():

    # TODO: Switch to environment variable rather than relative directory path?
    module_file_path = os.path.abspath(os.path.dirname(__file__))

    deepspeech_files_path = os.path.join(
        module_file_path, "../../models/__DeepSpeech_v0_9_3/data/models/"
    )
    model_chkpoint_file_path = os.path.join(
        deepspeech_files_path, "deepspeech-0.9.3-models.pbmm"
    )
    scorer_file_path = os.path.join(
        deepspeech_files_path, "deepspeech-0.9.3-models.scorer"
    )

    ds = Model(model_chkpoint_file_path)
    ds.setBeamWidth(500)
    ds.enableExternalScorer(scorer_file_path)

    return ds


def get_adv_audios_for_ds_client(audio):
    return np.asarray(audio["advs"]["none"] * 2**15, np.int16)


def get_original_audios_for_ds_client(audio):
    return np.asarray(audio["audio"]["none"] * 2**15, np.int16)


def get_ds_client_decodes(audio):
    ds = start_deepspeech_package_model()
    val = {k: ds.stt(audio[k]["none"]) for k in audio.keys()}
    return val


def fix_padding(delta, original, advex):

    padding = delta.size - original.size
    if padding > 0:
        delta = delta[0:original.size]
        advex = advex[0:original.size]
    else:
        pass

    return delta, original, advex


def normalise(array):
    if max(array) > 1.0 or min(array) < -1.0:
        array[array > 0] = array[array > 0] / (2 ** 15 - 1)
        array[array < 0] = array[array < 0] / (2 ** 15)
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

    metadata = OrderedDict(
        [
            ("created", created_time),
            ("modified", modified_time),
            ("sample", os.path.basename(json_file_path).rstrip(".json")),
        ]
    )
    return metadata


def get_misc_data(json_file_path):

    with open(json_file_path, "r") as in_f:
        data = json.load(in_f)[0]

    decodings = OrderedDict(
        [
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
                    ("initial", data["initial_taus"][0]),
                ]
            )
             ),
            ("decode", decodings),
        ]
    )

    return misc_data


def load_settings_file(indir):
    with open(os.path.join(indir, "settings.json"), "r") as in_f:
        data = json.load(in_f)[0]
    return {k: v[0] for k, v in data.items()}


def preprocess_audio_data(audio_as_list: list):

    # make sure we're using numpy arrays
    aud = np.asarray(audio_as_list, dtype=np.float32)

    # delta and adv are both padded to the maximum length of an example in
    # batch -- so we resize them back to equal the original.
    # delta, original, adv = fix_padding(delta, original.astype(np.float32), adv)

    # We need to normalise to the 32-bit float range.
    aud = normalise(aud)

    # Apply noise weightings
    a_w = weighting_curves.a_weighting(aud)
    itu = weighting_curves.itu_weighting(aud)

    return {
        "none": aud,
        "a": a_w,
        "itu": itu
    }


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
                _flatten_into(into, enumerate(value), p_key)
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

    with open(out_file_path, 'a+') as outfile:
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


def get_audio_data(json_file_path):

    with open(json_file_path, "r") as in_f:
        data = json.load(in_f)[0]

    audios = {
        k: preprocess_audio_data(data[k]) for k in ["audio", "advs", "deltas"]
    }

    return audios


def get_snr_stats(audios: dict):

    snr_analysis_fns = OrderedDict(
        [
            ("snr_energy_db", metrics.snr_energy_db),
            # ("snr_energy", metrics.snr_energy),
            ("snr_pow_db", metrics.snr_power_db),
            # ("snr_pow", metrics.snr_power),
            ("snr_seg_db", metrics.snr_segmented),
            ("snr_loudness_k_db", metrics.snr_loudness_k_weighted_db),
            ("snr_loudness_deman_db", metrics.snr_loudness_deman_db),
            ("snr_rms_amplitude_db", metrics.snr_rms_amplitude),
        ]
    )

    snr_stats = OrderedDict(
        [
            (
                snr_key,
                OrderedDict(
                    l_map(
                        lambda w: (w, snr_fn(audios["deltas"][w], audios["audio"][w])),
                        audios["deltas"].keys()
                    )
                )
             ) for snr_key, snr_fn in snr_analysis_fns.items()
        ]
    )
    return snr_stats


def get_lnorm_stats(audios: dict):

    l_norm_keys = [
        ("l1", 1),
        ("l2", 2),
        ("linf", np.inf),
    ]
    fnames = ["deltas", "audio", "advs"]
    weights = ["itu", "a", "none"]

    prod = itertools.product(l_norm_keys, fnames, weights)
    res = {
        "{l_s}.{f}.{w}".format(l_s=l_s, f=f, w=w):
            metrics.lnorm(audios[f][w], norm=l_t) for (l_s, l_t), f, w in prod
    }

    return OrderedDict(res)


def get_dsp_stats(audios: dict):

    dsp_analysis_fns = OrderedDict(
        [
            ("rms_amp_db", metrics.rms_amplitude_db),
            # ("rms_amp", metrics.rms_amplitude),
            ("energy_db", metrics.energy_db),
            # ("energy", metrics.energy),
            ("power_db", metrics.power_db),
            # ("power", metrics.power),
            ("k_loudness_db", metrics.loudness_k_weighted_db),
            ("deman_loudness_db", metrics.loudness_deman_db),
            ("crest_factor_db", metrics.crest_factor_db),
            ("thdn_db", metrics.thdn_db),
        ]
    )

    dsp_stats = OrderedDict(
        [
            (dsp_key, OrderedDict(
                [
                    (weight, OrderedDict(
                        [
                            (audio_file, dsp_fn(audios[audio_file][weight]))
                            for audio_file in ["deltas", "audio", "advs"]
                        ]
                    )) for weight in audios["deltas"].keys()
                ]
            )) for dsp_key, dsp_fn in dsp_analysis_fns.items()
        ]
    )
    return dsp_stats


def coalesce(*args):
    r = {}
    for arg in args:
        r.update(flatten_dict(arg))
    return OrderedDict(r)


def main():

    if len(sys.argv[1:]) == 1:
        indir = outdir = sys.argv[1]

    else:
        indir, outdir = sys.argv[1:]

    example_json_results_file_paths = [
        fp for fp in get_fps(indir) if "sample" in fp and ".json" in fp
    ]
    s = "Found {n} results files in directory: {d}".format(
        n=len(example_json_results_file_paths), d=indir
    )
    log(s)

    # ds = start_deepspeech_package_model()

    settings = load_settings_file(indir)

    unique_hash = random.getrandbits(128)

    with mp.Pool(mp.cpu_count() - 1) as p, pbar.ProgressBar(max_value=10) as bar:

        audios = p.map(get_audio_data, example_json_results_file_paths)
        bar.update(1)

        metadata = p.map(get_file_metadata, example_json_results_file_paths)
        bar.update(1)

        misc = p.map(get_misc_data, example_json_results_file_paths)
        bar.update(1)

        # client doesn't work for float32 examples and rounding up/down to int16
        # breaks the attack... leave this here as reference in case it's ever
        # dealt with

        # advs_only = p.map(get_adv_audios_for_ds_client, audios)
        # auds_only = p.map(get_original_audios_for_ds_client, audios)

        # client_decodes = l_map(
        #     lambda x: {
        #         "clean_client_decode": ds.stt(x[0]),
        #         "adv_client_decode": ds.stt(x[1])
        #     },
        #     zip(advs_only, auds_only)
        # )
        # bar.update(1)

        snr_stats = p.map(get_snr_stats, audios)
        bar.update(1)

        lnorm_stats = p.map(get_lnorm_stats, audios)
        bar.update(1)

        dsp_stats = p.map(get_dsp_stats, audios)
        bar.update(1)

        combined = p.starmap(
            coalesce,
            zip(
                l_map(lambda _: settings, misc),
                l_map(lambda _: {"results_hash": unique_hash}, misc),
                metadata,
                misc,
                lnorm_stats,
                snr_stats,
                dsp_stats
            )
        )
        base = pd.DataFrame.from_records(combined)
        bar.update(1)

        base.columns = [".".join(h) for h in list(base.columns.values)]
        csv_out_file_path = os.path.join(outdir, "stats.csv")
        headers = not os.path.exists(csv_out_file_path)

        base.to_csv(
            csv_out_file_path, mode="a+", header=headers
        )
        bar.update(1)

    log("Done processing {}".format(indir))


if __name__ == '__main__':
    main()
