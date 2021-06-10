import json
import boto3

import numpy as np

from bz2 import compress as bz2_compress
from os import path, makedirs
from cleverspeech.utils import WavFile


def convert_types_for_json(results):
    data = {}

    for k in results.keys():

        values = results[k]

        if type(values) in [np.float32, np.int32, np.int16, np.int64]:
            values = float(values)

        if type(values) is np.ndarray:
            values = values.tolist()

        if type(values) is list:
            if type(values[0]) is np.ndarray:
                for idx, value in enumerate(values):
                    values[idx] = value.tolist()

            if type(values[0]) in [np.float32, np.int32, np.int16, np.int64]:
                values = [float(v) for v in values]

        if type(values) is not list:
            values = [values]

        data[k] = values

    return data


def add_json_prefix_and_postfix(data, prefix="[\n", postfix="\n]"):
    return prefix + data + postfix


def prepare_json_data(data, indent=2):

    data = convert_types_for_json(data)
    json_data = json.dumps(
        data,
        indent=indent,
        sort_keys=True,
        ensure_ascii=True
    )

    return add_json_prefix_and_postfix(json_data)


def make_per_bound_path(outdir, bound_eps):
    bound_dir = "eps_{}".format(bound_eps)
    return path.join(outdir, bound_dir)


def safe_make_dirs(outdir):
    if not path.exists(outdir):
        makedirs(outdir, exist_ok=True)


def s3_pseudo_make_dirs(bucket, file_path):

    s3 = boto3.resource('s3')
    s3_bucket = s3.Bucket(bucket)
    return s3_bucket.Object(file_path)


def write_latest_audio_to_local_wav_file(outdir, data, bit_depth=16, sample_rate=16000):

    for wav_file in ["audio", "deltas", "advs"]:

        file_name = data['basenames'].rstrip(".wav")
        file_name = file_name + "_{}".format(wav_file)

        file_path = path.join(outdir, file_name)

        WavFile.write(
            file_path,
            data,
            sample_rate=sample_rate,
            bit_depth=bit_depth
        )


def write_settings_to_local_json_file(outdir, data):
    file_name = "settings.json"
    file_path = path.join(outdir, file_name)

    json_data = prepare_json_data(data)
    safe_make_dirs(outdir)

    with open(file_path, mode='w+', encoding='utf-8') as f:
        f.write(json_data)


def write_settings_to_s3(outdir, data):

    bucket = "cleverspeech-results"

    file_path = "settings.json"
    file_path = file_path.lstrip("./")
    file_path = path.join(outdir, file_path)

    json_data = prepare_json_data(data, indent=0)

    # we need to create the local directory for log files
    safe_make_dirs(outdir)

    s3_object = s3_pseudo_make_dirs(bucket, file_path)
    s3_object.put(
        Body=json_data,
        Tagging="costing:cleverSpeech"
    )


def write_latest_metadata_to_local_json_file(outdir, data):

    file_name = data['basenames'].rstrip(".wav") + ".json"
    file_path = path.join(outdir, file_name)

    json_data = prepare_json_data(data)
    safe_make_dirs(outdir)

    with open(file_path, mode='w+', encoding='utf-8') as f:
        f.write(json_data)


def write_per_bound_metadata_to_local_json_files(outdir, data):

    outdir = make_per_bound_path(outdir, data['step'])
    file_name = data['basenames'].rstrip(".wav") + ".json"
    file_path = path.join(outdir, file_name)

    json_data = prepare_json_data(data)
    safe_make_dirs(outdir)

    with open(file_path, mode='w+', encoding='utf-8') as f:
        f.write(json_data)


def write_latest_metadata_to_s3(outdir, data):

    bucket = "cleverspeech-results"

    file_path = data['basenames'].rstrip(".wav")
    file_path = file_path + ".json.bz2"
    file_path = file_path.lstrip("./")
    file_path = path.join(outdir, file_path)

    json_data = prepare_json_data(data, indent=0)

    # bz2 compression can reduce file size up to 4x, helpful when charged
    # for put requests... to decompress: bz2.decompress(compressed_data)

    compressed_json_data = bz2_compress(json_data.encode("ascii"))

    s3_object = s3_pseudo_make_dirs(bucket, file_path)
    s3_object.put(
        Body=compressed_json_data,
        Tagging="costing:cleverSpeech"
    )


def write_all_metadata_to_s3(outdir, data):

    bucket = "cleverspeech-results"

    outdir = make_per_bound_path(outdir, data['step'])

    file_path = data['basenames'].rstrip(".wav")
    file_path = file_path + ".json.bz2"
    file_path = file_path.lstrip("./")
    file_path = path.join(outdir, file_path)

    json_data = prepare_json_data(data, indent=0)

    # bz2 compression can reduce file size up to 4x, helpful when charged
    # for put requests... to decompress: bz2.decompress(compressed_data)

    compressed_json_data = bz2_compress(json_data.encode("ascii"))

    s3_object = s3_pseudo_make_dirs(bucket, file_path)
    s3_object.put(
        Body=compressed_json_data,
        Tagging="costing:cleverSpeech"
    )
