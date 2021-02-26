import os
import sys
import random
import argparse
import soundfile

import numpy as np

from datetime import datetime as time
from base64 import b64encode


def load_wav(fp, dtype):
    audio, sample_rate = soundfile.read(fp, dtype=dtype)
    audio = audio.astype(np.float32)
    try:
        assert min(audio) >= -2**15 and max(audio) <= 2**15 - 1
    except AssertionError as e:
        err = "Problem with source file resolution:\n"
        err += "min(audio)={}\n".format(min(audio))
        err += "max(audio)={}\n".format(max(audio))
        err += "input must be valid 16 bit audio: -2**15 <= x <= 2**15 -1\n"
        print(err)
        raise AssertionError
    return audio


def load_wavs(fps, dtype="int16"):
    return [load_wav(f, dtype) for f in fps]


def dump_wavs(outdir, example, keys, filepath_key="filepath", sample_rate=16000):

    basename = os.path.basename(example[filepath_key]).rstrip(".wav")
    outpath = os.path.join(outdir, basename)

    for k in keys:
        dump_wav(
            outpath + "_{}.wav".format(k),
            example[k],
            sample_rate=sample_rate
        )


def dump_wav(path, data, sample_rate=16000, bit_depth=16):

    """
    We *always* want to write out as 32-bit floating point wav files.
    Writing as 16 bit signed ints destroys information and affects the
    perturbation.

    Note that we need to perform some additional work if we want to pass 32 bit
    floating point encoded wav files back into to deepspeech - it expects 16 bit
    signed ints as file inputs (although uses float32 for calculations).

    :param path: file path to write to
    :param data: the numpy array being written
    :param sample_rate: sample rate of the output file
    :param bit_depth: How many signed integer bits the input data has for normalisation.

    :return: None

    TODO: handle varying bit depths, `bit_depth` arg is currently unused.
    """
    assert type(data) is np.ndarray
    assert type(path) is str

    # always convert to float32 for writing to maintain integrity.
    if data.dtype == np.int16 or data.dtype == np.int32:
        data = data.astype(np.float32)

    # Make sure we write out files correctly for float32.
    try:
        assert -2**15 <= min(data) and max(data) <= 2**15 - 1
    except AssertionError as e:
        err = "Conversion to float32 wav output failed for {}\n".format(
            path)
        err += "min(data)={}\n".format(min(data))
        err += "max(data)={}\n".format(max(data))
        err += "Output data must respect assertion: -2^15 <= x <= 2^15 - 1 \n"
        print(err)
        raise AssertionError

    # normalise
    normalised = np.where(
        np.less(data, np_zero(data.size, np.float32)),
        data / (2 ** 15),
        data / (2 ** 15 - 1)
    )

    soundfile.write(
        path,
        normalised,
        sample_rate,
        subtype="FLOAT",
        format="WAV"
    )


def dump_b64bytes(z):
    return b64encode(z.dumps()).decode()


def assert_positive_int(x):
    assert x > 0 and type(x) is int


def assert_int(x):
    assert type(x) is int


def assert_positive_float(x):
    assert x > 0 and type(x) is float


def assert_float(x):
    assert type(x) is float


def assert_bool(x):
    assert type(x) is bool


def lcomp(v, i=None):
    if i:
        return [x[i] for x in v]
    else:
        return [x for x in v]


def l_map(f, x):
    return list(map(f, x))


def np_arr(x, t):
    return np.array(x, dtype=t)


def np_zero(x, t):
    return np.zeros(x, dtype=t)


def np_one(x, t):
    return np.ones(x, dtype=t)


def log(*args, funcs=None, wrap=True, outdir=None, stdout=True):
    s = ""
    wrapper = "-" * 30
    if args:
        s += "\n".join(args)

    if funcs:
        if callable(funcs):
            s += funcs()
        else:
            for func in funcs:
                s += func()
    if wrap:
        s += "\n" + wrapper

    if outdir is not None:
        outfile = os.path.join(outdir, "log.txt")

        s = "{}\t".format(time.now()) + s
        with open(outfile, "a+") as f:
            f.write(s + "\n")

    if stdout:
        print(s)


class Timer:
    def __init__(self):

        self.start = None
        self.end = None
        self.delta = None

    def start(self):
        self.start = time.now()

    def stop(self):
        self.end = time.now()

    def print(self):
        self.delta = self.start - self.end
        days = self.delta.days
        hours = self.delta.seconds // 3600
        minutes = self.delta.seconds // 60 % 60
        seconds = self.delta.seconds // 60

        print("Time taken -- {d}:{h}:{m}:{s}".format(
                d=days,
                h=hours,
                m=minutes,
                s=seconds
        ))


def enum(x):
    return enumerate(x)


def run_decoding_check(attack, batch):
    """
    Do an initial decoding to verify everything is working
    """
    decodings, probs = attack.victim.inference(
        batch,
        feed=attack.feeds.examples,
        decoder="batch"
    )
    z = zip(batch.audios["basenames"], probs, decodings)
    s = ["{}\t{:.3f}\t{}".format(b, p, d) for b, p, d in z]
    log("Initial decodings:", '\n'.join(s), wrap=False)

    s = ["{:.0f}".format(x) for x in batch.audios["real_feats"]]
    log("Real Features: ", "\n".join(s), wrap=False)

    s = ["{:.0f}".format(x) for x in batch.audios["ds_feats"]]
    log("DS Features: ", "\n".join(s), wrap=False)

    s = ["{:.0f}".format(x) for x in batch.audios["n_samples"]]
    log("Real Samples: ", "\n".join(s), wrap=True)


def args(experiments):

    choices = list(experiments.keys())

    standard_non_required_args = {
        "gpu_device": int,
        "batch_size": int,
        "learning_rate": float,
        "nsteps": int,
        "decode_step": int,
        "constraint_update": str,
        "rescale": float,
        "max_spawns": int,
        "audio_indir": str,
        "outdir": str,
        "targets_path": str,
        "spawn_delay": int,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        nargs=1,
        choices=choices,
    )

    for k, v in standard_non_required_args.items():
        parser.add_argument(
            "--" + k,
            nargs=1,
            type=v,
            default=[None],
            required=False
        )

    args = parser.parse_args()

    # Disable tensorflow looking at tf.app.flags.FLAGS but we get to keep
    # the args in the args variable. Otherwise we get the following exception
    # `absl.flags._exceptions.UnrecognizedFlagError: Unknown command line flag`
    # because tf detects the flags in `DeepSpeech.Utils.Config` and gets
    # confused when it finds a duplicate key like `batch_size`
    while len(sys.argv) > 1:
        sys.argv.pop()

    experiment = args.experiment[0]
    log("Running new experiment: {}".format(experiment))

    def update_master_settings(d, k, v):
        if v is not None:
            d.update({k: v})

    master_settings = {}

    # TODO -- how to automate the args.x part of this.

    update_master_settings(
        master_settings, "gpu_device", args.gpu_device[0]
    )
    update_master_settings(
        master_settings, "batch_size", args.batch_size[0]
    )
    update_master_settings(
        master_settings, "learning_rate", args.learning_rate[0]
    )
    update_master_settings(
        master_settings, "nsteps", args.nsteps[0]
    )
    update_master_settings(
        master_settings, "decode_step", args.decode_step[0]
    )
    update_master_settings(
        master_settings, "constraint_update", args.constraint_update[0]
    )
    update_master_settings(
        master_settings, "rescale", args.rescale[0]
    )
    update_master_settings(
        master_settings, "max_spawns", args.max_spawns[0]
    )
    update_master_settings(
        master_settings, "audio_indir", args.audio_indir[0]
    )
    update_master_settings(
        master_settings, "outdir", args.outdir[0]
    )
    update_master_settings(
        master_settings, "targets_path", args.targets_path[0]
    )
    update_master_settings(
        master_settings, "spawn_delay", args.spawn_delay[0]
    )

    seed = 420
    random.seed(seed)
    np.random.seed(seed)

    experiments[experiment](master_settings)
