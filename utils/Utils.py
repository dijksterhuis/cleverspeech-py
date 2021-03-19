import numpy as np

from os import path
from datetime import datetime as time
from base64 import b64encode


def dump_b64bytes(z):
    return b64encode(z.dumps()).decode()


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
        outfile = path.join(outdir, "log.txt")

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
    log("Initial decodings:", '\n'.join(s), wrap=True)

    s = ["{:.0f}".format(x) for x in batch.audios["real_feats"]]
    log("Real Features: ", "\n".join(s), wrap=True)

    s = ["{:.0f}".format(x) for x in batch.audios["ds_feats"]]
    log("DS Features: ", "\n".join(s), wrap=True)

    s = ["{:.0f}".format(x) for x in batch.audios["n_samples"]]
    log("Real Samples: ", "\n".join(s), wrap=True)

