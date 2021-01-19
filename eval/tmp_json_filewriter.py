#!/usr/bin/env python

import os
import sys
import json

import numpy as np
import scipy.io.wavfile as wavfile

from base64 import b64decode

def main(json_filepath, outdir):
  with open(json_filepath, 'r') as f:
    data = json.load(f)
  for d in data:
    if "delta" in d.keys():
      delta = np.loads(b64decode(d["adv"][0]))
      print(delta)
      outpath = os.path.join(outdir, os.path.basename(d["filepath"][0]))
      wavfile.write(outpath, 16000, delta)

if __name__ == '__main__':
    args = sys.argv[1:]
    main(*args)
