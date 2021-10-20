import soundfile

import numpy as np

from cleverspeech.utils.Utils import np_zero


def load(fp, dtype):

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


def write(path, data, sample_rate=16000, max_value=1.0, min_value=-1.0):
    # change to write()

    """
    We *always* want to write out as 32-bit floating point wav files.
    Writing as 16 bit signed ints destroys information and affects the
    perturbation.

    Note that we need to perform some additional work if we want to pass 32 bit
    floating point encoded wav files back into to deepspeech - it expects 16 bit
    signed ints as file inputs (although uses float32 for calculations).

    :param min_value:
    :param path: file path to write to
    :param data: the numpy array being written
    :param sample_rate: sample rate of the output file
    :param max_value: How many signed integer bits the input data has for normalisation.

    :return: None
    """
    try:
        assert type(data) is np.ndarray
    except AssertionError:
        print("Wrong data type for data: {}".format(type(data)))
        raise

    assert type(path) is str

    # always convert to float32 for writing to maintain integrity.
    if data.dtype == np.int16 or data.dtype == np.int32:
        data = data.astype(np.float32)

    # Make sure we write out files correctly for float32.
    try:
        assert max_value >= min(data) >= min_value
        assert min_value <= max(data) <= max_value

    except AssertionError:
        print("Values in wav data were beyond max and min bounds!")
        print(" MAX: {ma} MIN: {mi}".format(ma=max(data), mi=min(data)))
        raise

    soundfile.write(
        path,
        data,
        sample_rate,
        subtype="FLOAT",
        format="WAV"
    )
