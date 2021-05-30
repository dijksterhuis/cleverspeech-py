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


def write(path, data, sample_rate=16000, bit_depth=16):
    # change to write()

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
        assert min(data) >= -2**(bit_depth - 1)
        assert max(data) <= 2**(bit_depth - 1) - 1

    except AssertionError as e:
        err = "Conversion to float32 wav output failed for {}\n".format(
            path)
        err += "min(data)={}\n".format(min(data))
        err += "max(data)={}\n".format(max(data))
        err += "Output data must respect assertion: -2^15 <= x <= 2^15 - 1 \n"
        print(err)
        raise

    # normalise
    normalised = np.where(
        np.less(data, np_zero(data.size, np.float32)),
        data / (2 ** (bit_depth - 1)),
        data / (2 ** (bit_depth - 1) - 1)
    )

    soundfile.write(
        path,
        normalised,
        sample_rate,
        subtype="FLOAT",
        format="WAV"
    )
