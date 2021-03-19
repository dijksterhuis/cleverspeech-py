from waveform_analysis import ITU_R_468_weight
from waveform_analysis import A_weighting
from scipy.signal import lfilter


def itu_weighting(x, sample_rate=16000):
    return ITU_R_468_weight(x, sample_rate)


def a_weighting(x, sample_rate=16000):
    b, a = A_weighting(sample_rate)
    return lfilter(b, a, x)

