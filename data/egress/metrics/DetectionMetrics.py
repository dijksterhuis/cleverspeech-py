import numpy as np
import python_speech_features as psf
from math import sqrt


def lnorm(x, norm=2):
    if norm == float('inf') or norm == np.inf:
        return np.max(np.abs(x))
    elif norm == 0.0:
        return np.count_nonzero(x != 0)
    else:
        return np.power(np.sum(np.power(np.abs(x), norm)), 1 / norm)


def peak_to_peak(x):
    """
    p2p = max(x) - min(x)

    :param x: audio signal vector of amplitudes over time
    :return: maximal peak-to-peak value of the signal
    """
    return max(x) - min(x)


def energy(x):
    """
    E = sum(x^2)

    :param x: audio signal vector of amplitudes over time
    :return: energy of the signal
    """
    return sum([i**2 for i in np.abs(x)])


def energy_db(x):
    """
    E = sum(x^2)

    :param x: audio signal vector of amplitudes over time
    :return: energy of the signal in decibels relative to full scale
    """
    return to_db(energy(x))


def power(x):
    """
    P = E / t

    :param x: audio signal vector of amplitudes over time
    :return: power of the signal
    """
    return energy(x) / x.size


def power_db(x):
    """
    P_db = 10.log10(P)

    :param x: audio signal vector of amplitudes over time
    :return: power of the signal in decibels relative to full scale
    """
    return to_db(power(x))


def rms_amplitude(x):
    """
    A_rms = (sum_n(A^2)/N)^1/2

    :param x: audio signal vector of amplitudes over time
    :return: rms amplitude of audio signal
    """
    return sqrt(power(x))


def rms_amplitude_db(x):
    """
    :param x: audio signal vector of amplitudes over time
    :return: rms amplitude of signal in decibels relative to full scale
    """
    return to_db(rms_amplitude(x))


def to_db(x):
    """
    db(x) = 10.log10(x)

    :param x: audio signal vector of values over time
    :return: audio signal converted to decibels relative to full scale
    """
    return 10 * np.log10(x)


def snr_power(error_signal, original_signal):
    """
    SNR = P(x) / P(e)

    :param error_signal: adversarial example signal
    :param original_signal: original example signal
    :return: SNR Power based on raw amplitudes.
    """
    return power(original_signal) - power(error_signal)


def snr_power_db(error_signal, original_signal):
    """
    SNR_db = 10.log10(P(x) / P(e))

    :param error_signal: adversarial example signal
    :param original_signal: original example signal
    :return: SNR power in decibels relative to full scale
    """
    return to_db(power(original_signal) / power(error_signal))


def snr_energy(error_signal, original_signal):
    """
    SNR_E = E_s / E_n

    :param error_signal: adversarial example signal
    :param original_signal: original example signal
    :return: SNR energy of input signals
    """
    return energy(original_signal) / energy(error_signal)


def snr_energy_db(error_signal, original_signal):
    """
    SNR_E = E_s / E_n

    :param error_signal: adversarial example signal
    :param original_signal: original example signal
    :return: SNR energy of input signals in decibels relative to full scale
    """
    return to_db(snr_energy(error_signal, original_signal))


def snr_rms_amplitude(error_signal, original_signal):
    """
    SNR_db = 10.log10[ (A_s / A_n) ^2] = 20.log10(A_s/A_n)

    :param error_signal: adversarial example signal
    :param original_signal: original example signal
    :return: SNR of the rms amplitudes of input signals
    """
    return 2 * to_db(rms_amplitude(original_signal) / rms_amplitude(error_signal))


def nsr_power(error_signal, original_signal):
    """
    SNR = P(e)/P(x)

    :param error_signal: adversarial example signal
    :param original_signal: original example signal
    :return: NSR power
    """
    return power(error_signal) - power(original_signal)


def nsr_power_db(error_signal, original_signal):
    """
    NSR_db = 10.log10(P(e)/P(x))

    :param error_signal: adversarial example signal adversarial example signal
    :param original_signal: original example signal
    :return: NSR power in decibels relative to full scale
    """
    return to_db(power(error_signal) / power(original_signal))


def nsr_energy(error_signal, original_signal):
    """
    NSR_E = E(e) / E(x)
    :param error_signal: adversarial example signal adversarial example signal
    :param original_signal: original example signal
    :return: NSR energy
    """
    return energy(error_signal) / energy(original_signal)


def nsr_rms_amplitude(error_signal, original_signal):
    """
    NSR_rms = 10.log10[ (A(e) / A(x)) ^2] = 20.log10(A(e)/A(x))

    :param error_signal: adversarial example signal adversarial example signal
    :param original_signal: original example signal
    :return: NSR of the rms amplitudes of input signals
    """
    return 2 * to_db(rms_amplitude(error_signal) / rms_amplitude(original_signal))


def snr_segmented(error_signal, original_signal, frame_size=512):

    """
    Calculate the Segmented Energy SNR as per Schoenherr et al.

    We *could* use snr_power_db in the function, but it returns the same value:
    the audio vector lengths are the same so the number of samples term N
    cancels out.

    :param error_signal: adversarial example signal adversarial example signal
    :param original_signal: original example signal
    :param frame_size:
    :return: Segmented SNR in decibels relative to full scale..
    """

    original_frame = psf.sigproc.framesig(error_signal, frame_size, frame_size)
    error_frame = psf.sigproc.framesig(original_signal, frame_size, frame_size)

    logged = to_db(snr_energy(original_frame, error_frame))

    return sum(logged) / frame_size



