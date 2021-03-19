import os

import numpy as np
from matplotlib import pyplot as plt
from cleverspeech.eval.DetectionMetrics import to_db

# FROM: https://stackoverflow.com/a/49157454/5945794


def short_time_fourier_transform(x, frame_size=512, overlap=0.85, window=np.hanning):
    """
    Short time fourier transform of audio signal
    """
    win = window(frame_size)
    hop_size = int(frame_size - np.floor(overlap * frame_size))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frame_size / 2.0))), x)

    # cols for windowing
    cols = np.ceil((len(samples) - frame_size) / float(hop_size)) + 1

    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frame_size))

    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(
            int(cols),
            frame_size
        ),
        strides=(samples.strides[0] * hop_size, samples.strides[0])
    ).copy()

    frames *= win

    return np.fft.rfft(frames)


def log_scale_spec(spec, sr=44100, factor=20.):
    """
    Scale frequency axis logarithmically
    """
    time_bins, freq_bins = np.shape(spec)

    scale = np.linspace(0, 1, freq_bins) ** factor
    scale *= (freq_bins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([time_bins, len(scale)]))

    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(
                spec[:, int(scale[i]):int(scale[i + 1])],
                axis=1
            )

    # list center freq of bins
    all_freqs = np.abs(np.fft.fftfreq(freq_bins * 2, 1. / sr)[:freq_bins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(all_freqs[int(scale[i]):])]
        else:
            freqs += [np.mean(all_freqs[int(scale[i]):int(scale[i + 1])])]

    return newspec, freqs


def generate_spectrogram(
        x,
        plot_path,
        sample_rate=16000,
        bin_size=2 ** 9,
        colormap="Greys"
):
    """
    Plot a spectrogram.

    :param x:
    :param sample_rate:
    :param bin_size:
    :param plot_path:
    :param colormap:
    :return:
    """
    s = short_time_fourier_transform(x, bin_size)

    sshow, freq = log_scale_spec(s, factor=1.0, sr=sample_rate)
    inverted_mag_spec = to_db(sshow)
    time_bins, freq_bins = np.shape(inverted_mag_spec)

    title = " ".join(plot_path.split(".")[0].split("/"))

    plt.rcParams.update({'font.size': 10})
    plt.figure(figsize=(6, 3))
    plt.imshow(
        np.transpose(inverted_mag_spec),
        origin="lower",
        aspect="auto",
        cmap=colormap,
        interpolation="none",
        vmin=0.0
    )
    plt.colorbar()
    plt.xlabel("time (n samples)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, time_bins - 1])
    plt.ylim([0, freq_bins])
    plt.title(title)

    x_locations = np.float32(np.linspace(0, time_bins - 1, 5))
    y_locations = np.int16(np.round(np.linspace(0, freq_bins - 1, 10)))

    x_ticks = (x_locations * len(x) / time_bins) + (0.5 * bin_size)
    y_ticks = ["%.02f" % freq[i] for i in y_locations]

    plt.xticks(x_locations, ["%.02f" % l for l in (x_ticks / sample_rate)])
    plt.yticks(y_locations, y_ticks)

    plt.savefig(plot_path, bbox_inches="tight")


