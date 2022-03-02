from pathlib import Path

import matplotlib.pyplot as plt


def plot_comparison(x, fs, expected_beats, librosa_beats, start=0, end=50):
    fig, ax = plt.subplots(nrows=1, sharex=True)
    ax.plot(x[start*fs:end*fs], label='waveform')

    ax.vlines(
        (expected_beats[(expected_beats >= start) & (expected_beats <= end)]-start)*fs,
              0, 1, alpha=0.5, color='r', linestyle='--', label='groundtruth'
    )
    ax.vlines(
        (librosa_beats[(librosa_beats >= start) & (librosa_beats <= end)]-start)*fs,
        0, 1, alpha=0.5, color='g', linestyle='--', label='librosa')

    ax.legend()

def create_folder(path):
    """
    create folder if it does not exist
    """
    if not path.is_dir():
        print(f"Creating folder for {path.parent}")
        path.mkdir(parents=True)

    return True
