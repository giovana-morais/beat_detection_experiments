import matplotlib.pyplot as plt
from mir_eval import beat

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


def run_beat_detection_experiment(experiment_name, onset_parameters,
    beat_parameters):

    # here, we receive every parameter for onset detection
    onset_subbands = librosa.onset.onset_strength_multi(x, fs, **onset_parameters)

    # FIXME: how to deal with multiple subbands?
    beat_parameters[f'onset_envelope'] = onset_subbands[0]
    bpm, beat_frames = librosa.beat.beat_track(**beat_parameters)
    beat_timestamps = librosa.frames_to_time(beat_frames, kwargs['fs'])

    np.savez(
            os.path.join(BASELINE_DEFAULT_PATH, os.path.basename(file)),
            onset=onset_subbands[0],
            reference=reference,
            estimated=beat_timestamps
    )

    return onset_subbands, beat_timestamps

def evaluate_experiment():
    return

def run_candombe_experiment(dataset_path, experiment_name, onset_parameters,
    beat_parameters, **kwargs):
    '''
    abstraction to run experiments within candombe dataset. for this data,
    we have both .wav and .csv files with beat timestamps.

    ---
        dataset_path: str
        experiment_name: str
            this parameter defines where all (onset, beats) files will be stored
            in order to reproduce it faster later
        **kwargs: dict
            additional parameters. if None, all default values
        will be used
    '''

    for file in dataset_path:
        x, fs = librosa.load(f'{file}.wav', mono=True, sr=kwargs['fs'])
        x_df = pd.read_csv(f'{file}.csv', names=['timestamp', 'beat'])
        reference = x_df['timestamp'].values
        onset_parameters['x'] = x

        beat_timestamps = run_beat_detection_experiment(onset_parameters, beat_parameters)
    return
