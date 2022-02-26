from mir_eval import beat

import numpy as np
import pandas as pd

def run_beat_detection_experiment(experiment_name, onset_parameters,
    beat_parameters, override=False):

    # here, we receive every parameter for onset detection
    onset_subbands = librosa.onset.onset_strength_multi(**onset_parameters)

    # FIXME: how to deal with multiple subbands?
    beat_parameters[f'onset_envelope'] = onset_subbands[0]
    bpm, beat_frames = librosa.beat.beat_track(**beat_parameters)
    beat_timestamps = librosa.frames_to_time(beat_frames, kwargs['fs'])

    if not os.path.isfile(filename) or override == True:
        np.savez(
                os.path.join(folder_path, os.path.basename(filename)),
                onset=onset_subbands[0],
                reference=reference,
                estimated=beat_timestamps
        )

    return onset_subbands, beat_timestamps

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
        onset_parameters['y'] = x

        onsets, beats = run_beat_detection_experiment(onset_parameters, beat_parameters)

    return
