from pathlib import Path
import time

import essentia
import essentia.standard as es
import librosa
import librosa.display
# IMPORTANT: since TCN is not available on pip version of madmom
# you have to build the library from source
import madmom
import matplotlib.pyplot as plt
import mir_eval
import numpy as np

base_path = Path.cwd().parent
experiments_path = base_path / "experiments_results/baselines"
dataset_path = base_path.parent.parent / "datasets/candombe"

def librosa_beats(audio):
    bpm, beats = librosa.beat.beat_track(x, sr=SR, units="time")
    return beats

def essentia_beats(audio):
    beats, confidence = es.BeatTrackerMultiFeature()(x)
    return beats

#refence for implementation https://github.com/CPJKU/madmom/issues/403
def madmom_rnn_beats(audio):
    beat_processor = madmom.features.beats.RNNBeatProcessor()
    beat_decoder = madmom.features.beats.DBNBeatTrackingProcessor(beats_per_bar=[4], fps=100)
    beats = beat_decoder(beat_processor(audio))
    return beats

def madmom_tcn_beats(audio):
    beat_processor = madmom.features.beats.TCNBeatProcessor()
    beat_decoder = madmom.features.beats.DBNBeatTrackingProcessor(beats_per_bar=[4], fps=100)
    beats = beat_decoder(beat_processor(audio))
    return beats
