#!/usr/bin/env python3


# Beat tracking example
import librosa
from librosa_extractor import MFE


mfe = MFE(librosa.example('nutcracker'))


print('Estimated tempo: {:.2f} beats per minute'.format(mfe.beat_tempo))


# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(mfe.beat_frames, sr=mfe.sample_rate)
