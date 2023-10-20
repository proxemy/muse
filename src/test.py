#!/usr/bin/env python3

import librosa
from librosa_extractor import MFE

mfe = MFE(librosa.example('nutcracker', hq=True))

print('Estimated tempo: {:.2f} beats per minute'.format(mfe.beat_tempo))

beat_times = librosa.frames_to_time(mfe.beat_frames, sr=mfe.sample_rate)

for f in mfe:
	print(f)
