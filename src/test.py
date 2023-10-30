#!/usr/bin/env python3

import librosa
from librosa_extractor import MFE

mfe = MFE(librosa.example('nutcracker', hq=True))

for f in mfe:
	print(f)
