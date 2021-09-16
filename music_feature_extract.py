#!/usr/bin/env python3

import librosa
import pathlib
import numpy as np

from pdb import set_trace as BP


def supported_formats():
	from soundfile import available_formats as sf_avail_fmts

	# mp3 is supported via librosas audioread dependecy
	return sorted([ *sf_avail_fmts().keys(), "MP3" ])


__doc__ = \
	"""
	This script takes all audio files, either singular or recursive
	if a directory is passed and generates the several music feature extracts
	for the model to train on.
	Supported formats: [ {} ]
	""".format(", ".join(supported_formats()))


##################
# business logic #
##################


def glob_music_files(input_pathes):
	"""
	If the 'input_pathes' list contains folders, these folders will recursively
	searched for all supported formats: '[ {} ]'
	""".format(", ".join(supported_formats()))

	ret = set()

	for p in input_pathes:
		if not p.exists():
			raise ValueError(f"'{p}' input path does not exist.")

		if p.is_file():
			if p.suffix[1:].upper() in supported_formats():
				ret.update({p})
			else:
				raise ValueError(f"'{p}' input file format is not supported.")

		elif p.is_dir():
			ret.update([
				f for f in p.rglob("*")
				if f.suffix[1:].upper() in supported_formats()
			])

	if not ret:
		raise ValueError(f"Could't find supported audio files at '{input_pathes}'")

	return ret


def extract_music_features(music_file, librosa_opts):
	y, sr = librosa.load(music_file)

	tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

	beat_times = librosa.frames_to_time(beat_frames, sr=sr)

	y_harmonic, y_percussive = librosa.effects.hpss(y)

	mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)

	mfcc_delta = librosa.feature.delta(mfcc)

	beat_mfcc_delta = librosa.util.sync(
		np.vstack([mfcc, mfcc_delta]), beat_frames
	)

	chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

	beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

	beat_feature = np.vstack([beat_chroma, beat_mfcc_delta])

	return {
		'y' : y,
		'sr' : sr,
		'tempo' : tempo,
		'beat_frames' : beat_frames,
		'beat_times' : beat_times,
		'y_harmonic' : y_harmonic,
		'y_percussive' : y_percussive,
		'mfcc' : mfcc,
		'mfcc_delta' : mfcc_delta,
		'beat_mfcc_delta' : beat_mfcc_delta,
		'chromagram' : chromagram,
		'beat_chroma' : beat_chroma,
		'beat_feature' : beat_feature
	}


if __name__ == "__main__":
	import argparse

	arg_parser = argparse.ArgumentParser(description=__doc__)

	for arg in [
		( "-i", {
			"dest" : "input_pathes",
			"type" : pathlib.Path,
			"required" : True,
			"action" : "append",
			"help" : "A single music file or path for recursive lookup. (multiple)"
		}),
		( "-o", {
			"dest" : "output_path",
			"type" : pathlib.Path,
			"default" : pathlib.Path().cwd(),
			"help" : "Output folder. New directories will be created there."
		}),
	]:
		arg_parser.add_argument(arg[0], **arg[1])

	args = arg_parser.parse_args()

	args.input_pathes = glob_music_files(args.input_pathes)

	music_features = dict()

	for music_file in args.input_pathes:
		m_features = extract_music_features(music_file, None)

		music_features.update({ music_file : m_features })
		BP()
