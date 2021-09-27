#!/usr/bin/env python3

import librosa
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from typing import List
from cached_property import cached_property

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


class MFExtractor:
	"""
	This class is a stateful wrapper around the libroa API.
	It laods a single music file.
	"""
	def __init__(
		self,
		file_path: pathlib.Path,
		sample_rate=22050,
		hop_length=512,
		n_mfcc=13
	):
		self.file_path = file_path
		self.sample_rate = sample_rate
		self.hop_length = hop_length
		self.n_mfcc = n_mfcc

	@cached_property
	def signal(self):
		# elm [1] is auto detected sample rate
		return librosa.load(self.file_path, self.sample_rate)[0]

	@cached_property
	def signal_harmonic_percussive(self):
		return librosa.effects.hpss(self.signal)

	@property
	def signal_harmonic(self):
		return self.signal_harmonic_percussive[0]

	@property
	def signal_percussive(self):
		return self.signal_harmonic_percussive[1]

	@property
	def beat_frames(self):
		return librosa.beat.beat_track(
			y=self.signal_percussive,
			sr=self.sample_rate
		)[1] # elm [0] is tempo

	#@cached_property
	#def beat_times(self):
	#	return librosa.frames_to_time(self.beat_frames, self.sample_rate)

	@cached_property
	def mfcc(self):
		return librosa.feature.mfcc(
			y=self.signal,
			sr=self.sample_rate,
			hop_length=self.hop_length,
			n_mfcc=self.n_mfcc
		)

	@cached_property
	def mfcc_delta(self):
		return librosa.feature.delta(self.mfcc)

	@cached_property
	def beat_mfcc_delta(self):
		return librosa.util.sync(
			np.vstack([self.mfcc, self.mfcc_delta]),
			self.beat_frames
		)

	@cached_property
	def chromagram(self):
		return librosa.feature.chroma_cqt(
			y=self.signal_harmonic,
			sr=self.sample_rate
		)

	@cached_property
	def beat_chroma(self):
		return librosa.util.sync(
			self.chromagram,
			self.beat_frames,
			aggregate=np.median # np.[min,max,std](default:avg)
		)

	@cached_property
	def beat_features(self):
		return np.vstack([self.beat_chroma, self.beat_mfcc_delta])

	@cached_property
	def short_time_fourier_transform(self):
		return librosa.stft(self.signal, hop_length=self.hop_length)

	@cached_property
	def spectrum(self):
		return np.abs(self.short_time_fourier_transform)

	@cached_property
	def log_spectrum(self):
		return librosa.amplitude_to_db(self.spectrum)

	@cached_property
	def viterbi(self):
		return librosa.amplitude_to_db(
			librosa.magphase(self.short_time_fourier_transform)[0],
			ref=np.max
		)

	def __iter__(self):
		for x in (
			# name					(cached_)property
			#("spectrum",			self.spectrum),
			("log_spectrum",		self.log_spectrum),
			("beat_chroma",			self.beat_chroma),
			("beat_features",		self.beat_features),
			("viterbi",				self.viterbi),
			("mfcc",				self.mfcc),
			("mfcc_delta",			self.mfcc_delta),
			("chromagram",			self.chromagram),
		):
			yield x


def glob_music_files(input_pathes: List[pathlib.Path]) -> List[pathlib.Path]:
	"""
	If the 'input_pathes' list contains folders, these folders will recursively
	searched for all supported formats: '[ {} ]'
	"""

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


glob_music_files.__doc__ = glob_music_files.__doc__.format(
	", ".join(supported_formats())
)


def store_music_features(music_extractor, output_path: pathlib.Path):
	mfile_name = music_extractor.file_path.name
	for ft_name, feature in music_extractor:
		try:
			fig, ax = plt.subplots()
			librosa.display.specshow(
				feature,
				ax=ax,
				sr=music_extractor.sample_rate,
				hop_length=music_extractor.hop_length
			)
			fig.savefig(mfile_name + f"_{ft_name}.png")
			plt.close(fig)
		except Exception as e:
			print("EXCEPTION:", e, ft_name)
			e.args = (*e.args, f"Failed to write feature: '{ft_name}'" )
			raise


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


	# clean/process/check the input args
	args.input_pathes = glob_music_files(args.input_pathes)

	if not args.output_path.exists():
		raise ValueError(f"'{args.output_path}' ouput path does't exist.")


	print("Reading music features ...")
	music_extractors = set()

	for music_file in args.input_pathes:
		music_extractors.add(MFExtractor(music_file))


	print("Writing music features ...")
	for mfe in music_extractors:
		store_music_features(mfe, args.output_path)

	BP()




# TODO: args: --reset --librosa-opts
