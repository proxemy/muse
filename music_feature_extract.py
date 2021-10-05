#!/usr/bin/env python3

import librosa, librosa.display
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from functools import cached_property
from dataclasses import dataclass

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


@dataclass(frozen=True)
class MFExtractor:
	"""
	This class is a stateful wrapper around the libroa API.
	It laods a single music file.
	"""

	file_path: pathlib.Path
	sample_rate: int = 22050
	hop_length: int = 512
	n_mfcc: int = 13
	n_mels: int = 128
	fmax: int = 8000

	@cached_property
	def signal(self):
		# elm [1] is auto detected sample rate with 'sr=None' param
		return librosa.load(self.file_path, sr=self.sample_rate)[0]

	@cached_property
	def signal_harmonic_percussive(self):
		return librosa.effects.hpss(self.signal)

	@property
	def signal_harmonic(self):
		return self.signal_harmonic_percussive[0]

	@property
	def signal_percussive(self):
		return self.signal_harmonic_percussive[1]

	@cached_property
	def decompose_harmonic_percussive(self):
		return librosa.decompose.hpss(
			self.short_time_fourier_transform,
			#kernel_size=16,
			margin=16,
		)

	@cached_property
	def decompose_harmonic(self):
		return self.decompose_harmonic_percussive[0]

	@cached_property
	def decompose_percussive(self):
		return self.decompose_harmonic_percussive[1]

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
			self.beat_frames,
			aggregate=np.median
		)

	@cached_property
	def beat_features(self):
		return np.vstack([self.chromagram_beat, self.beat_mfcc_delta])

	@cached_property
	def short_time_fourier_transform(self):
		return librosa.stft(self.signal, hop_length=self.hop_length)

	@cached_property
	def spectrum(self):
		return np.abs(self.short_time_fourier_transform)

	@cached_property
	def spectrogram_log(self):
		return librosa.amplitude_to_db(self.spectrum)

	@cached_property
	def viterbi(self):
		return librosa.amplitude_to_db(
			librosa.magphase(self.short_time_fourier_transform)[0],
			ref=np.max
		)

	@cached_property
	def chromagram_stft(self):
		return librosa.feature.chroma_stft(self.signal, sr=self.sample_rate)

	@cached_property
	def chromagram_cqt(self):
		return librosa.feature.chroma_cqt(self.signal, sr=self.sample_rate)

	@cached_property
	def chromagram_cens(self):
		return librosa.feature.chroma_cens(y=self.signal, sr=self.sample_rate)

	@cached_property
	def chromagram_beat(self):
		return librosa.util.sync(
			self.chromagram_harmonic,
			self.beat_frames,
			aggregate=np.median # np.[min,max,std](default:avg)
		)

	@cached_property
	def chromagram_harmonic(self):
		return librosa.feature.chroma_cqt(
			self.signal_harmonic,
			sr=self.sample_rate
		)

	@cached_property
	def chromagram_percussive(self):
		return librosa.feature.chroma_cqt(
			self.signal_percussive,
			sr=self.sample_rate
		)

	@cached_property
	def spectrogram_mel(self):
		return librosa.power_to_db(
			librosa.feature.melspectrogram(
				y=self.signal,
				hop_length=self.hop_length,
				n_mels=self.n_mels,
				fmax=self.fmax
			),
			ref=np.max
		)

	@cached_property
	def spectrogram_pcen(self):
		return librosa.pcen(
			np.abs(self.short_time_fourier_transform),
			sr=self.sample_rate,
			hop_length=self.hop_length
		)

	@cached_property
	def spectrogram_harmonic(self):
		return librosa.amplitude_to_db(
			np.abs(self.decompose_harmonic),
			ref=np.max(np.abs(self.short_time_fourier_transform))
		)

	@cached_property
	def spectrogram_percussive(self):
		return librosa.amplitude_to_db(
			np.abs(self.decompose_percussive),
			ref=np.max(np.abs(self.short_time_fourier_transform))
		)

	def __iter__(self):
		for v in [
			"beat_features",
			"beat_mfcc_delta",
			"viterbi",
			"mfcc",
			"mfcc_delta",
			"chromagram_stft",
			"chromagram_cqt",
			"chromagram_cens",
			"chromagram_beat",
			"chromagram_harmonic",
			"chromagram_percussive",
			"spectrogram_log",
			"spectrogram_mel",
			"spectrogram_pcen",
			"spectrogram_harmonic",
			"spectrogram_percussive",
		]:
			yield v, getattr(self, v)
			del self.__dict__[v] # to reduce the memory footprint


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
		print(f"   {ft_name} ...")
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


	music_extractors = set()

	for music_file in args.input_pathes:
		music_extractors.add(MFExtractor(music_file))


	print("Writing music features ...")
	for mfe in music_extractors:
		print(mfe.file_path.name)
		store_music_features(mfe, args.output_path)

	if False:
		BP()




# TODO: args: --reset --librosa-opts
