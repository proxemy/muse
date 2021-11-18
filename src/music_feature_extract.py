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

def supported_examples():
	return list(librosa.util.files.__TRACKMAP.keys())


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

	source: pathlib.Path
	is_example: bool = False
	sample_rate: int = 22050
	hop_length: int = 512
	n_mfcc: int = 13
	n_mels: int = 128
	fmax: int = 8000


	def __post_init__(self):
		"""Validates the construction arguments. Raises Exceptions."""
		if not isinstance(self.source, (str, pathlib.Path)):
			raise TypeError(f"Illegal initialization type '{type(self.source)}' for MFExtractor.")

		if self.is_example:
			if not isinstance(self.source, str) \
			or not self.source in supported_examples():
				raise ValueError(
					f"Illegal librosa example '{self.source}', use '{supported_examples()}'."
				)

		else:
			if not isinstance(self.source, pathlib.Path) \
			or not self.source.exists():
				raise FileNotFoundError(f"Music file '{self.source}' not found.")

	__ITERABLES__ = [
			"mfcc",
			"mfcc_delta",
			"mfcc_beat_delta",
			"chromagram_stft",
			"chromagram_cqt",
			"chromagram_cens",
			"spectrogram_mel",
			"spectrogram_pcen",
			"spectrogram_magphase",
			"spectrogram_harmonic",
			"spectrogram_percussive",
			"tempogram_autocorrelated",
			"tempogram_fourier"
		]

	def __iter__(self):
		for v in MFExtractor.__ITERABLES__:
			yield v, getattr(self, v)
			del self.__dict__[v] # to reduce memory footprint

	def is_supported(file_path_or_example) -> bool:
		return \
			file_path_or_example in supported_examples() or \
			pathlib.Path(file_path_or_example).suffix[1:].upper() in supported_formats()

	@property
	def name(self):
		return self.source.name if not self.is_example else self.source



	#################
	# librosa stuff #
	#################

	@cached_property
	def signal(self):
		# elm [1] is auto detected sample rate with 'sr=None' param
		return librosa.load(
			librosa.ex(self.source) if self.is_example else self.source,
			sr=self.sample_rate
		)[0]

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
		return librosa.decompose.hpss(self.short_time_fourier_transform)

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
	def mfcc_beat_delta(self):
		return librosa.util.sync(
			np.vstack([self.mfcc, self.mfcc_delta]),
			self.beat_frames,
			aggregate=np.median
		)

	@cached_property
	def short_time_fourier_transform(self):
		return librosa.stft(self.signal, hop_length=self.hop_length)

	@cached_property
	def chromagram_stft(self):
		return librosa.feature.chroma_stft(y=self.signal, sr=self.sample_rate)

	@cached_property
	def chromagram_cqt(self):
		return librosa.feature.chroma_cqt(y=self.signal, sr=self.sample_rate)

	@cached_property
	def chromagram_cens(self):
		return librosa.feature.chroma_cens(y=self.signal, sr=self.sample_rate)

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
	def spectrogram_magphase(self):
		return librosa.amplitude_to_db(
			librosa.magphase(self.short_time_fourier_transform)[0],
			ref=np.max
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

	@cached_property
	def onset_strength(self):
		return librosa.onset.onset_strength(
			y=self.signal,
			sr=self.sample_rate,
			hop_length=self.hop_length
		)

	@cached_property
	def tempogram_autocorrelated(self):
		return np.abs(librosa.feature.tempogram(
			self.onset_strength,
			sr=self.sample_rate,
			hop_length=self.hop_length,
			norm=False
		))

	@cached_property
	def tempogram_fourier(self):
		return np.abs(librosa.feature.fourier_tempogram(
			self.onset_strength,
			sr=self.sample_rate,
			hop_length=self.hop_length
		))



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
			if MFExtractor.is_supported(p):
				ret.update({p})
			else:
				raise ValueError(f"'{p}' input file format is not supported.")

		elif p.is_dir():
			ret.update([
				f for f in p.rglob("*") if MFExtractor.is_supported(f)
			])

	if not ret:
		raise ValueError(f"Could't find supported audio files at '{input_pathes}'")

	return list(ret)


glob_music_files.__doc__ = glob_music_files.__doc__.format(
	", ".join(supported_formats())
)


def store_music_features(music_extractor, output_path: pathlib.Path) -> None:
	mfile_name = music_extractor.name
	for ft_name, feature in music_extractor:
		print(f"   {ft_name} ...")
		try:
			fig, ax = plt.subplots()
			fig.frameon = False
			ax.set_axis_off()
			librosa.display.specshow(
				feature,
				ax=ax,
				sr=music_extractor.sample_rate,
				hop_length=music_extractor.hop_length,
				fmax=music_extractor.fmax
			)
			fig.savefig(
				mfile_name + f"_{ft_name}.png",
				bbox_inches='tight',
				pad_inches=0
			)
			plt.close(fig)
		except Exception as e:
			print("EXCEPTION:", e, ft_name)
			e.args = (*e.args, f"Failed to write feature: '{ft_name}'" )
			raise


if __name__ == "__main__":
	import argparse
	arg_parser = argparse.ArgumentParser(description=__doc__)

	grp = arg_parser.add_mutually_exclusive_group(required=True)
	grp.add_argument("-i",
		dest="input_pathes",
		type=pathlib.Path,
		action="append",
		default=[],
		help= "A single music file or path for recursive lookup. (multiple)"
	)
	grp.add_argument("-e",
		dest="examples",
		type=str,
		action="append",
		default=[],
		help="Librosa example audio to load/process. Does not get saved."
	)

	arg_parser.add_argument("-o",
		dest="output_path",
		type=pathlib.Path,
		default=pathlib.Path().cwd(),
		help="Output folder. New directories will be created there."
	)
	arg_parser.add_argument("--debug",
		dest="debug",
		action="store_true",
		default=False,
		help="Toggle on to print more output."
	)

	args = arg_parser.parse_args()

	if not args.debug:
		import warnings
		warnings.filterwarnings("ignore")

	if not args.output_path.exists():
		raise ValueError(f"'{args.output_path}' ouput path does't exist.")

	if args.input_pathes:
		args.input_pathes = glob_music_files(args.input_pathes)


	music_extractors = set()

	for music_file in args.input_pathes:
		music_extractors.add(MFExtractor(music_file))
	for music_exmaple in args.examples:
		music_extractors.add(MFExtractor(music_exmaple, is_example=True))


	print("Writing music features ...")
	for mfe in music_extractors:
		print(mfe.name)
		store_music_features(mfe, args.output_path)

	if False:
		BP() # hush hush little Syntastic
