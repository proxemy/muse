#!/usr/bin/env python3

import librosa
import numpy as np
from pathlib import Path
from functools import cached_property
from dataclasses import dataclass


@dataclass(frozen=True)
class MFE: # Music Feature Extractor

	source_file: Path # initialized by constructor
	sample_rate_forced: int = None # librosa will read the sample rate from input file
	#hop_length: int = 512
	n_mfcc: int = 13
	n_mels: int = 128
	fmax: int = 8000

	def __iter__(self):
		for extractor_method in [
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
		]:
			yield extractor_method, getattr(self, extractor_method)
			#import gc
			#gc.collect()
			#del self.__dict__[v] # to reduce memory footprint


	@cached_property
	def load(self):
		return librosa.load(self.source_file, sr=self.sample_rate_forced)

	@property
	def signal(self):
		return self.load[0]

	@property
	def sample_rate(self):
		return self.load[1]

	@cached_property
	def hop_length(self):
		return librosa.time_to_samples(1./200, sr=self.sample_rate)

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

	@cached_property
	def beat_track(self):
		return librosa.beat.beat_track(
			y=self.signal_percussive,
			sr=self.sample_rate
		)

	@property
	def beat_tempo(self):
		return self.beat_track[0]

	@property
	def beat_frames(self):
		return self.beat_track[1]

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
			y=self.signal,
			sr=self.sample_rate,
			onset_envelope=self.onset_strength,
			hop_length=self.hop_length,
			norm=None
		))

	@cached_property
	def tempogram_fourier(self):
		return np.abs(librosa.feature.fourier_tempogram(
			y=self.signal,
			sr=self.sample_rate,
			onset_envelope=self.onset_strength,
			hop_length=self.hop_length
		))
