#!/usr/bin/env python3


import unittest
from music_feature_extract import MFExtractor






class MFExtractorTest(unittest.TestCase):
	REFERENCE_RESULTS = \
	{
		"vibeace": {
			"mfcc" : -7206083671683317838,
			"mfcc_delta" : -4054128547765855004,
			"mfcc_beat_delta" : 1629464358134076612,
			"chromagram_stft" : 3981571625101903382,
			"chromagram_cqt" : 3802248392113137247,
			"chromagram_cens" : -1250527912563989085,
			"spectrogram_mel" : -5584022938782416151,
			"spectrogram_pcen" : -7041195504635244482,
			"spectrogram_magphase" : 1915781698322244099,
			"spectrogram_harmonic" : -3511387923605960957,
			"spectrogram_percussive" : -6083496458987600477,
			"tempogram_autocorrelated" : -452924369690473656,
			"tempogram_fourier" : 4832758760259490297,
		},
		"choice": {
			"mfcc" : 2960638511319998897,
			"mfcc_delta" : -2334344103802219919,
			"mfcc_beat_delta" : 6686746477965439310,
			"chromagram_stft" : 1774254514344899254,
			"chromagram_cqt" : -2305900307777939391,
			"chromagram_cens" : 7983568927715134155,
			"spectrogram_mel" : -5476589229134819581,
			"spectrogram_pcen" : 3465213531483176788,
			"spectrogram_magphase" : 3254151284310818721,
			"spectrogram_harmonic" : 751208318720237622,
			"spectrogram_percussive" : -1465743182368761520,
			"tempogram_autocorrelated" : -3555658334917057775,
			"tempogram_fourier" : -5094303337488798184,
		},
		"nutcracker": {
			"mfcc" : -5836251851795311123,
			"mfcc_delta" : 236120868034048862,
			"mfcc_beat_delta" : 8499801756935092638,
			"chromagram_stft" : 3405266839279436951,
			"chromagram_cqt" : -2692271519595848751,
			"chromagram_cens" : -7974958405678189660,
			"spectrogram_mel" : 6887733172066030934,
			"spectrogram_pcen" : -2151001576092906733,
			"spectrogram_magphase" : 101073302014557252,
			"spectrogram_harmonic" : 6958429882382301766,
			"spectrogram_percussive" : 1716814834186326908,
			"tempogram_autocorrelated" : -4511522563098967776,
			"tempogram_fourier" : 7409593033121915990,
		},
		"brahms": {
			"mfcc" : -5579427344810278848,
			"mfcc_delta" : -851625157571140215,
			"mfcc_beat_delta" : -4393009399564616787,
			"chromagram_stft" : -3931177781686674147,
			"chromagram_cqt" : -7361282454851485226,
			"chromagram_cens" : 1125475188065498809,
			"spectrogram_mel" : 4040563836993443595,
			"spectrogram_pcen" : 1437462994311164286,
			"spectrogram_magphase" : -3158866670282053895,
			"spectrogram_harmonic" : 9043261688568735513,
			"spectrogram_percussive" : -6112073388889479858,
			"tempogram_autocorrelated" : 362909557621852289,
			"tempogram_fourier" : 507126528249083739,
		},
		"trumpet": {
			"mfcc" : -4262972029490841443,
			"mfcc_delta" : -8769150666037383104,
			"mfcc_beat_delta" : -5039760820930826894,
			"chromagram_stft" : -6000260765937863663,
			"chromagram_cqt" : 3200993207204764362,
			"chromagram_cens" : -8369048249850022073,
			"spectrogram_mel" : -5187006916956913566,
			"spectrogram_pcen" : 8394155905990345281,
			"spectrogram_magphase" : -614623913601875790,
			"spectrogram_harmonic" : -2357161239163580257,
			"spectrogram_percussive" : -6536689062623313570,
			"tempogram_autocorrelated" : -7978764829502109566,
			"tempogram_fourier" : 4277167556812737065,
		},
		"fishin": {
			"mfcc" : -1029254834879894637,
			"mfcc_delta" : -8551027631710320158,
			"mfcc_beat_delta" : -6753100913255578764,
			"chromagram_stft" : 3248058267505041079,
			"chromagram_cqt" : 9061360904107204330,
			"chromagram_cens" : -5091125314887616590,
			"spectrogram_mel" : -3120085290045622753,
			"spectrogram_pcen" : 7491345524552880850,
			"spectrogram_magphase" : -658714939774535468,
			"spectrogram_harmonic" : -6079627711496806119,
			"spectrogram_percussive" : -6923224592000065947,
			"tempogram_autocorrelated" : 4499689695899229658,
			"tempogram_fourier" : 8349139274240650379,
		}
	}

	def test_examples(self):
		for example_name, reference_results in self.REFERENCE_RESULTS.items():
			print(f"Testing example: {example_name}")
			for feature_name, result in MFExtractor(example_name, is_example=True):
				print(f"\t{feature_name} ...")
				self.assertEqual(
					reference_results[feature_name],
					hash(result.data.hex())
				)
	
	def test(self):
		self.test_examples()
		pass



if __name__ == "__main__":
	unittest.main()

