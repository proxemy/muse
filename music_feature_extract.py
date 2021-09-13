#!/usr/bin/env python3


__supported_formats__ = [ "mp3", "ogg", "flac", "wav" ]

__doc__ = \
	"This script takes all audio files, either singular or recursive " \
	"if a directory is passed and generates the several music feature extracts " \
	"for the model to train on. " \
	"Supported formats are {}".format(str(__supported_formats__))

import argparse
import pathlib

from pdb import set_trace as BP


def glob_music_files(input_pathes):
	"""
	If the 'input_pathes' list contains folders, these folders will recursively
	searched for all supported formats: '{}'
	""".format(__supported_formats__)

	ret = []

	for p in input_pathes:
		if not p.exists():
			raise ValueError(f"'{p}' input path does not exist.")

		if p.is_file():
			ret.append(p)
		elif p.is_dir():
			for f in __supported_formats__:
				ret.extend(p.rglob("*." + f))

	return ret



if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser(description=__doc__)

	for arg in [
		( "-i", {
			"dest" : "input_pathes",
			"type" : pathlib.Path,
			"required" : True,
			"action" : "append",
			"help" : "A single music file or a path for recursive lookup. (multiple)"
		}),
		( "-o", {
			"dest" : "output_path",
			"type" : pathlib.Path,
			"default" : pathlib.Path().cwd(),
			"help" : "Output folder. New directory will be created there."
		}),
	]:
		arg_parser.add_argument(arg[0], **arg[1])

	args = arg_parser.parse_args()

	args.input_pathes = glob_music_files(args.input_pathes)

	print(args)
	BP()
