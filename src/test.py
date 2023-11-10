#!/usr/bin/env python3

import librosa, sys
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from pathlib import Path

from librosa_extractor import MFE


def parse_args(argv) -> ArgumentParser:
	parser = ArgumentParser(
		prog='Music Feature Extractor',
		description='Processes music files to retrieve feature data as images.'
	)

	# TODO: Add arguments:
	# * '-d DIR' for globbing input files

	parser.add_argument(
		'-i',
		dest="input_files",
		type=list,
		action="append",
		default=[],
		help= "A single music file to process. Default: librosa.example('nutcracler')."
	)

	parser.add_argument(
		'-o',
		dest='out_dir',
		type=Path,
		action='store',
		default=Path.cwd().joinpath('music_features'),
		help="Output directory to be created. Every input files gets their music features stored in equally named folders. Default: 'CWD/music_features'"
	)

	parser.add_argument(
		'-f',
		dest='extract_features',
		type=list,
		action='append',
		default=MFE.__ITERABLES__,
		help='Specify certain features to extract instead of all possible.'
	)

	parser.add_argument(
		'-r', '--recursive',
		dest='recursive_globbing',
		action='store_true',
		default=False,
		help='When a given input directory with "-d" is given, traverse the folder tree to capture all audio files below. Note: The outup directory will not regard sub folders, so duplicate files will overwrite each other.'
	)

	args = parser.parse_args(args=argv[1:])

	# set default input example
	if not args.input_files:
		args.input_files = [ Path(librosa.example('nutcracker', hq=True)) ]

	# validate given examples
	for feature in args.extract_features:
		if not hasattr(MFE, feature):
			raise ValueError("asd foo")

	return args


def save_plot(feature_data, mfe, out_path: Path) -> None:
	"""
	This function stores given feature data in 'output_dir'.
	"""

	figure, axes = plt.subplots()
	figure.frameon = False
	axes.set_axis_off()

	# initializes a matplotlib.collection.QuadMesh object, used in the next call
	librosa.display.specshow(
		feature_data,
		ax=axes,
		sr=mfe.sample_rate,
		hop_length=mfe.hop_length,
		fmax=mfe.fmax
	)

	# store file
	figure.savefig(out_path, bbox_inches='tight', pad_inches=0)

	# cleanup of matplotlib+librosa spaghetti mess
	plt.close(figure)
	del figure, axes


if __name__ == "__main__":

	args = parse_args(sys.argv)

	for i_file in args.input_files:
		print(f"Processing: {i_file}")

		mfe = MFE(i_file)

		out_dir=args.out_dir.joinpath(mfe.source_file.name)

		if not out_dir.exists():
			print(f"Creating directory: {out_dir}")
			out_dir.mkdir(parents=True)

		for feature_name, feature_data in mfe:
			print(f"... {feature_name}")
			out_path = out_dir / f"{feature_name}.png"
			try:
				save_plot(
					feature_data,
					mfe,
					out_path
				)

			except Exception as e:
				e.args = (*e.args, f"Failed to write feature: '{feature_name}'")
				raise
