#!/usr/bin/env python3

import librosa, sys
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from pathlib import Path

from librosa_extractor import MFE


def parse_args(argv) -> ArgumentParser:
	ret = ArgumentParser(
		prog='Music Feature Extractor',
		description='Processes music files to retrieve feature data as images.'
	)

	ret.add_argument(
		'-i',
		dest="input_files",
		type=Path,
		action="append",
		default=[ Path(librosa.example('nutcracker', hq=True)) ],
		help= "A single music file to process. Default: librosa.example('nutcracler')."
	)

	ret.add_argument(
		'-o',
		dest='out_dir',
		type=Path,
		action='store',
		default=Path.cwd().joinpath('music_features'),
		help="Output directory to be created. Every input files gets their music features stored in equally named folders. Default: 'CWD/music_features'"
	)

	return ret.parse_args(args=argv[1:])

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
