#!/usr/bin/env bash

set -x

#rm -i $(dirname "$@")/*.png

track_length=$(ffprobe -i "$@" -show_entries format=duration 2>&1 | sed -n "s/duration=//p")
track_length=$(awk "BEGIN {print int($track_length * 100)}")

fixed_filter_modes="mode=combined:color=moreland:scale=log:fscale=log:saturation=5:gain=.1:legend=0"

ffmpeg \
	-i "$@"\
	-lavfi showspectrumpic="$track_length"x500:"$fixed_filter_modes" \
	"$@"_spec.png

