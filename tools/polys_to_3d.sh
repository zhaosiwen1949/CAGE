#!/usr/bin/env bash

# Floor-plan corners -> original point-cloud 3D world coordinates.
# Reads the *_polys.json produced by infer_pointcloud.py and writes
# {name}_world3d.json + {name}_world3d.obj (floor + ceiling wall rings).
#
# Usage: bash tools/polys_to_3d.sh <*_polys.json OR directory> [output_dir]
# up_axis and floor/ceiling are read from the JSON (recorded by
# infer_pointcloud.py). Pass --up_axis / --floor / --ceiling to override.

INPUT="${1:-infer_out}"
OUTPUT_DIR="${2:-${INPUT}}"

python polys_to_3d.py \
               --input="${INPUT}" \
               --output_dir="${OUTPUT_DIR}" \
               --units=mm
