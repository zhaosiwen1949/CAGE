#!/usr/bin/env bash

# Floor-plan post-processing: align room walls to the point-cloud wall mask and
# close inter-room gaps.
# Usage: bash tools/align_floorplan.sh [polys.json] [point_cloud.ply] [output_dir]

POLYS="${1:-infer_out/xinghewan_da3_mvs_polys.json}"
PLY="${2:-data/custom/xinghewan_da3_mvs.ply}"
OUT="${3:-infer_out}"

python align_floorplan.py \
               --polys="${POLYS}" \
               --ply="${PLY}" \
               --output_dir="${OUT}"
