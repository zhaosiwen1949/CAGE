#!/usr/bin/env bash

# Evaluate an aligned floor-plan prediction against a realsee ground-truth
# layout folder (room_layout.json + openings_gt.json).
# Usage: bash tools/eval_floorplan.sh [aligned_polys.json] [gt_dir] [output_dir]

PRED="${1:-infer_out/xinghewan_da3_mvs_aligned_polys.json}"
GT_DIR="${2:-data/custom/xinghewan_floorplan}"
OUT="${3:-infer_out}"

python eval_floorplan.py \
               --pred="${PRED}" \
               --gt_dir="${GT_DIR}" \
               --output_dir="${OUT}"
