#!/usr/bin/env bash

# Point cloud -> floor plan inference with CAGE.
# Usage: bash tools/infer_pointcloud.sh <path-to.ply OR directory> [output_dir]

INPUT="${1:-data/stru3d/point_cloud.ply}"
OUTPUT_DIR="${2:-infer_out}"

python infer_pointcloud.py \
               --backbone=swinv2_L_192_22k \
               --checkpoint=checkpoint/CAGE_stru3d_swinv2.pth \
               --input="${INPUT}" \
               --output_dir="${OUTPUT_DIR}" \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --device=cuda
