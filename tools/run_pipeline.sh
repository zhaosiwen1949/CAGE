#!/usr/bin/env bash
#
# Batch pipeline: point cloud -> floor plan -> aligned floor plan.
#
# For every "<scene>_da3_mvs.ply" in an input folder, run inference
# (tools/infer_pointcloud.sh) then wall alignment + split + opening detection
# (tools/align_floorplan.sh), writing all products of each scene into
# <output_root>/<scene>/.
#
# Usage:
#   bash tools/run_pipeline.sh <folder> [output_root]
#     <folder>       directory holding point clouds named <scene>_da3_mvs.ply
#                    (relative to your current directory is fine)
#     [output_root]  where per-scene output folders go (default: infer_out)
#
# Example: data/custom/xinghewan_da3_mvs.ply -> infer_out/xinghewan/ containing
#          xinghewan_da3_mvs_polys.json, ..._aligned_polys.json, ..._openings.png
#
# NOTE: tools/infer_pointcloud.sh needs torch + the model checkpoint on a GPU
#       and CANNOT run on every machine; run this where that stack is available.

# Re-exec under bash if started with a POSIX shell (e.g. `sh tools/run_pipeline.sh`,
# where /bin/sh is dash): this script uses bash arrays, nullglob and pipefail,
# which dash rejects (e.g. "set: Illegal option -o pipefail").
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

set -euo pipefail

IN_DIR="${1:?usage: bash tools/run_pipeline.sh <folder> [output_root]}"
OUT_ROOT="${2:-infer_out}"

# Locate this script's dir (so the wrappers are found regardless of PWD) and the
# repo root (the wrappers call "python infer_pointcloud.py" relative to it).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

# Resolve input/output against the CALLER's PWD before we cd into the repo, so
# the folder arg may be relative to where the user ran the command.
if [[ ! -d "${IN_DIR}" ]]; then
  echo "error: input folder not found: ${IN_DIR}" >&2
  exit 1
fi
IN_DIR_ABS="$(cd "${IN_DIR}" && pwd)"
mkdir -p "${OUT_ROOT}"
OUT_ROOT_ABS="$(cd "${OUT_ROOT}" && pwd)"

# Gather the point clouds (no match -> empty array, not the literal glob).
shopt -s nullglob
plys=( "${IN_DIR_ABS}"/*_da3_mvs.ply )
shopt -u nullglob
if (( ${#plys[@]} == 0 )); then
  echo "error: no *_da3_mvs.ply files in ${IN_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"                      # python <script>.py resolves from here
echo "found ${#plys[@]} point cloud(s) in ${IN_DIR}"

for ply in "${plys[@]}"; do
  stem="$(basename "${ply}" .ply)"    # e.g. xinghewan_da3_mvs
  scene="${stem%_da3_mvs}"            # e.g. xinghewan
  out_dir="${OUT_ROOT_ABS}/${scene}"
  polys="${out_dir}/${stem}_polys.json"

  echo
  echo "==== [${scene}] ${ply} -> ${out_dir} ===="
  mkdir -p "${out_dir}"

  echo "-- inference --"
  bash "${SCRIPT_DIR}/infer_pointcloud.sh" "${ply}" "${out_dir}"

  echo "-- alignment --"
  bash "${SCRIPT_DIR}/align_floorplan.sh" "${polys}" "${ply}" "${out_dir}"

  echo "==== [${scene}] done ===="
done

echo
echo "all done -> ${OUT_ROOT}/"
