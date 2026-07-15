#!/usr/bin/env bash
#
# Batch evaluation: aligned floor-plan predictions -> metrics vs ground truth.
#
# Companion to tools/run_pipeline.sh. The full flow is:
#     bash tools/run_pipeline.sh <ply_folder> [output_root]   # predict
#     bash tools/run_eval.sh     [output_root] [gt_root]       # evaluate
#
# run_pipeline.sh writes one folder per scene under <output_root> (default
# "infer_out"), e.g. infer_out/xinghewan/xinghewan_da3_mvs_aligned_polys.json.
# This script walks those per-scene folders, matches each to its ground-truth
# layout folder <gt_root>/<scene>_floorplan (default gt_root "data/floorplan"),
# and runs tools/eval_floorplan.sh, writing _eval.json / _eval.txt /
# _eval_overlay.png back into the scene's own folder.
#
# Usage:
#   bash tools/run_eval.sh [output_root] [gt_root]
#     [output_root]  folder holding per-scene prediction subfolders
#                    (default: infer_out -- same as run_pipeline.sh's output)
#     [gt_root]      folder holding <scene>_floorplan GT subfolders
#                    (default: data/floorplan)
#
# Example: infer_out/xinghewan/  +  data/floorplan/xinghewan_floorplan/
#          -> infer_out/xinghewan/xinghewan_da3_mvs_eval.{json,txt}

# Re-exec under bash if started with a POSIX shell (dash rejects nullglob etc.).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

set -euo pipefail

OUT_ROOT="${1:-infer_out}"
GT_ROOT="${2:-data/floorplan}"

# Locate this script's dir (to find the eval wrapper) and the repo root (the
# wrapper calls "python eval_floorplan.py" relative to it).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

# Resolve the roots against the CALLER's PWD before we cd into the repo, so the
# args may be given relative to where the user ran the command.
if [[ ! -d "${OUT_ROOT}" ]]; then
  echo "error: prediction root not found: ${OUT_ROOT}" >&2
  exit 1
fi
OUT_ROOT_ABS="$(cd "${OUT_ROOT}" && pwd)"
if [[ ! -d "${GT_ROOT}" ]]; then
  echo "error: ground-truth root not found: ${GT_ROOT}" >&2
  exit 1
fi
GT_ROOT_ABS="$(cd "${GT_ROOT}" && pwd)"

cd "${REPO_DIR}"                      # python eval_floorplan.py resolves from here

# Gather the per-scene prediction subfolders (no match -> empty array).
shopt -s nullglob
subdirs=( "${OUT_ROOT_ABS}"/*/ )
shopt -u nullglob
if (( ${#subdirs[@]} == 0 )); then
  echo "error: no scene subfolders in ${OUT_ROOT}" >&2
  exit 1
fi

ok=(); skipped=(); failed=()
for sub in "${subdirs[@]}"; do
  sub="${sub%/}"                      # strip trailing slash
  scene="$(basename "${sub}")"

  # The prediction file run_pipeline.sh produced. Glob rather than assume the
  # "_da3_mvs" infix, so a scene named differently still resolves.
  shopt -s nullglob
  preds=( "${sub}"/*_aligned_polys.json )
  shopt -u nullglob
  if (( ${#preds[@]} == 0 )); then
    echo "-- [${scene}] skip: no *_aligned_polys.json in ${sub}"
    skipped+=( "${scene}" )
    continue
  fi
  pred="${preds[0]}"

  gt_dir="${GT_ROOT_ABS}/${scene}_floorplan"
  if [[ ! -d "${gt_dir}" ]]; then
    echo "-- [${scene}] skip: no GT folder ${GT_ROOT}/${scene}_floorplan"
    skipped+=( "${scene}" )
    continue
  fi

  echo
  echo "==== [${scene}] eval ===="
  echo "pred: ${pred}"
  echo "gt:   ${gt_dir}"
  if bash "${SCRIPT_DIR}/eval_floorplan.sh" "${pred}" "${gt_dir}" "${sub}"; then
    ok+=( "${scene}" )
  else
    echo "!! [${scene}] eval failed" >&2
    failed+=( "${scene}" )
  fi
done

echo
echo "==== summary ===="
echo "evaluated (${#ok[@]}): ${ok[*]:-<none>}"
echo "skipped   (${#skipped[@]}): ${skipped[*]:-<none>}"
echo "failed    (${#failed[@]}): ${failed[*]:-<none>}"

# Non-zero exit if any scene errored, so callers/CI can detect failure.
(( ${#failed[@]} == 0 ))
