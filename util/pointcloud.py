"""Shared point-cloud -> density-map pipeline for CAGE.

Extracted from infer_pointcloud.py so that both the inference script and the
floor-plan alignment post-processor (align_floorplan.py) share one numerically
identical implementation of "read .ply -> reorder up-axis -> crop outliers ->
yaw-correct -> project to a 256x256 density map".

Deliberately free of torch / model / plotting dependencies: only numpy plus the
two data_preprocess helpers (read_scene_pc, generate_density). The inference
script keeps the torch tensor wrapping; the alignment script re-projects using a
FIXED normalization recovered from the JSON so its density map is pixel-aligned
with the already-predicted polygons.
"""

import os
import sys

import numpy as np

# This file lives in <REPO>/util/, so the repo root is two levels up. Replicates
# the sys.path setup from infer_pointcloud.py (lines 27-35): stru3d_utils.py runs
# `sys.path.append('../data_preprocess')` at import time, so inserting the
# absolute paths first makes its bare `from common_utils import ...` resolve
# regardless of cwd.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'data_preprocess'))
sys.path.insert(0, os.path.join(REPO, 'data_preprocess', 'stru3d'))

from common_utils import read_scene_pc                      # noqa: E402
from stru3d_utils import generate_density                   # noqa: E402


def reorder_up_axis(xyz, up_axis):
    """Reorder columns so the vertical (up) axis is the 3rd column.

    generate_density projects xyz[:, :2] as the floor plane and ignores xyz[:, 2].
    Structured3D is z-up, so for a z-up cloud no change is needed. For a y-up cloud
    the floor plane is (x, z), so we map [x, y, z] -> [x, z, y]; for x-up it is
    (y, z) -> [y, z, x].
    """
    if up_axis == 'z':
        return xyz
    if up_axis == 'y':
        return xyz[:, [0, 2, 1]]
    if up_axis == 'x':
        return xyz[:, [1, 2, 0]]
    raise ValueError('Unknown up_axis: {}'.format(up_axis))


def floor_hflip_needed(up_axis):
    """Whether the floor projection must be mirrored left-right to read as a
    conventional top-down plan.

    reorder_up_axis moves the up-axis to column 2 via a column permutation whose
    parity sets the floor-plane handedness: z-up is the identity [0,1,2] and x-up
    is the 3-cycle [1,2,0] (both EVEN), but y-up is the swap [0,2,1] (ODD). The
    odd swap flips handedness, so a y-up cloud projects to a MIRROR image -- the
    floor is effectively seen "from below" -- relative to the z-up convention the
    model was trained on. Flipping that density map left-right (np.fliplr) puts it
    back into the standard top-down orientation. z-up / x-up need no flip.
    """
    return up_axis == 'y'


def _projection_sharpness(coords_1d, lo, hi, bins=256):
    """Peakiness of a 1D point distribution: sum of squared normalized histogram.

    Maximal when points concentrate into a few bins, i.e. when walls parallel to
    this axis collapse onto shared coordinates. The histogram range is FIXED by
    the caller (not derived from the data) so that (a) outliers cannot coarsen the
    bins and (b) the bounding box growing under rotation cannot bias the score.
    """
    h, _ = np.histogram(coords_1d, bins=bins, range=(lo, hi))
    total = h.sum()
    if total == 0:
        return 0.0
    h = h.astype(np.float64) / total
    return float(np.sum(h * h))


def estimate_yaw(xyz, search_deg=45.0, step_deg=0.5, bins=256, max_points=100000,
                 wall_band=(20.0, 80.0)):
    """Estimate the yaw (rotation about the vertical axis) that best axis-aligns
    the floor plane, using the Manhattan-world assumption.

    For each candidate angle we rotate the floor-plane points and score how sharply
    they project onto the x and y axes; the best angle makes walls parallel to the
    image axes. Two robustness measures matter for real (furnished, noisy) clouds:

      * Only mid-height "wall" points are scored. The floor and ceiling slabs
        project top-down to filled areas whose marginal histograms are broad and
        nearly rotation-invariant, swamping the thin, rotation-sensitive wall
        lines. `wall_band` gives the height percentiles kept (default 20-80).
      * A single fixed histogram range is shared by every candidate angle (see
        `_projection_sharpness`).

    Expects the reordered cloud (column 2 = up axis). Returns the angle to
    ROTATE BY to correct.
    """
    # keep mid-height wall points; drop the floor/ceiling slabs
    h = xyz[:, 2]
    h_lo, h_hi = np.percentile(h, wall_band[0]), np.percentile(h, wall_band[1])
    band = (h >= h_lo) & (h <= h_hi)
    pts = xyz[band][:, :2]
    if len(pts) < 100:                      # band too thin -> fall back to all
        pts = xyz[:, :2]
    pts = np.asarray(pts, dtype=np.float64)

    if len(pts) > max_points:
        # subsample for speed; orientation is a global property
        idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
        pts = pts[idx]

    # center, then fix one histogram range for every angle: rotation about the
    # center keeps points within +/- their max radius, so this range never clips.
    pts = pts - pts.mean(axis=0, keepdims=True)
    radius = float(np.max(np.hypot(pts[:, 0], pts[:, 1]))) if len(pts) else 0.0
    if radius <= 0:
        return 0.0

    best_angle, best_score = 0.0, -1.0
    angles = np.arange(-search_deg, search_deg + 1e-9, step_deg)
    for a in angles:
        r = np.deg2rad(a)
        c, s = np.cos(r), np.sin(r)
        x = pts[:, 0] * c - pts[:, 1] * s
        y = pts[:, 0] * s + pts[:, 1] * c
        score = (_projection_sharpness(x, -radius, radius, bins) +
                 _projection_sharpness(y, -radius, radius, bins))
        if score > best_score:
            best_score, best_angle = score, a
    return best_angle


def rotate_floor_plane(xyz, angle_deg):
    """Rotate the floor-plane columns (0, 1) about the vertical axis by angle_deg."""
    r = np.deg2rad(angle_deg)
    c, s = np.cos(r), np.sin(r)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    out = xyz.astype(np.float64).copy()
    out[:, :2] = xyz[:, :2].astype(np.float64) @ rot.T
    return out.astype(xyz.dtype)


def preprocess_xyz(ply_path, up_axis='y', pct_low=2.0, pct_high=98.0, crop_iqr_k=3.0):
    """Read a .ply, reorder to (floor-plane, floor-plane, up) and reject outliers.

    Mirrors the pre-rotation part of infer_pointcloud.load_input exactly. Returns
    the cropped, reordered cloud plus the FULL-cloud (pre-crop) percentile extents
    `lo`/`hi` (used downstream to record floor/ceiling).

    Reject MVS/photogrammetry flyaway points WITHOUT clipping the room:
      * height axis: keep the [pct_low, pct_high] band (1D -> cannot chamfer the
        top-down footprint; also removes floor-noise / sky flyaways).
      * floor plane: a RADIAL Tukey fence about the median centre. A radius test
        is rotation-invariant, so it never chamfers the yawed room's corners.
    """
    points = read_scene_pc(ply_path)
    xyz = points[:, :3].astype(np.float32)
    xyz = reorder_up_axis(xyz, up_axis)

    lo = np.percentile(xyz, pct_low, axis=0).astype(np.float64)
    hi = np.percentile(xyz, pct_high, axis=0).astype(np.float64)

    keep_h = (xyz[:, 2] >= lo[2]) & (xyz[:, 2] <= hi[2])
    cx, cy = np.median(xyz[:, 0]), np.median(xyz[:, 1])
    rad = np.hypot(xyz[:, 0] - cx, xyz[:, 1] - cy)
    rq1, rq3 = np.percentile(rad, 25), np.percentile(rad, 75)
    keep_xy = rad <= rq3 + crop_iqr_k * (rq3 - rq1)
    core = keep_h & keep_xy
    n_before = len(xyz)
    if int(core.sum()) >= 100:
        xyz = xyz[core]
    print('  cropped {} -> {} points ({:.1f}% kept; height pct [{:g}, {:g}], '
          'radial IQR k={:g})'.format(n_before, len(xyz),
          100.0 * len(xyz) / max(n_before, 1), pct_low, pct_high, crop_iqr_k))
    return xyz, lo, hi


def resolve_yaw(xyz, align=True, rotation_deg=None, search_deg=45.0, step_deg=0.5):
    """Pick the yaw to apply: explicit override, else Manhattan estimate, else 0."""
    if rotation_deg is not None:
        return float(rotation_deg)
    if align:
        return estimate_yaw(xyz, search_deg=search_deg, step_deg=step_deg)
    return 0.0


def density_from_xyz(xyz_rot, width=256, height=256, density_gain=1.0, hflip=False):
    """Project a (yaw-corrected) cloud to a [0,1] density map, recomputing the
    normalization from the data. Mirrors infer_pointcloud.load_input's projection
    step (generate_density + optional contrast gain).

    `hflip` mirrors the finished density image left-right (np.fliplr) to undo the
    from-below projection of an odd-permutation up-axis (see floor_hflip_needed).
    This is a pure image-space flip: the returned `norm` (min/max in the un-flipped
    ps frame) is unchanged, and pixel_to_world(..., hflip=True) inverts the column
    so world coordinates -- and thus polys_to_3d -- stay identical."""
    density, norm = generate_density(xyz_rot, width=width, height=height)
    # Contrast boost: generate_density normalizes by the single busiest cell, so a
    # few very dense cells push the walls into the low grey range. Multiply by a
    # gain and re-clamp to [0, 1]; gain == 1.0 leaves the map unchanged.
    if density_gain != 1.0:
        density = np.clip(density * float(density_gain), 0.0, 1.0)
    if hflip:
        density = np.ascontiguousarray(np.fliplr(density))
    return density, norm


def density_fixed_norm(xyz_rot, min_coords, max_coords, image_res=(256, 256), hflip=False):
    """Project a (yaw-corrected) cloud to a [0,1] density map using a FIXED
    normalization (min/max) instead of recomputing it from the data.

    Replicates generate_density's ps-frame transform and binning (stru3d_utils.py
    :24-59) but with the caller-supplied min/max. Passing the min_coords/max_coords
    stored in a *_polys.json yields a density map on the exact same pixel grid as
    the polygons in that JSON, so a wall mask thresholded from it is pixel-aligned
    with the predicted rooms.

    `hflip` mirrors the finished map left-right (np.fliplr), matching
    density_from_xyz so the alignment mask lines up with polygons expressed in the
    same flipped (top-down) frame.
    """
    ps = xyz_rot.astype(np.float64) * -1.0
    ps[:, 0] *= -1.0
    ps[:, 1] *= -1.0                        # ps[:,0]=+x, ps[:,1]=+y (see generate_density)

    min_coords = np.asarray(min_coords, dtype=np.float64)
    max_coords = np.asarray(max_coords, dtype=np.float64)
    image_res = np.asarray(image_res)

    span = (max_coords[None, :2] - min_coords[None, :2])
    coords = np.round((ps[:, :2] - min_coords[None, :2]) / span * image_res[None])
    coords = np.minimum(np.maximum(coords, np.zeros_like(image_res)), image_res - 1)
    coords = coords.astype(np.int32)

    density = np.zeros((int(image_res[1]), int(image_res[0])), dtype=np.float32)
    uniq, counts = np.unique(coords, return_counts=True, axis=0)
    uniq = uniq.astype(np.int32)
    density[uniq[:, 1], uniq[:, 0]] = counts
    peak = float(density.max())
    if peak > 0:
        density = density / peak
    if hflip:
        density = np.ascontiguousarray(np.fliplr(density))
    return density


def pixel_to_world(poly_px, min_coords, max_coords, hflip=False):
    """Inverse-project 256-space pixel corners to world (density-projection frame).

    generate_density maps world x/y -> [0,1] via (p - min)/(max - min) then to
    pixels by *image_res (256); engine decodes corners with *255, so we invert with
    /255 to match the actual pixel values. The 255-vs-256 discrepancy is a sub-0.4%
    scale ambiguity inherent to the codebase.

    `hflip` must match the flag used to render the density map: np.fliplr on the
    256-wide image maps column c -> 255 - c, so we decode (255 - col) here. This
    exactly cancels the flip, leaving world_mm identical to the un-flipped case --
    so polys_to_3d keeps overlaying on the original .ply with no change.
    """
    mn = np.asarray(min_coords, dtype=np.float64)
    mx = np.asarray(max_coords, dtype=np.float64)
    poly = np.asarray(poly_px, dtype=np.float64)
    col = (255. - poly[:, 0]) if hflip else poly[:, 0]
    wx = mn[0] + (col / 255.) * (mx[0] - mn[0])
    wy = mn[1] + (poly[:, 1] / 255.) * (mx[1] - mn[1])
    return np.stack([wx, wy], axis=1)
