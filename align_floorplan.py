"""
Floor-plan post-processing: align room walls and close inter-room gaps.

`infer_pointcloud.py` predicts each room polygon independently, so adjacent
rooms leave gaps and many walls sit a few degrees off horizontal/vertical
(see *_pred_floorplan.png). This script snaps the walls onto the real wall
positions recovered from the point cloud and makes near-axis walls exactly
axis-aligned, so neighbouring rooms share one wall coordinate and the gaps
vanish.

Pipeline:
  1. Read {name}_polys.json (room polygons in 256x256 pixel space) + its .ply.
  2. Rebuild a density map that is PIXEL-ALIGNED with those polygons, by reusing
     the stored normalization (applied_yaw_deg + min/max_coords) instead of
     re-estimating anything (util.pointcloud.density_fixed_norm).
  3. Threshold it (Otsu on non-zero density, or a percentile) -> a WALL MASK
     (bright density = walls).
  4. Align: classify each edge as near-horizontal / near-vertical / diagonal;
     union-find the endpoints that must share an x (vertical edges) or a y
     (horizontal edges); cluster those shared coordinates across all rooms
     within a pixel tolerance; snap each cluster onto the nearest wall line in
     the mask (or the cluster mean if the mask is silent there). Diagonal-only
     vertices keep their original coordinate on the unconstrained axis.
  5. Split: the model often merges several small rooms into one polygon. A
     SECOND mask built from a near-ceiling height band (walls reach the
     ceiling; furniture / bar counters do not, and door openings are closed by
     their lintels) exposes the interior partition walls. A room is cut along
     such a wall only under strict evidence -- the wall must span nearly the
     whole room chord and reach both boundaries -- so single rooms with tall
     clutter (kitchen duct/cabinets) are never split. Recursive; --no_split
     turns it off.
  6. Openings: classify every wall position by the VERTICAL distribution of
     points at the wall plane (and in front of it) into wall / sill / occluded /
     open-doorway / no-data, then read off doors and passages as the open runs.
     Exterior openings are dropped (this pass keeps interior doors/passages);
     a connectivity backstop guarantees every room reaches a neighbour. See the
     detect_openings section below. --no_openings turns it off.
  7. Re-project the snapped pixels to world coords and write:
       {name}_aligned_polys.json, {name}_aligned_floorplan.png,
       {name}_mask.png, {name}_split_mask.png, {name}_density_hist.png,
       {name}_aligned_overlay.png, {name}_openings.png

This script has NO torch / model dependency; it only reuses the point-cloud ->
density pipeline from util.pointcloud (shared with infer_pointcloud.py).
"""

import argparse
import json
import os

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
import matplotlib
matplotlib.use('Agg')                                       # headless / no display
import matplotlib.pyplot as plt                             # noqa: E402

from matplotlib.patches import Polygon as MplPolygon         # noqa: E402

from util.plot_utils import plot_floorplan_with_regions     # noqa: E402
from util.pointcloud import (                               # noqa: E402
    rotate_floor_plane,
    preprocess_xyz,
    density_fixed_norm,
    float_pixels,
    estimate_floor_ceiling,
    pixel_to_world,
    floor_hflip_needed,
)


def get_args_parser():
    parser = argparse.ArgumentParser('CAGE floor-plan alignment post-processor',
                                     add_help=False)
    parser.add_argument('--polys', required=True, type=str,
                        help='Path to a {name}_polys.json produced by infer_pointcloud.py.')
    parser.add_argument('--ply', required=True, type=str,
                        help='Path to the .ply point cloud that produced --polys.')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Where to write outputs (default: same dir as --polys).')

    # Crop params: must match what generated --polys, so the rebuilt density map
    # contains the same points. Yaw and min/max come from the JSON, not re-estimated.
    parser.add_argument('--pct_low', default=2.0, type=float,
                        help='Lower height percentile for outlier rejection (match infer).')
    parser.add_argument('--pct_high', default=98.0, type=float,
                        help='Upper height percentile for outlier rejection (match infer).')
    parser.add_argument('--crop_iqr_k', default=3.0, type=float,
                        help='Radial Tukey-fence multiplier for floor-plane crop (match infer).')

    # Mask threshold
    parser.add_argument('--mask_method', default='otsu', type=str,
                        choices=('otsu', 'knee', 'percentile'),
                        help="How to threshold density into a wall mask. otsu: cleanest "
                             "walls on bimodal histograms; knee: chord-distance knee of "
                             "the sorted density curve, thicker but more continuous walls "
                             "(use when walls come out broken); percentile: manual.")
    parser.add_argument('--mask_percentile', default=80.0, type=float,
                        help="Percentile of NON-ZERO density used when "
                             "--mask_method=percentile (>= this -> wall).")

    # Alignment tolerances (pixels / degrees, in the 256 grid)
    parser.add_argument('--angle_tol', default=8.0, type=float,
                        help="An edge within this many degrees of horizontal/vertical "
                             "is straightened to that axis; steeper edges stay diagonal.")
    parser.add_argument('--snap_tol', default=5.0, type=float,
                        help="Wall coordinates (x of vertical edges, y of horizontal "
                             "edges) within this many pixels are clustered and snapped "
                             "to one shared line -- this is what closes inter-room gaps.")
    parser.add_argument('--collapse_diag_len', default=20.0, type=float,
                        help="Collapse any run of consecutive diagonal edges whose TOTAL "
                             "length is <= this many pixels into a right-angle corner "
                             "(0 = off). Removes tiny chamfers the model hallucinates on "
                             "corners AND straightens near-axis short edges that sit just "
                             "beyond --angle_tol (so they get snapped instead of drifting "
                             "and causing inter-room overlaps). Long real slanted walls "
                             "(octagon, etc.) are kept. Default 20.")
    parser.add_argument('--spike_angle_deg', default=60.0, type=float,
                        help="Cleanup: a vertex whose interior angle is below this AND "
                             "whose opening (distance between its two neighbours) is at "
                             "most --spike_max_gap px is a needle spike and gets removed. "
                             "Acute corners do not occur in floor plans. 0 = off.")
    parser.add_argument('--spike_max_gap', default=10.0, type=float,
                        help="Max opening (px) for the needle-spike rule; keeps genuine "
                             "wide wedge shapes safe from the angle test.")
    parser.add_argument('--collinear_tol', default=2.0, type=float,
                        help="Cleanup: drop a vertex whose perpendicular distance to the "
                             "chord through its two neighbours is <= this many pixels. "
                             "Removes near-collinear noise and the tips of shallow "
                             "backtrack notches (e.g. on diagonal walls). 0 = only drop "
                             "strictly collinear points.")
    parser.add_argument('--wall_min_run', default=5, type=int,
                        help="When snapping a wall to the mask, only pixels in a "
                             "continuous run of at least this length count as wall. "
                             "Rejects dashed / broken columns (clutter) that are not "
                             "real walls; raise it to be stricter about continuity.")

    # Room splitting (interior partition walls from a near-ceiling band)
    parser.add_argument('--no_split', action='store_true',
                        help="Disable splitting merged rooms along interior partition "
                             "walls detected in the near-ceiling structural mask "
                             "(splitting is ON by default).")
    parser.add_argument('--split_band_lo', default=0.75, type=float,
                        help="Lower bound of the near-ceiling band, as a fraction of the "
                             "cropped height range. The band must sit BELOW the ceiling "
                             "plane itself (a horizontal plane projects everywhere) and "
                             "above furniture / door tops. Band-sweep on xinghewan: "
                             "lo in [0.65, 0.82] all give the same correct cuts; lower "
                             "pulls in mid-height furniture lines, higher thins the "
                             "walls. 0.75 centres that safe zone.")
    parser.add_argument('--split_band_hi', default=0.95, type=float,
                        help="Upper bound of the near-ceiling band (fraction of height "
                             "range). Keep < 1.0: the topmost slice holds the ceiling "
                             "plane remnants (per-room dropped ceilings sit at different "
                             "heights) and floods the mask with blobs. Safe zone on "
                             "xinghewan: [0.92, 0.98].")
    parser.add_argument('--split_mask_percentile', default=80.0, type=float,
                        help="Percentile of non-zero band density used to threshold the "
                             "structural mask (band statistics are not bimodal, so Otsu "
                             "is unreliable here).")
    parser.add_argument('--split_min_cover', default=0.5, type=float,
                        help="A cut requires the partition wall's continuity-aware pixel "
                             "count (runs >= --wall_min_run) to cover at least this "
                             "fraction of the room chord along the cut line. Works with "
                             "the door-aware gap rules; a free-standing kitchen "
                             "duct/cabinet line fails the end-anchoring instead.")
    parser.add_argument('--split_end_gap', default=3, type=int,
                        help="A cut requires the wall pixels to reach within this many "
                             "pixels of BOTH room boundaries along the cut line.")
    parser.add_argument('--split_door_min', default=7, type=int,
                        help="Interior gaps in the cut line are only legal if they are "
                             "mask noise (<= 2 px) or door-sized: [door_min, door_max] "
                             "px. A 3..6 px opening (~0.3-0.6 m) does not exist in a "
                             "real wall (it is a shower screen / clutter line) and "
                             "rejects the cut.")
    parser.add_argument('--split_door_max', default=24, type=int,
                        help="Upper bound (px) of a door/passage gap in the cut line; "
                             "wider holes mean the 'wall' is not a real partition.")
    parser.add_argument('--split_main_cover', default=0.4, type=float,
                        help="Cross-check: the cut line must ALSO reach this continuity-"
                             "aware cover in the MAIN full-height mask. A real partition "
                             "is a full-height wall and shows in both masks; a curtain "
                             "box / dropped-ceiling edge lives only near the ceiling and "
                             "a wardrobe front is broken by clutter at lower heights. "
                             "0 disables the cross-check.")
    parser.add_argument('--split_min_size', default=8, type=int,
                        help="Minimum bbox side (px) of every sub-room a cut produces; "
                             "cuts creating thinner slivers are rejected.")

    # --- Opening (door / passage) detection, from per-position vertical profiles
    parser.add_argument('--no_openings', action='store_true',
                        help="Skip door / passage detection (openings are detected "
                             "by default and written to _aligned_polys.json + "
                             "_openings.png).")
    # Height zones as fractions of the detected floor->ceiling span (~2.7 m). The
    # floor-plane noise tail reaches ~0.18, so the sill zone starts at 0.20.
    parser.add_argument('--zone_floor', nargs=2, type=float, default=[-0.08, 0.18],
                        help="Floor-plane band: seeing the floor through a gap proves "
                             "the position was scanned (open doorway).")
    parser.add_argument('--zone_low', nargs=2, type=float, default=[0.20, 0.33],
                        help="Sill band: wall here but open above = window/sill.")
    parser.add_argument('--zone_mid', nargs=2, type=float, default=[0.40, 0.72],
                        help="Mid band: doors, windows and passages are all open here.")
    parser.add_argument('--zone_top', nargs=2, type=float, default=[0.78, 0.96],
                        help="Lintel band (below the ceiling plane).")
    parser.add_argument('--wall_tol', default=1.5, type=float,
                        help="Half-width (px) of the wall-plane slab (~0.15 m).")
    parser.add_argument('--front_tol', default=5.0, type=float,
                        help="Furniture-in-front search half-width (px) for occlusion.")
    parser.add_argument('--open_rel_thr', default=0.35, type=float,
                        help="A band counts as occupied at >= this fraction of the wall "
                             "line's own wall level (75th pct of mid-band density). "
                             "Relative, so weak walls / curtains / frame-leakage separate "
                             "by ratio not absolute count.")
    parser.add_argument('--open_min_pts', default=5, type=int,
                        help="A (position, band) cell below this many points is empty.")
    parser.add_argument('--open_min_wall_dens', default=60.0, type=float,
                        help="Minimum wall level for a line to be judged; weaker lines "
                             "are all no-data.")
    parser.add_argument('--top_open_thr', default=0.12, type=float,
                        help="A real opening is open to the ceiling SOMEWHERE (per-run "
                             "minimum top-band ratio below this). A solid wall with only "
                             "a mid-height scan gap keeps wall above along the whole run "
                             "and is rejected. Real doorways dip to ~0.08, wall gaps stay "
                             ">=0.16; 0.12 sits in that gap.")
    parser.add_argument('--floor_min_pts', default=5, type=int,
                        help="Absolute floor-band count proving a position was scanned "
                             "(open doorway). Absolute because doorway floor is dimmer "
                             "than a wall and would fail a relative test.")
    parser.add_argument('--open_hole_min', default=6, type=int,
                        help="Minimum opening width (px, ~0.6 m) to report.")
    parser.add_argument('--door_min', default=7, type=int,
                        help="Openings in [door_min, door_max] px are doors; wider ones "
                             "are passages / openings.")
    parser.add_argument('--door_max', default=24, type=int)
    parser.add_argument('--keep_exterior_openings', action='store_true',
                        help="Keep openings on exterior walls (dropped by default; this "
                             "pass focuses on interior doors/passages).")
    parser.add_argument('--no_ensure_connectivity', action='store_true',
                        help="Do not recover a door for rooms left with no interior "
                             "opening (connectivity backstop is on by default).")

    parser.add_argument('--no_room_labels', action='store_true',
                        help="Do not draw room index numbers on the floorplan / overlay "
                             "images (labels are drawn by default for easier QA).")
    parser.add_argument('--no_floor_hflip', action='store_true',
                        help="Disable the left-right flip that un-mirrors a y-up floor "
                             "projection. By default the output is emitted top-down "
                             "(matching infer_pointcloud.py); z/x-up are never flipped. "
                             "Legacy JSONs without an 'hflip' marker are converted "
                             "automatically.")
    return parser


# ---------------------------------------------------------------------------
# Union-Find (disjoint set) over vertex indices
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:            # path compression
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------
def density_to_mask(density, method='otsu', percentile=80.0):
    """Threshold a [0,1] density map into a binary wall mask (bright = wall).

    All statistics are computed over NON-ZERO density only: most of the
    256x256 grid is empty (0), and including it would collapse the threshold to
    'zero vs non-zero' rather than 'wall vs floor/furniture'.

    Methods:
      otsu       -- maximize between-class variance; cleanest walls when the
                    non-zero histogram is clearly bimodal.
      knee       -- knee of the sorted-descending density curve, found as the
                    point farthest below the chord joining the curve's ends
                    (Kneedle). Derivative-free and parameter-free; picks a more
                    permissive threshold than Otsu -> thicker but more
                    CONTINUOUS walls. Prefer it when the histogram is skewed /
                    unimodal or when Otsu leaves broken wall lines.
      percentile -- fixed percentile of non-zero density (manual fallback).
    """
    dens_u8 = np.clip(density * 255.0, 0, 255).astype(np.uint8)
    nz = dens_u8[dens_u8 > 0]
    if nz.size == 0:
        return np.zeros_like(dens_u8), 0.0
    if method == 'otsu':
        thr, _ = cv2.threshold(nz.reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'knee':
        curve = np.sort(nz.astype(np.float64))[::-1]        # rank -> value, descending
        if curve[0] == curve[-1]:
            thr = float(curve[0])
        else:
            x = np.linspace(0.0, 1.0, curve.size)
            y = (curve - curve[-1]) / (curve[0] - curve[-1])
            thr = float(curve[int(np.argmax((1.0 - x) - y))])
    else:
        thr = float(np.percentile(nz, percentile))
    mask = (dens_u8 >= thr).astype(np.uint8)
    return mask, float(thr)


# ---------------------------------------------------------------------------
# Alignment core (operates purely in 256 pixel space)
# ---------------------------------------------------------------------------
def _classify_edge(p, q, angle_tol):
    """Return 'H', 'V' or 'D' for the edge p->q given the near-axis tolerance."""
    dx = float(q[0] - p[0])
    dy = float(q[1] - p[1])
    if dx == 0.0 and dy == 0.0:
        return None
    adx, ady = abs(dx), abs(dy)
    # acute angle to the x-axis; small -> horizontal, near 90 -> vertical
    ang = np.degrees(np.arctan2(ady, adx))
    if ang <= angle_tol:
        return 'H'
    if ang >= 90.0 - angle_tol:
        return 'V'
    return 'D'


def _simplify_polygon(pts, spike_angle_deg=60.0, spike_max_gap=10.0, collinear_tol=2.0):
    """Remove duplicate, (near-)collinear, backtracking and needle-spike vertices.

    Repeatedly drops, until a fixed point is reached:
      1. consecutive duplicate points;
      2. any vertex B whose perpendicular distance to the chord A--C is at most
         collinear_tol pixels (|cross(B-A, C-B)| / |C-A|). This covers exact
         forward-collinear mid points, exact reversal apexes (zero-area spurs),
         AND the NEAR-collinear tips of a backtrack notch -- e.g. a diagonal
         wall that juts out and comes back within a couple of pixels, which the
         exact test (== 0) and the diagonal-collapse pass both miss. Removing
         such a tip moves the outline by <= collinear_tol px. A/C coincident
         (a pure spur to the same point) is treated as distance 0.
      3. needle spikes: the unsigned angle at B between BA and BC is below
         spike_angle_deg AND the opening |AC| is at most spike_max_gap pixels.
         Acute corners do not occur in real floor plans, but the angle test
         alone would also hit a genuine wedge-shaped room, so the narrow-gap
         guard restricts it to thin needles (outward or inward).
         Pass spike_angle_deg=0 to disable rule 3, collinear_tol=0 to keep rule
         2 exact (drop only strictly collinear points).

    Returns a list of (x, y) tuples; may have fewer than 3 points if the
    input was degenerate.
    """
    pts = [(int(p[0]), int(p[1])) for p in pts]
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        dedup = [pt for i, pt in enumerate(pts) if pt != pts[i - 1]]   # cyclic dedup
        if len(dedup) != len(pts):
            pts, changed = dedup, True
            continue
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            ac = float(np.hypot(c[0] - a[0], c[1] - a[1]))
            perp = 0.0 if ac <= 1e-9 else abs(cross) / ac      # dist from B to chord A--C
            if perp <= collinear_tol:
                pts.pop(i)
                changed = True
                break
            if spike_angle_deg > 0.0:
                u = (a[0] - b[0], a[1] - b[1])
                v = (c[0] - b[0], c[1] - b[1])
                cos_b = ((u[0] * v[0] + u[1] * v[1])
                         / (np.hypot(*u) * np.hypot(*v)))
                ang_b = float(np.degrees(np.arccos(np.clip(cos_b, -1.0, 1.0))))
                if ang_b < spike_angle_deg and ac <= spike_max_gap:
                    pts.pop(i)
                    changed = True
                    break
    return pts


def _collapse_short_diagonals(poly, angle_tol, max_len):
    """Collapse short runs of consecutive diagonal edges into right-angle corners.

    RoomFormer sometimes chamfers a corner with one or two tiny oblique edges
    (prediction noise). A maximal run of consecutive 'D' edges whose TOTAL path
    length is <= max_len is replaced by the corner implied by its two flanking
    axis edges:
      V ... H / H ... V (perpendicular flanks) -> a single corner point
      V ... V / H ... H (parallel flanks)      -> a perpendicular 2-point segment
    Long runs (real slanted walls, e.g. an octagon room) are left untouched.
    The synthesized edges are near-axis, so the later cluster/snap stage
    straightens them and pulls them onto the wall mask like any other wall.
    """
    pts = [(int(p[0]), int(p[1])) for p in np.asarray(poly)]
    pts = [pt for i, pt in enumerate(pts) if pt != pts[i - 1]]   # cyclic dedup
    m = len(pts)
    if m < 3:
        return np.asarray(poly, dtype=np.int32)
    kinds = [_classify_edge(pts[k], pts[(k + 1) % m], angle_tol) for k in range(m)]
    n_axis = sum(1 for k in kinds if k in ('H', 'V'))
    if 'D' not in kinds or n_axis < 2:       # nothing to collapse / nothing to anchor on
        return np.asarray(pts, dtype=np.int32)

    replacement = [None] * m                 # run start vertex -> points to insert there
    dropped = [False] * m
    collapsed = 0
    for s in range(m):
        # maximal cyclic D-run starting at edge s
        if kinds[s] != 'D' or kinds[(s - 1) % m] == 'D':
            continue
        e, run_len, n_edges = s, 0.0, 0
        while kinds[e % m] == 'D':
            p, q = pts[e % m], pts[(e + 1) % m]
            run_len += float(np.hypot(q[0] - p[0], q[1] - p[1]))
            n_edges += 1
            e += 1
        if run_len > max_len:
            continue
        a, b = s, (s + n_edges) % m          # run spans vertices a .. b (cyclic, inclusive)
        run_vids = [(a + t) % m for t in range(n_edges + 1)]
        kp, kn = kinds[(a - 1) % m], kinds[b]
        va, vb = pts[a], pts[b]
        if kp == 'V' and kn == 'H':
            repl = [(va[0], vb[1])]
        elif kp == 'H' and kn == 'V':
            repl = [(vb[0], va[1])]
        elif kp == 'V' and kn == 'V':        # parallel flanks: bridge at the mean level
            yc = int(round(np.mean([pts[v][1] for v in run_vids])))
            repl = [(va[0], yc), (vb[0], yc)]
        else:                                # 'H' and 'H'
            xc = int(round(np.mean([pts[v][0] for v in run_vids])))
            repl = [(xc, va[1]), (xc, vb[1])]
        for v in run_vids:
            dropped[v] = True
        replacement[a] = repl
        collapsed += 1

    if collapsed == 0:
        return np.asarray(pts, dtype=np.int32)
    out = []
    for k in range(m):
        if replacement[k] is not None:
            out.extend(replacement[k])
        elif not dropped[k]:
            out.append(pts[k])
    out = [pt for i, pt in enumerate(out) if pt != out[i - 1]]   # cyclic dedup
    if len(out) < 3:
        return np.asarray(pts, dtype=np.int32)
    return np.asarray(out, dtype=np.int32)


def _wall_score(seg, min_run):
    """Continuity-aware wall score for a candidate line segment.

    Returns (denoised, total):
      denoised = number of wall pixels that belong to a CONTINUOUS run of length
                 >= min_run. Dashed / broken columns (furniture, clutter) are made
                 of tiny runs and score ~0, so they cannot win over a genuine wall.
      total    = raw wall-pixel count (secondary key; also the graceful fallback
                 when no candidate has a long-enough run).
    A real wall keeps its long segments (and sums several of them across a
    doorway), so denoised rewards genuine walls without over-favouring whichever
    parallel line merely happens to be the single longest.
    """
    s = (np.asarray(seg) > 0).astype(np.int8)
    total = int(s.sum())
    if total == 0:
        return 0, 0
    d = np.diff(np.concatenate(([0], s, [0])))
    runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    denoised = int(runs[runs >= min_run].sum())
    return denoised, total


def _cluster_and_snap(coords, members_span, perp_span, mask, axis, snap_tol, wall_min_run=5):
    """Cluster group-representative coordinates and pick a snapped target for each.

    coords         : list of group representative coordinates (float)
    members_span   : list of (lo, hi) extents of each group's member coords on the
                     SNAP axis (defines the search window along that axis)
    perp_span      : list of (lo, hi) extents of each group on the PERPENDICULAR
                     axis -- the span the wall actually covers (its two endpoints)
    mask           : binary wall mask (H, W)
    axis           : 'x' -> snap to a column (use mask[:, c]); 'y' -> a row.
    Returns a list `target[i]` (int pixel) aligned with `coords`.

    The wall-pixel count for a candidate line is taken ONLY over the cluster's
    perpendicular span, not the whole row/column. A horizontal wall shared by a
    couple of rooms occupies a limited x-range; summing the full width would let
    unrelated walls elsewhere on that row (e.g. the other half of the building)
    dominate the argmax and pull the snap onto the wrong line.
    """
    H, W = mask.shape
    ng = len(coords)

    # Cluster two groups together only if they are BOTH close on the snap axis
    # (|rep_i - rep_j| <= snap_tol) AND overlapping/touching on the perpendicular
    # axis. Two walls at a similar coordinate but disjoint perpendicular spans are
    # different walls in different parts of the plan (e.g. left vs right half of
    # the building) and must not be forced onto one shared line -- otherwise a
    # 1-D chain on the coordinate alone would merge the whole floor's walls at a
    # given level and drag the snap toward whichever half has more wall pixels.
    uf = UnionFind(ng)
    for i in range(ng):
        for k in range(i + 1, ng):
            if abs(coords[i] - coords[k]) > snap_tol:
                continue
            gap = max(perp_span[i][0], perp_span[k][0]) - min(perp_span[i][1], perp_span[k][1])
            if gap <= snap_tol:                    # overlap (gap<=0) or a small bridge
                uf.union(i, k)
    clusters_map = {}
    for i in range(ng):
        clusters_map.setdefault(uf.find(i), []).append(i)
    clusters = list(clusters_map.values())

    target = [0] * len(coords)
    pad = int(np.ceil(snap_tol))
    for cl in clusters:
        lo = min(members_span[i][0] for i in cl)
        hi = max(members_span[i][1] for i in cl)
        mean_c = int(round(sum(coords[i] for i in cl) / len(cl)))
        # perpendicular extent the wall spans: restrict the pixel count to here
        p0 = max(0, min(perp_span[i][0] for i in cl))
        p1 = max(p0, max(perp_span[i][1] for i in cl))

        limit = W if axis == 'x' else H
        p_limit = H if axis == 'x' else W
        p1 = min(p_limit - 1, p1)
        a = max(0, lo - pad)
        b = min(limit - 1, hi + pad)
        # Score each candidate line by a CONTINUITY-AWARE count (local to the
        # wall's perpendicular span): pixels in runs >= wall_min_run only. A real
        # wall is a continuous line; a dashed / broken column (furniture, clutter)
        # can match the raw count of a genuine wall while clearly not being one, so
        # it must not win. Continuity is used to REJECT such lines, not to maximize
        # continuity outright -- otherwise the snap would jump to whichever parallel
        # line in the window happens to be the single longest, even if it is a
        # different wall. Among wall-like candidates, more wall pixels wins, then
        # (raw count as fallback when nothing is long enough), then nearness to mean.
        best_line, best_score = mean_c, (-1, -1)
        for c in range(a, b + 1):
            seg = mask[p0:p1 + 1, c] if axis == 'x' else mask[c, p0:p1 + 1]
            score = _wall_score(seg, wall_min_run)
            if score > best_score or (score == best_score
                                      and abs(c - mean_c) < abs(best_line - mean_c)):
                best_score, best_line = score, c
        snapped = best_line if (best_score[0] > 0 or best_score[1] > 0) else mean_c
        for i in cl:
            target[i] = int(np.clip(snapped, 0, limit - 1))
    return target


def align_rooms(rooms_px, mask, angle_tol=8.0, snap_tol=5.0, collapse_diag_len=0.0,
                spike_angle_deg=60.0, spike_max_gap=10.0, collinear_tol=2.0,
                wall_min_run=5):
    """Snap room polygons onto the wall mask and straighten near-axis walls.

    rooms_px : list of (N_i, 2) int arrays, pixel [col, row], polygon not closed.
    Returns a new list of (M_i, 2) int arrays with shared/straightened walls.
    """
    # 0) optionally collapse tiny chamfer runs into right-angle corners first,
    #    so the synthesized edges take part in the clustering/snapping below
    if collapse_diag_len > 0.0:
        rooms_px = [_collapse_short_diagonals(p, angle_tol, collapse_diag_len)
                    for p in rooms_px]

    # 1) flatten vertices into a global index space
    offsets, verts = [], []
    for poly in rooms_px:
        offsets.append(len(verts))
        for pt in poly:
            verts.append([int(pt[0]), int(pt[1])])
    offsets.append(len(verts))
    n = len(verts)
    if n == 0:
        return [np.asarray(p, dtype=np.int32) for p in rooms_px]

    ufx, ufy = UnionFind(n), UnionFind(n)
    x_con = [False] * n          # vertex is endpoint of >=1 vertical edge -> x snaps
    y_con = [False] * n          # vertex is endpoint of >=1 horizontal edge -> y snaps

    # 2) classify edges, union shared coordinates
    for r, poly in enumerate(rooms_px):
        base, m = offsets[r], len(poly)
        for k in range(m):
            i = base + k
            j = base + (k + 1) % m
            kind = _classify_edge(verts[i], verts[j], angle_tol)
            if kind == 'V':                       # share x
                ufx.union(i, j)
                x_con[i] = x_con[j] = True
            elif kind == 'H':                     # share y
                ufy.union(i, j)
                y_con[i] = y_con[j] = True

    def resolve_axis(uf, constrained, coord_idx, axis):
        """Cluster + snap the shared coordinate (coord_idx: 0=x, 1=y) of every
        constrained vertex. Returns {vertex: new_coord} for constrained vertices."""
        groups = {}
        for v in range(n):
            if not constrained[v]:
                continue
            groups.setdefault(uf.find(v), []).append(v)
        gkeys = list(groups.keys())
        perp_idx = 1 - coord_idx                 # extent the wall covers (perp to snap axis)
        reps = [float(np.mean([verts[v][coord_idx] for v in groups[g]])) for g in gkeys]
        spans = [(min(verts[v][coord_idx] for v in groups[g]),
                  max(verts[v][coord_idx] for v in groups[g])) for g in gkeys]
        perp_spans = [(min(verts[v][perp_idx] for v in groups[g]),
                       max(verts[v][perp_idx] for v in groups[g])) for g in gkeys]
        targets = _cluster_and_snap(reps, spans, perp_spans, mask, axis, snap_tol,
                                    wall_min_run=wall_min_run)
        new_coord = {}
        for gi, g in enumerate(gkeys):
            for v in groups[g]:
                new_coord[v] = targets[gi]
        return new_coord

    new_x = resolve_axis(ufx, x_con, 0, 'x')
    new_y = resolve_axis(ufy, y_con, 1, 'y')

    # 3) rebuild polygons; keep original coord on any unconstrained axis
    out = []
    for r, poly in enumerate(rooms_px):
        base = offsets[r]
        pts = []
        for k in range(len(poly)):
            v = base + k
            x = new_x.get(v, verts[v][0])
            y = new_y.get(v, verts[v][1])
            pts.append((int(x), int(y)))
        # Straightening the flanking walls only now exposes short diagonals as
        # clean-corner chamfers, so collapse AGAIN post-snap (the pre-align pass
        # ran before the flanks became axis-aligned). Then simplify: snapping
        # makes duplicates / collinear runs / zero-area spurs / near-collinear
        # backtrack tips removable.
        if collapse_diag_len > 0.0:
            pts = [tuple(int(v) for v in p)
                   for p in _collapse_short_diagonals(pts, angle_tol, collapse_diag_len)]
        simplified = _simplify_polygon(pts, spike_angle_deg=spike_angle_deg,
                                       spike_max_gap=spike_max_gap,
                                       collinear_tol=collinear_tol)
        if len(simplified) >= 3:
            out.append(np.asarray(simplified, dtype=np.int32))
        else:
            out.append(np.asarray(poly, dtype=np.int32))   # too degenerate: keep original
    return out


# ---------------------------------------------------------------------------
# Room splitting (interior partition walls from a near-ceiling band)
# ---------------------------------------------------------------------------
def ceiling_band_mask(xyz_rot, min_coords, max_coords, image_res, hflip,
                      band_lo=0.80, band_hi=0.95, percentile=80.0):
    """Project only the near-ceiling points into a STRUCTURAL wall mask.

    Walls run all the way to the ceiling; furniture, bar counters and open door
    leaves stop well below it, and door OPENINGS are closed by their lintels.
    So a density map of just the near-ceiling band shows interior partition
    walls as continuous lines even across doorways, with the clutter gone --
    exactly the evidence needed to split merged rooms. The band must stay
    below the ceiling plane itself: the ceiling is a horizontal plane and
    would project onto every interior pixel, washing the map out.

    The band is expressed as fractions of the cropped height range (xyz_rot has
    already been percentile/radially cropped, so min/max are robust). Uses the
    same stored min/max_coords + hflip as the polygons -> pixel-aligned.
    Returns (mask, n_band_points, threshold_u8).
    """
    h = xyz_rot[:, 2]
    h0, h1 = float(h.min()), float(h.max())
    lo = h0 + band_lo * (h1 - h0)
    hi = h0 + band_hi * (h1 - h0)
    sel = (h >= lo) & (h <= hi)
    density = density_fixed_norm(xyz_rot[sel], min_coords, max_coords, image_res,
                                 hflip=hflip)
    mask, thr = density_to_mask(density, method='percentile', percentile=percentile)
    return mask, int(sel.sum()), thr


_SPLIT_NOISE_GAP = 2        # a hole this short (px) in the cut line is mask noise
_SPLIT_MAX_NOISE_GAPS = 1   # a real wall line is clean; more holes = clutter line


def _denoised_cover(wall, chord_len, min_run):
    """(cover, gaps): continuity-aware cover of a candidate line and the list of
    interior hole lengths between its first and last wall pixel."""
    widx = np.flatnonzero(wall)
    if widx.size == 0:
        return 0.0, None
    sub = (wall[widx[0]:widx[-1] + 1] > 0).astype(np.int8)
    d = np.diff(np.concatenate(([0], sub, [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = ends - starts
    cover = runs[runs >= min_run].sum() / float(chord_len)
    return float(cover), starts[1:] - ends[:-1]


def _best_split_line_axis(region, region_int, struct_mask, main_mask, axis,
                          min_cover, main_cover, end_gap, min_size,
                          wall_min_run, door_min, door_max):
    """Best door-aware full-span partition wall along one axis, or None.

    axis='x' scans vertical cut lines (columns), axis='y' horizontal (rows).
    The room chord at a line is restricted to STRICTLY INTERIOR pixels
    (region_int: room area at least min_size deep on BOTH sides of the line,
    along the cut axis): where the line rides on the room's own stepped
    boundary wall, that wall must not masquerade as partition evidence (a
    bay-window step does exactly that -- the cut line sits a couple of px
    inside the polygon, on the boundary wall's own thickness).
    A candidate line c qualifies only if, over that interior chord:
      * anchoring -- the first/last wall pixel reach within end_gap px of both
        chord ends: a partition wall is attached to the room boundary on both
        sides, a free-standing duct/cabinet/counter line on at most one;
      * coverage -- the continuity-aware count (runs >= wall_min_run, see
        _wall_score) covers >= min_cover of the chord;
      * door-aware gaps -- every interior hole in the line is either mask
        noise (<= _SPLIT_NOISE_GAP px, at most _SPLIT_MAX_NOISE_GAPS of them)
        or door-sized ([door_min, door_max] px, at most one: the lintel above
        a doorway is often NOT reconstructed by MVS, so a legal door hole must
        be tolerated -- but a hole too narrow for any door (shower screens,
        wardrobe fronts) or a second opening rejects the line, and so does a
        fragmented tail of many small holes (window / curtain-box clutter);
      * full height -- the line also reaches main_cover in the MAIN mask: a
        real partition is a floor-to-ceiling wall and shows at every height;
        a curtain box / dropped-ceiling edge exists only near the ceiling and
        a wardrobe front is broken by clutter below, so they fail here.
    Lines closer than min_size to the room's extent ends are not considered,
    so both sub-rooms keep at least that thickness. Returns (line, cover).
    """
    reg_l = region.T if axis == 'x' else region        # reg_l[c] = pixels on line c
    int_l = region_int.T if axis == 'x' else region_int
    msk_l = struct_mask.T if axis == 'x' else struct_mask
    main_l = main_mask.T if axis == 'x' else main_mask
    occ = np.flatnonzero(reg_l.sum(axis=1) > 0)
    if occ.size == 0:
        return None
    best = None
    lo = max(occ[0] + min_size, 1)
    hi = min(occ[-1] - min_size, reg_l.shape[0] - 2)
    for c in range(lo, hi + 1):
        interior = int_l[c]
        rows = np.flatnonzero(interior)
        if rows.size < max(min_size, wall_min_run):    # chord too short to judge
            continue
        wall = msk_l[c] * interior
        widx = np.flatnonzero(wall)
        if widx.size == 0:
            continue
        if widx[0] - rows[0] > end_gap or rows[-1] - widx[-1] > end_gap:
            continue                                   # not anchored on both ends
        cover, gaps = _denoised_cover(wall, rows.size, wall_min_run)
        if cover < min_cover:
            continue
        noise = gaps <= _SPLIT_NOISE_GAP
        door = (gaps >= door_min) & (gaps <= door_max)
        if np.count_nonzero(~noise & ~door) > 0:       # a hole no real wall has
            continue
        if np.count_nonzero(noise) > _SPLIT_MAX_NOISE_GAPS or np.count_nonzero(door) > 1:
            continue
        if main_cover > 0.0:
            cover_m, _ = _denoised_cover(main_l[c] * interior, rows.size, wall_min_run)
            if cover_m < main_cover:
                continue                               # not a full-height wall
        if best is None or cover > best[1]:
            best = (int(c), float(cover))
    return best


def _find_split_line(poly, struct_mask, main_mask, min_cover, main_cover,
                     end_gap, min_size, wall_min_run, door_min, door_max):
    """Strongest interior partition wall of a room: ('x'|'y', line, cover) | None."""
    H, W = struct_mask.shape
    region = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(region, [np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)], 1)
    best = None
    for axis in ('x', 'y'):
        # dilate 1 px PERPENDICULAR to the cut line so a wall wobbling between
        # two adjacent columns/rows still reads as one continuous line
        kernel = np.ones((1, 3), np.uint8) if axis == 'x' else np.ones((3, 1), np.uint8)
        dil = cv2.dilate(struct_mask, kernel)
        dil_main = cv2.dilate(main_mask, kernel)
        # interior = room area >= min_size deep on both sides ALONG the cut
        # axis (1-D erosion): mirrors the min-size rule per chord pixel and
        # keeps the room's own boundary walls out of the evidence
        ero_kernel = (np.ones((1, 2 * min_size + 1), np.uint8) if axis == 'x'
                      else np.ones((2 * min_size + 1, 1), np.uint8))
        region_int = cv2.erode(region, ero_kernel)
        cand = _best_split_line_axis(region, region_int, dil, dil_main, axis,
                                     min_cover, main_cover, end_gap, min_size,
                                     wall_min_run, door_min, door_max)
        if cand is not None and (best is None or cand[1] > best[2]):
            best = (axis, cand[0], cand[1])
    return best


def _cut_polygon(poly, axis, line):
    """Cut a pixel polygon by the axis line; both halves KEEP the line itself,
    so the two sub-rooms share the cut coordinate (the usual shared-wall
    convention -- zero-area overlap, no gap). Returns shapely Polygons."""
    P = ShapelyPolygon([(float(x), float(y)) for x, y in poly])
    if not P.is_valid:
        P = P.buffer(0)
    minx, miny, maxx, maxy = P.bounds
    if axis == 'x':
        halves = (shapely_box(minx - 1, miny - 1, line, maxy + 1),
                  shapely_box(line, miny - 1, maxx + 1, maxy + 1))
    else:
        halves = (shapely_box(minx - 1, miny - 1, maxx + 1, line),
                  shapely_box(minx - 1, line, maxx + 1, maxy + 1))
    pieces = []
    for half in halves:
        inter = P.intersection(half)
        for g in getattr(inter, 'geoms', [inter]):
            if g.geom_type == 'Polygon' and g.area > 1e-6:
                pieces.append(g)
    return pieces


def split_rooms(rooms_px, struct_mask, main_mask, min_cover=0.5, main_cover=0.4,
                end_gap=3, min_size=8, wall_min_run=5, door_min=7, door_max=24,
                spike_angle_deg=60.0, spike_max_gap=10.0,
                collinear_tol=2.0, max_depth=6):
    """Recursively split rooms along interior partition walls in struct_mask.

    Returns (new_rooms, parent_ids, records): new_rooms expand each input room
    into its sub-rooms in place (children sorted top-to-bottom, left-to-right),
    parent_ids[i] is the ORIGINAL index each output room came from, records
    lists every applied cut. A cut is abandoned (room kept whole) if any
    resulting piece is a sliver thinner than min_size or degenerates during
    simplification -- better to leave a merged room than to invent a bad one.
    """
    new_rooms, parent_ids, records = [], [], []
    for orig_idx, poly in enumerate(rooms_px):
        leaves = []
        queue = [(np.asarray(poly, dtype=np.int32), 0)]
        while queue:
            cur, depth = queue.pop(0)
            found = None
            if depth < max_depth and len(cur) >= 3:
                found = _find_split_line(cur, struct_mask, main_mask, min_cover,
                                         main_cover, end_gap, min_size,
                                         wall_min_run, door_min, door_max)
            children = []
            if found is not None:
                axis, line, cover = found
                pieces = _cut_polygon(cur, axis, line)
                ok = (len(pieces) >= 2
                      and all(min(g.bounds[2] - g.bounds[0],
                                  g.bounds[3] - g.bounds[1]) >= min_size
                              for g in pieces))
                if ok:
                    for g in pieces:
                        pts = [(int(round(x)), int(round(y)))
                               for x, y in list(g.exterior.coords)[:-1]]
                        pts = _simplify_polygon(pts, spike_angle_deg=spike_angle_deg,
                                                spike_max_gap=spike_max_gap,
                                                collinear_tol=collinear_tol)
                        if len(pts) >= 3:
                            children.append(np.asarray(pts, dtype=np.int32))
                    if len(children) < 2:              # degenerated: abandon the cut
                        children = []
                if children:
                    records.append({'parent': orig_idx, 'axis': axis,
                                    'line': int(line), 'cover': round(cover, 3)})
            if children:
                queue.extend((ch, depth + 1) for ch in children)
            else:
                leaves.append(cur)
        leaves.sort(key=lambda p: (int(p[:, 1].min()), int(p[:, 0].min())))
        new_rooms.extend(leaves)
        parent_ids.extend([orig_idx] * len(leaves))
    return new_rooms, parent_ids, records


# ---------------------------------------------------------------------------
# Opening (door / passage) detection via per-position vertical profiles
#
# The 2-D masks above cannot tell a doorway from an occluded / never-scanned
# wall stretch: both are empty at mid height. So openings are found from the
# VERTICAL distribution of points AT each wall position. For each position we
# count points at the wall plane (+-wall_tol px) in four height bands and count
# points just IN FRONT of the wall, then classify (see CLASS_MEANING):
#   W wall      mid band occupied at the wall plane
#   S sill      low occupied, mid empty (window / furniture-behind-a-door)
#   O occluded  wall plane empty but furniture in front at mid height -> wall
#   D doorway   wall plane empty low+mid, floor visible (position was scanned)
#   U no-data   nothing anywhere: never scanned, NOT evidence of a door
# Openings are the maximal non-W runs; a run is a real opening only if it opens
# to the ceiling somewhere (per-run min top-ratio < top_open_thr), which
# separates real doors from a solid wall that merely lost its mid-height band.
# ---------------------------------------------------------------------------
CLS = {'W': 0, 'S': 1, 'O': 2, 'D': 3, 'U': 4}
CLASS_MEANING = {'W': 'wall', 'S': 'sill', 'O': 'occluded', 'D': 'doorway',
                 'U': 'no-data'}


def runs_of(profile, value):
    """List of (start, end_exclusive) runs where profile == value."""
    p = (np.asarray(profile) == value).astype(np.int8)
    d = np.diff(np.concatenate(([0], p, [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def collect_wall_lines(rooms_px, line_tol=1):
    """Group axis-aligned polygon edges into shared wall lines.

    Returns a list of {axis: 'x'|'y', line: int, members: [(room, a, b)]} where
    a<=b is each edge's extent along the wall. Edges of different rooms on the
    same line (within line_tol) share one group, so a wall between two rooms is
    scanned once. Diagonal edges (bay-window arcs) are skipped -- out of scope.
    """
    edges = {'x': [], 'y': []}                     # axis -> [(line, a, b, room)]
    for ridx, poly in enumerate(rooms_px):
        m = len(poly)
        for k in range(m):
            p, q = poly[k], poly[(k + 1) % m]
            if p[0] == q[0] and p[1] != q[1]:      # vertical edge -> shares an x
                a, b = sorted((int(p[1]), int(q[1])))
                edges['x'].append((int(p[0]), a, b, ridx))
            elif p[1] == q[1] and p[0] != q[0]:    # horizontal edge -> shares a y
                a, b = sorted((int(p[0]), int(q[0])))
                edges['y'].append((int(p[1]), a, b, ridx))
    groups = []
    for axis in ('x', 'y'):
        es = sorted(edges[axis])
        used = [False] * len(es)
        for i, (line, a, b, r) in enumerate(es):
            if used[i]:
                continue
            member = [(r, a, b)]
            used[i] = True
            for j in range(i + 1, len(es)):
                lj, aj, bj, rj = es[j]
                if lj - line > line_tol:
                    break
                if not used[j]:
                    member.append((rj, aj, bj))
                    used[j] = True
            groups.append({'axis': axis, 'line': int(line), 'members': member})
    return groups


def build_label_raster(rooms_px, res=256):
    """Room-id raster: lab[row, col] = room index, or -1 for empty space."""
    lab = np.full((res, res), -1, dtype=np.int16)
    for i, p in enumerate(rooms_px):
        m = np.zeros((res, res), dtype=np.uint8)
        cv2.fillPoly(m, [p.reshape(-1, 1, 2)], 1)
        lab[m > 0] = i
    return lab


def wall_is_exterior(lab, axis, line, s, e, offs=(2, 3, 4), cov_thr=0.5):
    """True if a wall span borders empty space on at least one perpendicular side.

    Samples the room raster a few px off the wall on both sides over [s, e]; if
    either side's room-coverage fraction is below cov_thr the wall is on the
    building's outer boundary. Robust where counting overlapping room edges is
    not: two stacked rooms' edges can land on one line yet the wall is still
    exterior because the far side is empty.
    """
    H, W = lab.shape
    left, right = [], []
    for t in range(s, e + 1):
        if not (0 <= t < (H if axis == 'x' else W)):
            continue
        for dd in offs:
            if axis == 'x':
                left.append(lab[t, max(0, line - dd)] >= 0)
                right.append(lab[t, min(W - 1, line + dd)] >= 0)
            else:
                left.append(lab[max(0, line - dd), t] >= 0)
                right.append(lab[min(H - 1, line + dd), t] >= 0)
    lc = float(np.mean(left)) if left else 0.0
    rc = float(np.mean(right)) if right else 0.0
    return (lc < cov_thr) or (rc < cov_thr)


def classify_positions(zone_dens, zone_cnt, front_dens, front_cnt,
                       rel_thr, min_pts, min_wall_dens, floor_min_pts):
    """Classify each wall position W/S/O/D/U; also return per-position top-ratio.

    zone_dens / zone_cnt: (n_pos, 4) for [floor, low, mid, top] (density = count
    / band height-fraction); front_*: (n_pos,) mid-band points in front of the
    wall. Occupancy is judged RELATIVE to the line's own wall level (75th pct of
    mid-band density) so weakly scanned walls, curtains and frame leakage
    separate by ratio, not absolute counts.

    top_ratio (top-band density / wall level) is used at RUN level by the
    caller: position-level top does not separate real doors from wall scan gaps,
    but a run's MINIMUM top-ratio does (a doorway opens to the ceiling somewhere,
    a wall gap keeps wall above throughout). Priority: mid->W, low->S,
    front->O, floor-scanned(absolute)->D, else U.
    """
    n = zone_dens.shape[0]
    wall_level = float(np.percentile(zone_dens[:, 2], 75))
    if wall_level < min_wall_dens:
        return 'U' * n, np.zeros(n)
    thr = rel_thr * wall_level
    top_ratio = zone_dens[:, 3] / wall_level
    occ = (zone_dens >= thr) & (zone_cnt >= min_pts)
    floor_scanned = zone_cnt[:, 0] >= floor_min_pts
    fr = (front_dens >= thr) & (front_cnt >= min_pts)
    out = []
    for i in range(n):
        _, low_o, mid_o, _ = occ[i]
        if mid_o:
            out.append('W')
        elif low_o:
            out.append('S')
        elif fr[i]:
            out.append('O')
        elif floor_scanned[i]:
            out.append('D')
        else:
            out.append('U')
    return ''.join(out), top_ratio


def detect_openings(rooms_px, xyz_rot, hfrac, min_coords, max_coords, image_res,
                    hflip, args):
    """Detect interior doors / passages from per-position vertical profiles.

    Returns (openings, walls_debug, undecided). `openings` are interior doors /
    passages (exterior dropped unless args.keep_exterior_openings). `undecided`
    are pure no-data interior spans, kept only for the debug figure. `hfrac` is
    the height of every point as a fraction of the floor->ceiling span.
    """
    res = int(image_res[0])
    fcol, frow = float_pixels(xyz_rot, min_coords, max_coords, res, hflip)
    px2m = float((max_coords[0] - min_coords[0]) / (res - 1))
    lab = build_label_raster(rooms_px, res=res)
    groups = collect_wall_lines(rooms_px, line_tol=1)
    zones = [args.zone_floor, args.zone_low, args.zone_mid, args.zone_top]

    # pre-sort point indices by integer col and row for fast per-line slab gather
    icol = np.round(fcol).astype(np.int32)
    irow = np.round(frow).astype(np.int32)
    order_c, order_r = np.argsort(icol, kind='stable'), np.argsort(irow, kind='stable')
    col_sorted, row_sorted = icol[order_c], irow[order_r]

    def gather(axis, lo, hi):
        """Indices of points whose perpendicular integer coord is in [lo, hi]."""
        if axis == 'x':
            i0, i1 = np.searchsorted(col_sorted, [lo, hi + 1])
            return order_c[i0:i1]
        i0, i1 = np.searchsorted(row_sorted, [lo, hi + 1])
        return order_r[i0:i1]

    openings, walls_debug, undecided = [], [], []
    for g in groups:
        axis, line = g['axis'], g['line']
        a = min(m[1] for m in g['members'])
        b = max(m[2] for m in g['members'])
        n = b - a + 1
        if n < 2 * args.wall_min_run:
            continue
        perp = fcol if axis == 'x' else frow
        along = frow if axis == 'x' else fcol

        wt, ft = args.wall_tol, args.front_tol
        cand = gather(axis, int(np.floor(line - ft)), int(np.ceil(line + ft)))
        d = np.abs(perp[cand] - line)
        pos = np.round(along[cand]).astype(np.int32) - a
        inside = (pos >= 0) & (pos < n)
        hf = hfrac[cand]

        # count wall-plane points per position per height band
        zone_cnt = np.zeros((n, 4), dtype=np.int32)
        wall_sel = inside & (d <= wt)
        for zi, (f0, f1) in enumerate(zones):
            zsel = wall_sel & (hf >= f0) & (hf <= f1)
            np.add.at(zone_cnt[:, zi], pos[zsel], 1)
        # count mid-band points just IN FRONT of the wall (occlusion evidence)
        front_cnt = np.zeros(n, dtype=np.int32)
        fsel = (inside & (d > wt) & (d <= ft)
                & (hf >= args.zone_mid[0]) & (hf <= args.zone_mid[1]))
        np.add.at(front_cnt, pos[fsel], 1)

        band_h = np.array([f1 - f0 for f0, f1 in zones])
        zone_dens = zone_cnt / band_h[None, :]
        # front slab is wider than the wall slab: rescale to per-slab-width density
        front_dens = (front_cnt / (args.zone_mid[1] - args.zone_mid[0])
                      * (2 * wt) / (2 * (ft - wt)))
        cls, top_ratio = classify_positions(
            zone_dens, zone_cnt, front_dens, front_cnt, args.open_rel_thr,
            args.open_min_pts, args.open_min_wall_dens, args.floor_min_pts)

        cover_rooms = np.zeros(n, dtype=np.int8)       # how many room edges cover pos
        for r, ma, mb in g['members']:
            cover_rooms[ma - a:mb - a + 1] += 1

        walls_debug.append({'axis': axis, 'line': int(line),
                            'span': [int(a), int(b)],
                            'rooms': sorted({m[0] for m in g['members']}),
                            'classes': cls})

        arr = np.array([CLS[c] for c in cls])
        if (arr == CLS['W']).sum() < args.wall_min_run:
            continue                                    # no wall on this line at all
        for s, e in runs_of((arr == CLS['W']).astype(np.int8), 0):
            w = e - s
            if w < args.open_hole_min:
                continue
            if cover_rooms[s:e].min() < 1:
                continue                                # not on any room's edge
            sub = arr[s:e]
            frac = {k: float((sub == v).mean()) for k, v in CLS.items()}
            if frac['O'] >= 0.5:
                continue                                # occluded wall, not an opening
            # run-level lintel test: a genuine opening is open to the ceiling
            # SOMEWHERE; a solid wall with only a mid-height scan gap keeps wall
            # above along the whole run (min top-ratio stays high) -> reject.
            if float(top_ratio[s:e].min()) >= args.top_open_thr:
                continue
            exterior = wall_is_exterior(lab, axis, line, a + s, a + e - 1)
            rooms_here = sorted({r for r, ma, mb in g['members']
                                 if not (mb < a + s or ma > a + e - 1)})
            geom = {'axis': axis, 'line': int(line),
                    'span': [int(a + s), int(a + e - 1)],
                    'width_px': int(w), 'width_m': round(w * px2m, 2),
                    'rooms': rooms_here, 'exterior': bool(exterior),
                    'classes': cls[s:e]}

            if frac['U'] >= 0.6:
                undecided.append(geom)                  # never scanned: keep for debug only
                continue
            if exterior and not args.keep_exterior_openings:
                continue                                # this pass: interior only
            # interior S-dominant is NOT a window (no interior windows in a flat):
            # the "sill" is furniture right behind a door -> classify as a door.
            kind = 'door' if w <= args.door_max else 'passage'
            openings.append({'type': kind, **geom})
    return openings, walls_debug, undecided


def _side_room(lab, axis, line, pos, side, offs=(2, 3, 4)):
    """Most common room id `side` px off the wall at `pos` (-1 = empty space)."""
    H, W = lab.shape
    vals = []
    for dd in offs:
        c = line + side * dd
        if axis == 'x':
            vals.append(int(lab[min(max(pos, 0), H - 1), min(max(c, 0), W - 1)]))
        else:
            vals.append(int(lab[min(max(c, 0), H - 1), min(max(pos, 0), W - 1)]))
    vals = [v for v in vals if v >= 0]
    return max(set(vals), key=vals.count) if vals else -1


def ensure_connectivity(openings, walls_debug, rooms_px, lab, args, px2m):
    """Guarantee every room has >=1 interior opening to another room.

    A room isolated by the strict filtering (all its wall gaps read as wall, or
    its only opening was pure no-data) gets one door recovered: the best-scoring
    non-wall run on a wall it shares with a neighbour, scored by how open it
    looks (D through-floor > U no-data > S sill) and door-width fit. A room whose
    shared walls are entirely wall gets a nominal door at the midpoint of its
    longest shared wall. Recovered doors carry recovered='connectivity'|'nominal'.
    Returns the list of recovered openings (also appended to `openings`).
    """
    n = len(rooms_px)
    adj = {i: set() for i in range(n)}
    for op in openings:
        for i in op['rooms']:
            for j in op['rooms']:
                if i != j:
                    adj[i].add(j)

    recovered = []
    for r in range(n):
        if adj[r]:
            continue
        best = None                                    # (score, axis, line, s, e, w, other, how)
        for wd in walls_debug:
            if r not in wd['rooms']:
                continue
            axis, line, a = wd['axis'], wd['line'], wd['span'][0]
            arr = np.array([CLS[c] for c in wd['classes']])
            for s, e in runs_of((arr == CLS['W']).astype(np.int8), 0):
                w = e - s
                if w < args.open_hole_min:
                    continue
                mid = a + (s + e) // 2
                sides = {_side_room(lab, axis, line, mid, -1),
                         _side_room(lab, axis, line, mid, +1)}
                if r not in sides:
                    continue
                other = sorted(x for x in sides if x >= 0 and x != r)
                if not other:
                    continue                           # exterior side; need a neighbour
                sub = arr[s:e]
                fD = float((sub == CLS['D']).mean())
                fU = float((sub == CLS['U']).mean())
                fS = float((sub == CLS['S']).mean())
                fit = 1.0 if args.door_min <= w <= args.door_max else 0.4
                score = (fD + 0.5 * fU + 0.3 * fS) * fit
                how = 'connectivity' if (fD > 0 or fS > 0) else 'nominal'
                if best is None or score > best[0]:
                    best = (score, axis, line, a + s, a + e - 1, w, other[0], how)
        if best is None:
            # no openable run anywhere: nominal door at the longest shared wall
            longest = None
            for wd in walls_debug:
                if r not in wd['rooms'] or len(set(wd['rooms'])) < 2:
                    continue
                a, b = wd['span']
                if longest is None or (b - a) > (longest[3] - longest[2]):
                    longest = (wd['axis'], wd['line'], a, b)
            if longest is None:
                print('  ! room {} shares no interior wall; cannot recover'.format(r))
                continue
            axis, line, a, b = longest
            mid, hw = (a + b) // 2, args.door_min // 2
            other = sorted(x for x in {_side_room(lab, axis, line, mid, -1),
                                       _side_room(lab, axis, line, mid, +1)}
                           if x >= 0 and x != r)
            best = (0.0, axis, line, mid - hw, mid + hw, args.door_min,
                    other[0] if other else -1, 'nominal')

        score, axis, line, s_abs, e_abs, w, other, how = best
        op = {'type': 'door', 'recovered': how, 'axis': axis, 'line': int(line),
              'span': [int(s_abs), int(e_abs)], 'width_px': int(w),
              'width_m': round(w * px2m, 2),
              'rooms': sorted({r, other}) if other >= 0 else [r], 'exterior': False}
        openings.append(op)
        recovered.append(op)
        adj[r].add(other)
        if other >= 0:
            adj[other].add(r)
        print('  + room {} isolated -> recovered {} door {}={} {}..{} '
              '(links room {})'.format(r, how, axis, line, s_abs, e_abs, other))
    return recovered


# ---------------------------------------------------------------------------
# Debug figure
# ---------------------------------------------------------------------------
def save_density_hist(density, thr_u8, method, out_path):
    """Sorted (descending) non-zero density values with the chosen threshold marked."""
    dens_u8 = np.clip(density * 255.0, 0, 255).astype(np.uint8)
    nz = np.sort(dens_u8[dens_u8 > 0].astype(np.float64))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    if nz.size:
        ax.plot(np.arange(nz.size), nz, lw=1.2, color='#1f77b4', label='density (desc)')
        ax.axhline(thr_u8, color='#d62728', ls='--', lw=1.2,
                   label='{} threshold = {:.1f}'.format(method, thr_u8))
        n_wall = int((nz >= thr_u8).sum())
        ax.axvline(n_wall, color='#2ca02c', ls=':', lw=1.0,
                   label='wall pixels = {}'.format(n_wall))
    ax.set_xlabel('pixel rank (brightest first)')
    ax.set_ylabel('density value (0-255)')
    ax.set_title('Sorted non-zero density & mask threshold')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _label_anchor(poly, res=256):
    """Interior point of the polygon farthest from its edges (pole of
    inaccessibility, via distance transform). Keeps a room number inside the
    room even for concave L-shaped polygons where the vertex centroid would
    land outside. Returns (x, y) in the polygon's own pixel space."""
    m = np.zeros((res, res), dtype=np.uint8)
    cv2.fillPoly(m, [np.asarray(poly, dtype=np.int32)], 255)
    if int(m.sum()) == 0:                                   # degenerate -> centroid
        c = np.asarray(poly, dtype=np.float64).mean(axis=0)
        return int(round(c[0])), int(round(c[1]))
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    _, _, _, maxloc = cv2.minMaxLoc(dist)
    return int(maxloc[0]), int(maxloc[1])


def _draw_room_labels(image, rooms_px, src_res, fill, halo,
                      font_scale, thickness):
    """Draw each room's index (its position in rooms_px) at the room's interior
    anchor, scaled from src_res to the image size. `fill`/`halo` are colour
    tuples matching the image's channel count (3 for BGR, 4 for BGRA)."""
    s = image.shape[0] / float(src_res)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, poly in enumerate(rooms_px):
        ax, ay = _label_anchor(poly, res=src_res)
        text = str(idx)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        org = (int(round(ax * s)) - tw // 2, int(round(ay * s)) + th // 2)
        cv2.putText(image, text, org, font, font_scale, halo,
                    thickness + 4, cv2.LINE_AA)          # readability halo
        cv2.putText(image, text, org, font, font_scale, fill,
                    thickness, cv2.LINE_AA)
    return image


def save_overlay(mask, rooms_px, out_path, scale=3, label_rooms=True):
    """Aligned polygon outlines drawn over the wall mask, for visual QA."""
    H, W = mask.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[mask > 0] = (90, 90, 90)                         # walls in grey
    for poly in rooms_px:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
    canvas = cv2.resize(canvas, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    if label_rooms:
        _draw_room_labels(canvas, rooms_px, src_res=W,
                          fill=(0, 255, 255), halo=(0, 0, 0),
                          font_scale=0.9, thickness=2)      # yellow on dark
    cv2.imwrite(out_path, canvas)


_OPEN_COLORS = {'door': '#ff3b30', 'passage': '#00d26a'}
_CLS_RGB = {'W': (60, 60, 60), 'S': (58, 122, 254), 'O': (255, 165, 0),
            'D': (255, 59, 48), 'U': (200, 200, 200)}


def save_openings_debug(rooms_px, openings, walls_debug, undecided, out_path,
                        res=256):
    """Two-panel QA figure: detected openings (left) and the per-position
    vertical classification of every wall (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.6))

    ax = axes[0]
    for ridx, poly in enumerate(rooms_px):
        ax.add_patch(MplPolygon(poly, closed=True, fill=False,
                                edgecolor='0.55', linewidth=1.2))
        ax.annotate(str(ridx), poly.mean(axis=0), color='0.4', fontsize=9,
                    ha='center', va='center')
    for op in undecided:                               # no-scan-data spans
        (s, e), line = op['span'], op['line']
        xy = ([line, line], [s, e]) if op['axis'] == 'x' else ([s, e], [line, line])
        ax.plot(*xy, color='#9e9e9e', linewidth=2.5, linestyle=(0, (2, 2)),
                solid_capstyle='butt')
    for i, op in enumerate(openings):
        (s, e), line = op['span'], op['line']
        xy = ([line, line], [s, e]) if op['axis'] == 'x' else ([s, e], [line, line])
        recovered = 'recovered' in op
        color = '#d400d4' if recovered else _OPEN_COLORS[op['type']]
        ls = (0, (1, 1)) if recovered else '-'
        ax.plot(*xy, color=color, linewidth=4, linestyle=ls, solid_capstyle='butt')
        tx, ty = (line, (s + e) / 2) if op['axis'] == 'x' else ((s + e) / 2, line)
        ax.annotate(str(i), (tx, ty), color=color, fontsize=7, fontweight='bold',
                    ha='left', va='bottom')
    handles = [plt.Line2D([0], [0], color=_OPEN_COLORS['door'], lw=4, label='door'),
               plt.Line2D([0], [0], color=_OPEN_COLORS['passage'], lw=4, label='passage'),
               plt.Line2D([0], [0], color='#d400d4', lw=4, ls=(0, (1, 1)), label='recovered'),
               plt.Line2D([0], [0], color='#9e9e9e', lw=2.5, ls=(0, (2, 2)), label='no-data')]
    ax.legend(handles=handles, loc='lower left', fontsize=9)
    ax.set_title('detected openings (index = list order)')
    ax.set_xlim(8, res - 8); ax.set_ylim(res - 8, 16)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')

    ax = axes[1]
    canvas = np.full((res, res, 3), 255, dtype=np.uint8)
    for wd in walls_debug:
        a = wd['span'][0]
        for i, ch in enumerate(wd['classes']):
            if wd['axis'] == 'x':
                canvas[a + i, wd['line']] = _CLS_RGB[ch]
            else:
                canvas[wd['line'], a + i] = _CLS_RGB[ch]
    ax.imshow(canvas, interpolation='nearest')
    for poly in rooms_px:
        ax.add_patch(MplPolygon(poly, closed=True, fill=False,
                                edgecolor='0.8', linewidth=0.5))
    handles = [plt.Line2D([0], [0], color=np.array(c) / 255, lw=4,
                          label='{} {}'.format(k, CLASS_MEANING[k]))
               for k, c in _CLS_RGB.items()]
    ax.legend(handles=handles, loc='lower left', fontsize=8)
    ax.set_title('per-position vertical classification')
    ax.set_xlim(8, res - 8); ax.set_ylim(res - 8, 16)
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('door / passage detection (wall-plane vertical profiles)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    with open(args.polys, 'r') as f:
        data = json.load(f)
    norm = data['normalization']
    rooms_px = [np.asarray(r['pixel'], dtype=np.int32) for r in data['rooms']]
    min_coords = np.asarray(norm['min_coords'], dtype=np.float64)
    max_coords = np.asarray(norm['max_coords'], dtype=np.float64)
    image_res = np.asarray(norm.get('image_res', [256, 256]))
    up_axis = norm.get('up_axis', 'y')
    applied_yaw = float(norm.get('applied_yaw_deg', 0.0))

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.polys))
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.polys))[0]
    if name.endswith('_polys'):
        name = name[:-len('_polys')]

    # Frame handling: `src_hflip` is the frame the JSON polygons live in (legacy
    # JSONs predate the flip and lack the marker -> False); `dst_hflip` is the
    # frame we emit (top-down for y-up, matching infer_pointcloud.py). If they
    # differ we mirror the polygon columns so they line up with the flipped mask.
    src_hflip = bool(norm.get('hflip', False))
    dst_hflip = (not args.no_floor_hflip) and floor_hflip_needed(up_axis)
    if src_hflip != dst_hflip:
        w1 = int(image_res[0]) - 1                         # fliplr maps col c -> 255 - c
        rooms_px = [np.column_stack([w1 - p[:, 0], p[:, 1]]).astype(np.int32)
                    for p in rooms_px]
        print('  polygons converted to {} frame (hflip {} -> {})'.format(
            'top-down' if dst_hflip else 'raw', src_hflip, dst_hflip))

    # 1) rebuild a density map pixel-aligned with the polygons (stored yaw + min/max)
    print('Rebuilding aligned density map from {} ...'.format(args.ply))
    xyz, _lo, _hi = preprocess_xyz(args.ply, up_axis=up_axis, pct_low=args.pct_low,
                                   pct_high=args.pct_high, crop_iqr_k=args.crop_iqr_k)
    if applied_yaw != 0.0:
        xyz = rotate_floor_plane(xyz, applied_yaw)
    density = density_fixed_norm(xyz, min_coords, max_coords, image_res, hflip=dst_hflip)

    # 2) wall mask
    mask, thr_u8 = density_to_mask(density, method=args.mask_method,
                                   percentile=args.mask_percentile)
    print('  mask: method={} threshold={:.1f}/255 wall_pixels={}'.format(
        args.mask_method, thr_u8, int(mask.sum())))

    # 3) align
    aligned = align_rooms(rooms_px, mask, angle_tol=args.angle_tol, snap_tol=args.snap_tol,
                          collapse_diag_len=args.collapse_diag_len,
                          spike_angle_deg=args.spike_angle_deg,
                          spike_max_gap=args.spike_max_gap,
                          collinear_tol=args.collinear_tol,
                          wall_min_run=args.wall_min_run)
    if args.collapse_diag_len > 0.0:
        n_before = sum(len(p) for p in rooms_px)
        n_after = sum(len(p) for p in aligned)
        print('  collapse_diag_len={:.1f}px: {} -> {} vertices'.format(
            args.collapse_diag_len, n_before, n_after))

    # 3b) split merged rooms along interior partition walls: a near-ceiling
    #     band exposes them (lintels close the doorways, furniture drops out)
    struct_mask, splits = None, []
    parent_ids = list(range(len(aligned)))
    if not args.no_split:
        struct_mask, n_band, split_thr = ceiling_band_mask(
            xyz, min_coords, max_coords, image_res, dst_hflip,
            band_lo=args.split_band_lo, band_hi=args.split_band_hi,
            percentile=args.split_mask_percentile)
        n_rooms_before = len(aligned)
        aligned, parent_ids, splits = split_rooms(
            aligned, struct_mask, mask, min_cover=args.split_min_cover,
            main_cover=args.split_main_cover, end_gap=args.split_end_gap,
            min_size=args.split_min_size, wall_min_run=args.wall_min_run,
            door_min=args.split_door_min, door_max=args.split_door_max,
            spike_angle_deg=args.spike_angle_deg,
            spike_max_gap=args.spike_max_gap, collinear_tol=args.collinear_tol)
        print('  split: ceiling band [{:.2f},{:.2f}] ({} pts, thr={:.1f}/255): '
              '{} -> {} rooms'.format(args.split_band_lo, args.split_band_hi,
                                      n_band, split_thr, n_rooms_before,
                                      len(aligned)))
        for rec in splits:
            print('    room {} cut at {}={} (cover {:.0%})'.format(
                rec['parent'], rec['axis'], rec['line'], rec['cover']))
        if splits:
            # Cut lines are chosen per room from the struct mask, INDEPENDENTLY
            # of where neighbouring rooms' walls were already snapped. On a
            # thick wall band both picks are "on the wall" yet a few px apart
            # (e.g. a cut at y=68 next to a wall snapped to y=65), leaving a
            # staircase between rooms. A second alignment pass clusters the new
            # axis-aligned cut edges with those neighbouring coordinates
            # (within snap_tol) and snaps them onto ONE shared line.
            aligned = align_rooms(aligned, mask, angle_tol=args.angle_tol,
                                  snap_tol=args.snap_tol,
                                  collapse_diag_len=args.collapse_diag_len,
                                  spike_angle_deg=args.spike_angle_deg,
                                  spike_max_gap=args.spike_max_gap,
                                  collinear_tol=args.collinear_tol,
                                  wall_min_run=args.wall_min_run)
            print('  re-aligned {} rooms after splitting'.format(len(aligned)))

    # 3c) openings: detect interior doors / passages from per-position vertical
    #     profiles of the point cloud at each (final) wall, then guarantee every
    #     room reaches a neighbour.
    openings, undecided, walls_dbg = [], [], []
    if not args.no_openings:
        res = int(image_res[0])
        h = xyz[:, 2]
        floor_h, ceil_h, span_h = estimate_floor_ceiling(h)
        hfrac = (h - floor_h) / span_h                 # 0 = floor plane, 1 = ceiling
        px2m = float((max_coords[0] - min_coords[0]) / (res - 1))
        openings, walls_dbg, undecided = detect_openings(
            aligned, xyz, hfrac, min_coords, max_coords, image_res, dst_hflip, args)
        recovered = []
        if not args.no_ensure_connectivity:
            lab = build_label_raster(aligned, res=res)
            recovered = ensure_connectivity(openings, walls_dbg, aligned, lab,
                                            args, px2m)
        n_doors = sum(1 for o in openings if o['type'] == 'door')
        print('  openings: {} doors, {} passages ({} recovered for connectivity)'
              .format(n_doors, len(openings) - n_doors, len(recovered)))

    # 4) outputs
    split_parents = {rec['parent'] for rec in splits}
    rooms_out = []
    for i, r in enumerate(aligned):
        world = pixel_to_world(r, min_coords, max_coords, hflip=dst_hflip)
        room_out = {
            'pixel': r.astype(int).tolist(),
            'world_mm': world.tolist(),
            'world_m': (world / 1000.).tolist(),
        }
        if parent_ids[i] in split_parents:
            room_out['split_from'] = parent_ids[i]     # pre-split room index
        rooms_out.append(room_out)
    norm_out = dict(norm)
    norm_out['hflip'] = dst_hflip                          # record the emitted frame
    result = {
        'num_rooms': len(rooms_out),
        'rooms': rooms_out,
        'normalization': norm_out,
        'align_info': {
            'source_polys': os.path.basename(args.polys),
            'mask_method': args.mask_method,
            'mask_percentile': args.mask_percentile,
            'mask_threshold_u8': thr_u8,
            'angle_tol_deg': args.angle_tol,
            'snap_tol_px': args.snap_tol,
            'collapse_diag_len_px': args.collapse_diag_len,
            'spike_angle_deg': args.spike_angle_deg,
            'spike_max_gap_px': args.spike_max_gap,
            'collinear_tol_px': args.collinear_tol,
            'wall_min_run': args.wall_min_run,
            'hflip': dst_hflip,
            'split_enabled': not args.no_split,
            'split_band': [args.split_band_lo, args.split_band_hi],
            'split_mask_percentile': args.split_mask_percentile,
            'split_min_cover': args.split_min_cover,
            'split_main_cover': args.split_main_cover,
            'split_end_gap_px': args.split_end_gap,
            'split_door_px': [args.split_door_min, args.split_door_max],
            'split_min_size_px': args.split_min_size,
            'splits': splits,
            'openings_enabled': not args.no_openings,
            'open_zones': {'floor': args.zone_floor, 'low': args.zone_low,
                           'mid': args.zone_mid, 'top': args.zone_top},
            'open_rel_thr': args.open_rel_thr,
            'top_open_thr': args.top_open_thr,
            'open_door_px': [args.door_min, args.door_max],
            'openings_keep_exterior': args.keep_exterior_openings,
        },
        'openings': openings,
        'openings_undecided': undecided,               # no-scan-data spans (QA only)
    }
    json_path = os.path.join(out_dir, '{}_aligned_polys.json'.format(name))
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    floorplan = plot_floorplan_with_regions(aligned, scale=1000)
    if not args.no_room_labels:
        src_res = int(image_res[0])
        _draw_room_labels(floorplan, aligned, src_res=src_res,
                          fill=(0, 0, 0, 255), halo=(255, 255, 255, 255),
                          font_scale=1.4, thickness=3)      # black on pastel
    cv2.imwrite(os.path.join(out_dir, '{}_aligned_floorplan.png'.format(name)), floorplan)
    cv2.imwrite(os.path.join(out_dir, '{}_mask.png'.format(name)), (mask * 255).astype(np.uint8))
    if struct_mask is not None:
        cv2.imwrite(os.path.join(out_dir, '{}_split_mask.png'.format(name)),
                    (struct_mask * 255).astype(np.uint8))
    save_density_hist(density, thr_u8, args.mask_method,
                      os.path.join(out_dir, '{}_density_hist.png'.format(name)))
    save_overlay(mask, aligned, os.path.join(out_dir, '{}_aligned_overlay.png'.format(name)),
                 label_rooms=not args.no_room_labels)
    if not args.no_openings:
        save_openings_debug(aligned, openings, walls_dbg, undecided,
                            os.path.join(out_dir, '{}_openings.png'.format(name)),
                            res=int(image_res[0]))

    print('Wrote:')
    suffixes = ['_aligned_polys.json', '_aligned_floorplan.png', '_mask.png',
                '_density_hist.png', '_aligned_overlay.png']
    if struct_mask is not None:
        suffixes.insert(3, '_split_mask.png')
    if not args.no_openings:
        suffixes.append('_openings.png')
    for suffix in suffixes:
        print('  {}'.format(os.path.join(out_dir, name + suffix)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CAGE floor-plan alignment', parents=[get_args_parser()])
    main(parser.parse_args())
