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
  5. Re-project the snapped pixels to world coords and write:
       {name}_aligned_polys.json, {name}_aligned_floorplan.png,
       {name}_mask.png, {name}_density_hist.png, {name}_aligned_overlay.png

This script has NO torch / model dependency; it only reuses the point-cloud ->
density pipeline from util.pointcloud (shared with infer_pointcloud.py).
"""

import argparse
import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')                                       # headless / no display
import matplotlib.pyplot as plt                             # noqa: E402

from util.plot_utils import plot_floorplan_with_regions     # noqa: E402
from util.pointcloud import (                               # noqa: E402
    rotate_floor_plane,
    preprocess_xyz,
    density_fixed_norm,
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

    # 4) outputs
    rooms_out = []
    for r in aligned:
        world = pixel_to_world(r, min_coords, max_coords, hflip=dst_hflip)
        rooms_out.append({
            'pixel': r.astype(int).tolist(),
            'world_mm': world.tolist(),
            'world_m': (world / 1000.).tolist(),
        })
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
        },
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
    save_density_hist(density, thr_u8, args.mask_method,
                      os.path.join(out_dir, '{}_density_hist.png'.format(name)))
    save_overlay(mask, aligned, os.path.join(out_dir, '{}_aligned_overlay.png'.format(name)),
                 label_rooms=not args.no_room_labels)

    print('Wrote:')
    for suffix in ('_aligned_polys.json', '_aligned_floorplan.png', '_mask.png',
                   '_density_hist.png', '_aligned_overlay.png'):
        print('  {}'.format(os.path.join(out_dir, name + suffix)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CAGE floor-plan alignment', parents=[get_args_parser()])
    main(parser.parse_args())
