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
                        choices=('otsu', 'percentile'),
                        help="How to threshold density into a wall mask.")
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

    Otsu / the percentile are computed over NON-ZERO density only: most of the
    256x256 grid is empty (0), and including it would collapse the threshold to
    'zero vs non-zero' rather than 'wall vs floor/furniture'.
    """
    dens_u8 = np.clip(density * 255.0, 0, 255).astype(np.uint8)
    nz = dens_u8[dens_u8 > 0]
    if nz.size == 0:
        return np.zeros_like(dens_u8), 0.0
    if method == 'otsu':
        thr, _ = cv2.threshold(nz.reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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


def _cluster_and_snap(coords, members_span, mask, axis, snap_tol):
    """Cluster group-representative coordinates and pick a snapped target for each.

    coords         : list of group representative coordinates (float)
    members_span   : list of (lo, hi) integer extents of each group's member coords
    mask           : binary wall mask (H, W)
    axis           : 'x' -> snap to a column (use mask[:, c]); 'y' -> a row.
    Returns a list `target[i]` (int pixel) aligned with `coords`.
    """
    H, W = mask.shape
    order = sorted(range(len(coords)), key=lambda i: coords[i])

    # single-linkage clustering of the sorted representatives within snap_tol
    clusters = []                    # list of lists of group indices
    for i in order:
        if clusters and coords[i] - coords[clusters[-1][-1]] <= snap_tol:
            clusters[-1].append(i)
        else:
            clusters.append([i])

    target = [0] * len(coords)
    pad = int(np.ceil(snap_tol))
    for cl in clusters:
        lo = min(members_span[i][0] for i in cl)
        hi = max(members_span[i][1] for i in cl)
        mean_c = int(round(sum(coords[i] for i in cl) / len(cl)))

        limit = W if axis == 'x' else H
        a = max(0, lo - pad)
        b = min(limit - 1, hi + pad)
        # wall-pixel count per candidate line within the cluster's span
        best_line, best_count = mean_c, -1
        for c in range(a, b + 1):
            cnt = int(mask[:, c].sum()) if axis == 'x' else int(mask[c, :].sum())
            # prefer higher count; break ties toward the cluster mean
            if cnt > best_count or (cnt == best_count and abs(c - mean_c) < abs(best_line - mean_c)):
                best_count, best_line = cnt, c
        snapped = best_line if best_count > 0 else mean_c
        for i in cl:
            target[i] = int(np.clip(snapped, 0, limit - 1))
    return target


def align_rooms(rooms_px, mask, angle_tol=8.0, snap_tol=5.0):
    """Snap room polygons onto the wall mask and straighten near-axis walls.

    rooms_px : list of (N_i, 2) int arrays, pixel [col, row], polygon not closed.
    Returns a new list of (M_i, 2) int arrays with shared/straightened walls.
    """
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
        reps = [float(np.mean([verts[v][coord_idx] for v in groups[g]])) for g in gkeys]
        spans = [(min(verts[v][coord_idx] for v in groups[g]),
                  max(verts[v][coord_idx] for v in groups[g])) for g in gkeys]
        targets = _cluster_and_snap(reps, spans, mask, axis, snap_tol)
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
        # drop consecutive duplicates created by snapping
        dedup = []
        for pt in pts:
            if not dedup or dedup[-1] != pt:
                dedup.append(pt)
        if len(dedup) > 1 and dedup[0] == dedup[-1]:
            dedup.pop()
        if len(dedup) >= 3:
            out.append(np.asarray(dedup, dtype=np.int32))
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


def save_overlay(mask, rooms_px, out_path, scale=3):
    """Aligned polygon outlines drawn over the wall mask, for visual QA."""
    H, W = mask.shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[mask > 0] = (90, 90, 90)                         # walls in grey
    for poly in rooms_px:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
    canvas = cv2.resize(canvas, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
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

    # 1) rebuild a density map pixel-aligned with the polygons (stored yaw + min/max)
    print('Rebuilding aligned density map from {} ...'.format(args.ply))
    xyz, _lo, _hi = preprocess_xyz(args.ply, up_axis=up_axis, pct_low=args.pct_low,
                                   pct_high=args.pct_high, crop_iqr_k=args.crop_iqr_k)
    if applied_yaw != 0.0:
        xyz = rotate_floor_plane(xyz, applied_yaw)
    density = density_fixed_norm(xyz, min_coords, max_coords, image_res)

    # 2) wall mask
    mask, thr_u8 = density_to_mask(density, method=args.mask_method,
                                   percentile=args.mask_percentile)
    print('  mask: method={} threshold={:.1f}/255 wall_pixels={}'.format(
        args.mask_method, thr_u8, int(mask.sum())))

    # 3) align
    aligned = align_rooms(rooms_px, mask, angle_tol=args.angle_tol, snap_tol=args.snap_tol)

    # 4) outputs
    rooms_out = []
    for r in aligned:
        world = pixel_to_world(r, min_coords, max_coords)
        rooms_out.append({
            'pixel': r.astype(int).tolist(),
            'world_mm': world.tolist(),
            'world_m': (world / 1000.).tolist(),
        })
    result = {
        'num_rooms': len(rooms_out),
        'rooms': rooms_out,
        'normalization': norm,
        'align_info': {
            'source_polys': os.path.basename(args.polys),
            'mask_method': args.mask_method,
            'mask_percentile': args.mask_percentile,
            'mask_threshold_u8': thr_u8,
            'angle_tol_deg': args.angle_tol,
            'snap_tol_px': args.snap_tol,
        },
    }
    json_path = os.path.join(out_dir, '{}_aligned_polys.json'.format(name))
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    floorplan = plot_floorplan_with_regions(aligned, scale=1000)
    cv2.imwrite(os.path.join(out_dir, '{}_aligned_floorplan.png'.format(name)), floorplan)
    cv2.imwrite(os.path.join(out_dir, '{}_mask.png'.format(name)), (mask * 255).astype(np.uint8))
    save_density_hist(density, thr_u8, args.mask_method,
                      os.path.join(out_dir, '{}_density_hist.png'.format(name)))
    save_overlay(mask, aligned, os.path.join(out_dir, '{}_aligned_overlay.png'.format(name)))

    print('Wrote:')
    for suffix in ('_aligned_polys.json', '_aligned_floorplan.png', '_mask.png',
                   '_density_hist.png', '_aligned_overlay.png'):
        print('  {}'.format(os.path.join(out_dir, name + suffix)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CAGE floor-plan alignment', parents=[get_args_parser()])
    main(parser.parse_args())
