"""Evaluate a predicted floor plan against a realsee ground-truth room layout.

Compares ``{name}_aligned_polys.json`` (from align_floorplan.py) with the
ground-truth layout in a folder like ``data/custom/xinghewan_floorplan/``:

  room_layout.json        per-pano wall lines (metres, y-up, one global
                          frame); the horizontal (ceiling-level) lines of
                          each pano trace the room polygon on the floor
                          plane (x, z).  INNER wall surfaces.
  rooms_centerline.json   wall-CENTERLINE room polygons (same convention as
                          the prediction) — the default GT (--gt_geometry).
  doors_windows.json      door/window/opening POSITIONS on the wall
                          centrelines, optional -- scored by
                          eval_doors_windows (the single opening evaluation).

Pipeline
  1. GT rooms:    per pano polygonize horizontal lines -> union per room name.
  2. Pred rooms:  pixel polygons -> metres via the pixel_to_world affine
                  (per-axis!  x and y spans differ) from `normalization`.
  3. Register:    both sides are Manhattan-aligned, so search 4 x 90-degree
                  rotations x mirror x scale grid; per candidate the optimal
                  translation comes from mask cross-correlation
                  (cv2.matchTemplate on rasterized union masks), coarse 5 cm
                  then a fine 1 cm pass around the winner.
  4. Metrics:     room-level IoU + precision/recall/F1 (greedy match at
                  --match_iou), corner precision/recall at --corner_tol
                  thresholds, boundary Chamfer distance, total-outline IoU
                  and area, and door/window position P/R/F1 against
                  doors_windows.json (when present).
  5. Merged-GT:   when one prediction swallowed several GT rooms (>=95%
                  coverage each), unify those GT rooms and score all room/
                  corner/chamfer metrics again ("accuracy if we accept the
                  merges"); pred-side splits are not unified.
  6. Outputs:     {name}_eval.json, {name}_eval.txt (the console summary),
                  {name}_eval_overlay.png (rooms panel + a doors/windows panel
                  when doors_windows.json is present).

Caveats (see docs/eval_floorplan.md):
  - With --gt_geometry inner, GT polygons are the INNER wall surfaces while
    predictions share zero-thickness wall centerlines, so even a perfect
    prediction loses roughly half a wall thickness (6-16 cm) per side:
    room IoU has a ceiling below 1.0 and corner distances a floor above 0.
    The default centerline GT shares the prediction's convention and has
    no such ceiling.
  - room_layout.json missed 3 rooms visible on the image (卫生间A/衣帽间A/
    阳台C); they are recovered from the page's floorplan SVG into the file
    referenced by floorplan.json's `rooms_extra` field (rooms_extra.json),
    which load_gt_rooms merges.  Unmatched predictions are still listed with
    a coverage diagnosis rather than blindly counted as false positives.
  - `world_mm` in prediction jsons holds whatever unit the .ply was in
    (metres for MVS scenes); we therefore rebuild metric coordinates from
    `pixel` + `normalization` and let --pred_units pick the unit.
"""

import argparse
import json
import os

import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
from shapely import affinity


# --------------------------------------------------------------------------- #
# Ground truth loading
# --------------------------------------------------------------------------- #

def _close_open_ring(lines, snap=1e-4):
    """Bridge dangling endpoints so an almost-closed line loop polygonizes."""
    counts = {}
    for ln in lines:
        for pt in (ln.coords[0], ln.coords[-1]):
            key = (round(pt[0] / snap), round(pt[1] / snap))
            counts.setdefault(key, [0, pt])[0] += 1
    open_ends = [pt for n, pt in counts.values() if n % 2 == 1]
    lines = list(lines)
    while len(open_ends) >= 2:
        a = open_ends.pop()
        b = min(open_ends, key=lambda p: (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2)
        open_ends.remove(b)
        lines.append(LineString([a, b]))
    return lines


def load_gt_rooms(gt_dir, simplify_tol=0.03, include_extra=True):
    """room_layout.json -> {room_name: shapely Polygon} on the (x, z) plane.

    Each pano entry stores wall segments in 3D; segments with constant y
    (height) trace the room outline at ceiling level.  Multi-pano rooms are
    unioned (the panos agree to a few millimetres) and simplified to drop
    the sliver corners that union noise creates.

    Rooms missing from room_layout.json (recovered from the page's floorplan
    SVG) are merged in unless include_extra=False.  The extra-rooms file is
    resolved from floorplan.json's `rooms_extra.local_path` when present,
    falling back to `rooms_extra.json` in gt_dir.
    """
    with open(os.path.join(gt_dir, 'room_layout.json')) as f:
        layout = json.load(f)

    per_room = {}
    for pano in layout:
        lines = []
        for ln in pano['lines']:
            s, e = ln['start'], ln['end']
            if abs(s[1] - e[1]) > 1e-6:     # vertical edge, not a plan line
                continue
            if abs(s[0] - e[0]) < 1e-9 and abs(s[2] - e[2]) < 1e-9:
                continue                     # degenerate
            lines.append(LineString([(s[0], s[2]), (e[0], e[2])]))
        # polygonize needs noded input; union the segments first so touching
        # endpoints (and tiny overlaps between duplicated edges) are merged.
        polys = list(polygonize(unary_union(lines)))
        if not polys:
            # Some rooms (卫C) come with an open ring: a small wall jog edge
            # is missing from the data.  Find endpoints used only once and
            # bridge nearest pairs, then polygonize again.
            lines = _close_open_ring(lines)
            polys = list(polygonize(unary_union(lines)))
        if not polys:
            print('warning: could not polygonize %s pano %d'
                  % (pano['roomName'], pano['panoIndex']))
            continue
        pano_poly = unary_union(polys)
        per_room.setdefault(pano['roomName'], []).append(pano_poly)

    rooms = {}
    for name, polys in per_room.items():
        merged = unary_union(polys).buffer(0)
        merged = merged.simplify(simplify_tol)
        if merged.geom_type == 'MultiPolygon':   # keep the dominant part
            merged = max(merged.geoms, key=lambda g: g.area)
        rooms[name] = merged

    if include_extra:
        extra_path = _extra_rooms_path(gt_dir)
        if extra_path:
            with open(extra_path) as f:
                extra = json.load(f)
            added = [r['name'] for r in extra['rooms'] if r['name'] not in rooms]
            for room in extra['rooms']:
                if room['name'] not in rooms:
                    rooms[room['name']] = Polygon(room['polygon']).buffer(0)
            if added:
                print('merged %d extra room(s) from %s: %s'
                      % (len(added), os.path.basename(extra_path), ', '.join(added)))
    return rooms


def _meta_local_path(gt_dir, field):
    """Resolve a GT-side data file: floorplan.json's `{field}.local_path`
    takes precedence, else the conventional `{field}.json` in gt_dir."""
    meta_path = os.path.join(gt_dir, 'floorplan.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        rel = (meta.get(field) or {}).get('local_path')
        if rel:
            path = os.path.join(gt_dir, rel)
            if os.path.exists(path):
                return path
            print('warning: floorplan.json %s.local_path=%r not found'
                  % (field, rel))
    path = os.path.join(gt_dir, '%s.json' % field)
    return path if os.path.exists(path) else None


def _extra_rooms_path(gt_dir):
    return _meta_local_path(gt_dir, 'rooms_extra')


def load_gt_rooms_centerline(gt_dir):
    """rooms_centerline.json -> {room_name: Polygon} of wall-CENTERLINE
    outlines (metres, room_layout world frame).

    Same convention as the prediction's zero-thickness shared walls, so the
    half-wall-thickness ceiling effect of the inner-surface GT disappears.
    Resolved via floorplan.json's `rooms_centerline.local_path`; returns
    None when no centerline file exists (caller falls back to inner)."""
    path = _meta_local_path(gt_dir, 'rooms_centerline')
    if path is None:
        return None
    with open(path) as f:
        data = json.load(f)
    return {r['name']: Polygon(r['polygon']).buffer(0) for r in data['rooms']}


def _seg_orient(seg):
    """'h' if the segment runs mostly along x, else 'v' (along y/z)."""
    seg = np.asarray(seg, dtype=np.float64)
    d = seg[1] - seg[0]
    return 'h' if abs(d[0]) >= abs(d[1]) else 'v'


def _seg_width_iou(seg_a, seg_b, orient):
    """1-D interval IoU of two opening segments along their shared wall axis.

    Both segments lie on (nearly) the same wall, so project each onto the
    along-wall coordinate (x for a horizontal wall, y/z for a vertical one)
    and take the overlap / union of the two intervals.  Captures how well the
    predicted opening's WIDTH and placement match the ground truth: 1.0 =
    identical extent, 0.0 = no overlap along the wall."""
    ax = 0 if orient == 'h' else 1
    a0, a1 = sorted(np.asarray(seg_a)[:, ax])
    b0, b1 = sorted(np.asarray(seg_b)[:, ax])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def load_gt_doors_windows(gt_dir):
    """doors_windows.json -> list of door/window/opening items in the GT
    (rooms_centerline / room_layout world) frame, or None if the file is
    absent (caller then skips the door/window evaluation).

    Resolved via floorplan.json's `doors_windows.local_path`, else the
    conventional doors_windows.json in gt_dir.  Each item keeps its type
    (door / window / opening), subtype, nearest room, centre point, occupied
    wall sub-segment and the segment's orientation ('h'/'v')."""
    path = _meta_local_path(gt_dir, 'doors_windows')
    if path is None:
        return None
    with open(path) as f:
        data = json.load(f)
    items = []
    for it in data.get('items', []):
        seg = np.asarray(it['segment'], dtype=np.float64)
        items.append({
            'type': it['type'],
            'subtype': it.get('subtype', ''),
            'room': it.get('room'),
            'center': np.asarray(it['center'], dtype=np.float64),
            'seg': seg,
            'orient': _seg_orient(seg),
            'width_m': it.get('width_m'),
        })
    return items


# --------------------------------------------------------------------------- #
# Prediction loading
# --------------------------------------------------------------------------- #

def pixel_to_metres(cols, rows, norm, unit_scale):
    """Pixel -> metres, mirroring infer_pointcloud.py:pixel_to_world.

    The density map normalizes each axis independently to [0, 255], so x and
    y use different scales; hflip mirrored only the image, decode col as
    255 - col to get back the un-flipped world frame.
    """
    mn = np.asarray(norm['min_coords'], dtype=np.float64)
    mx = np.asarray(norm['max_coords'], dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    if bool(norm.get('hflip', False)):
        cols = 255. - cols
    wx = (mn[0] + (cols / 255.) * (mx[0] - mn[0])) * unit_scale
    wy = (mn[1] + (rows / 255.) * (mx[1] - mn[1])) * unit_scale
    return np.stack([wx, wy], axis=1)


def load_pred(pred_json, unit_scale):
    """aligned_polys.json -> (room Polygons in metres, openings in metres, raw)."""
    with open(pred_json) as f:
        data = json.load(f)
    norm = data['normalization']

    rooms = []
    for room in data['rooms']:
        px = np.asarray(room['pixel'], dtype=np.float64)
        world = pixel_to_metres(px[:, 0], px[:, 1], norm, unit_scale)
        poly = Polygon(world).buffer(0)
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda g: g.area)
        rooms.append(poly)

    openings = []
    for op in data.get('openings', []):
        s, e = op['span']
        if op['axis'] == 'x':   # wall on a fixed column, span along rows
            pts = pixel_to_metres([op['line'], op['line']], [s, e], norm, unit_scale)
        else:                   # wall on a fixed row, span along columns
            pts = pixel_to_metres([s, e], [op['line'], op['line']], norm, unit_scale)
        openings.append({
            'raw': op,
            'seg': pts,                       # 2x2 endpoints in metres
            'center': pts.mean(axis=0),
        })
    return rooms, openings, data


# --------------------------------------------------------------------------- #
# Registration (Manhattan: 4 rotations x mirror x scale x translation)
# --------------------------------------------------------------------------- #

_ORIENT = []
for k in range(4):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    for mirror in (False, True):
        mat = rot @ (np.diag([-1., 1.]) if mirror else np.eye(2))
        _ORIENT.append((k * 90, mirror, mat))


def _rasterize(polys, res, origin=None, shape=None, pad=0):
    """Union-rasterize polygons to a uint8 mask at `res` metres/pixel."""
    pts_all = np.concatenate([np.asarray(p.exterior.coords) for p in polys])
    if origin is None:
        origin = pts_all.min(axis=0) - pad * res
    if shape is None:
        extent = pts_all.max(axis=0) - origin
        shape = (int(np.ceil(extent[1] / res)) + 1 + pad,
                 int(np.ceil(extent[0] / res)) + 1 + pad)
    mask = np.zeros(shape, dtype=np.uint8)
    for p in polys:
        pix = np.round((np.asarray(p.exterior.coords) - origin) / res).astype(np.int32)
        cv2.fillPoly(mask, [pix], 1)
    return mask, np.asarray(origin, dtype=np.float64)


def _best_translation(gt_mask, gt_origin, pred_polys, res):
    """Optimal translation of pred onto gt by mask cross-correlation.

    Returns (translation vector metres, IoU at that translation).
    """
    pred_mask, pred_origin = _rasterize(pred_polys, res)
    ph, pw = pred_mask.shape
    padded = np.zeros((gt_mask.shape[0] + 2 * ph, gt_mask.shape[1] + 2 * pw),
                      dtype=np.uint8)
    padded[ph:ph + gt_mask.shape[0], pw:pw + gt_mask.shape[1]] = gt_mask
    score = cv2.matchTemplate(padded.astype(np.float32),
                              pred_mask.astype(np.float32), cv2.TM_CCORR)
    _, max_val, _, max_loc = cv2.minMaxLoc(score)
    inter = float(max_val)
    union = float(pred_mask.sum() + gt_mask.sum() - inter)
    iou = inter / union if union > 0 else 0.
    # max_loc is (x, y) of the template inside `padded`; convert to metres.
    shift_px = np.array([max_loc[0] - pw, max_loc[1] - ph], dtype=np.float64)
    trans = gt_origin + shift_px * res - pred_origin
    return trans, iou


def register(pred_rooms, gt_rooms, args):
    """Search orientation x scale x translation maximizing union-mask IoU.

    Returns dict with rotation_deg, mirror, scale, translation and a
    transform(points Nx2) -> Nx2 function mapping prediction to GT frame.
    """
    gt_polys = list(gt_rooms.values())
    coarse = args.raster_res
    gt_mask, gt_origin = _rasterize(gt_polys, coarse)

    if args.scale is not None:
        scales = [args.scale]
    else:
        lo, hi = args.scale_range
        scales = np.arange(lo, hi + 1e-9, args.scale_step)

    best = None
    for rot_deg, mirror, mat in _ORIENT:
        base = [affinity.affine_transform(
            p, [mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1], 0, 0])
            for p in pred_rooms]
        for s in scales:
            cand = [affinity.scale(p, xfact=s, yfact=s, origin=(0, 0)) for p in base]
            trans, iou = _best_translation(gt_mask, gt_origin, cand, coarse)
            if best is None or iou > best['iou']:
                best = {'iou': iou, 'rotation_deg': rot_deg, 'mirror': mirror,
                        'mat': mat, 'scale': float(s), 'translation': trans}

    # Fine pass: 1 cm raster, searching only +-2 coarse cells around the
    # winner (full-plan correlation at 1 cm would be needlessly huge).
    fine = args.raster_res_fine
    mat, s = best['mat'], best['scale']
    moved = [affinity.affine_transform(
        p, [mat[0, 0] * s, mat[0, 1] * s, mat[1, 0] * s, mat[1, 1] * s,
            best['translation'][0], best['translation'][1]])
        for p in pred_rooms]
    win = int(np.ceil(2 * args.raster_res / fine))
    pts = np.concatenate([np.asarray(p.exterior.coords)
                          for p in gt_polys + moved])
    origin = pts.min(axis=0)
    extent = pts.max(axis=0) - origin
    shape = (int(np.ceil(extent[1] / fine)) + 1,
             int(np.ceil(extent[0] / fine)) + 1)
    pred_mask, _ = _rasterize(moved, fine, origin=origin, shape=shape)
    gt_mask_f, _ = _rasterize(gt_polys, fine,
                              origin=origin - win * fine,
                              shape=(shape[0] + 2 * win, shape[1] + 2 * win))
    score = cv2.matchTemplate(gt_mask_f.astype(np.float32),
                              pred_mask.astype(np.float32), cv2.TM_CCORR)
    _, max_val, _, max_loc = cv2.minMaxLoc(score)
    inter = float(max_val)
    union = float(pred_mask.sum() + gt_mask_f.sum() - inter)
    iou_f = inter / union if union > 0 else 0.
    if iou_f > best['iou']:
        # template at (win, win) == zero extra shift; beyond it the pred
        # content matches GT further along +x/+y, so move pred by +shift.
        shift = np.array([max_loc[0] - win, max_loc[1] - win]) * fine
        best['translation'] = best['translation'] + shift
        best['iou'] = iou_f

    mat, s, t = best['mat'], best['scale'], best['translation']

    def transform(pts):
        pts = np.asarray(pts, dtype=np.float64)
        return pts @ (mat.T * s) + t

    best['transform'] = transform
    return best


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def match_rooms(pred_polys, gt_rooms, min_iou):
    """Greedy 1:1 matching by descending IoU.  Returns (pairs, iou_of_pair)."""
    names = list(gt_rooms.keys())
    cand = []
    for i, pp in enumerate(pred_polys):
        for name in names:
            gp = gt_rooms[name]
            inter = pp.intersection(gp).area
            if inter <= 0:
                continue
            iou = inter / pp.union(gp).area
            if iou >= min_iou:
                cand.append((iou, i, name))
    cand.sort(reverse=True)
    used_p, used_g, pairs = set(), set(), {}
    for iou, i, name in cand:
        if i in used_p or name in used_g:
            continue
        used_p.add(i)
        used_g.add(name)
        pairs[i] = (name, iou)
    return pairs


def _geoms(poly):
    """Polygon -> [poly]; MultiPolygon -> its parts.  Merged inner-surface GT
    unions keep real wall gaps between constituents and may be MultiPolygon."""
    return list(poly.geoms) if poly.geom_type == 'MultiPolygon' else [poly]


def corner_metrics(pred_poly, gt_poly, tols):
    """Corner precision/recall at each tolerance + matched mean distance."""
    pc = np.vstack([np.asarray(g.exterior.coords)[:-1] for g in _geoms(pred_poly)])
    gc = np.vstack([np.asarray(g.exterior.coords)[:-1] for g in _geoms(gt_poly)])
    dmat = np.linalg.norm(pc[:, None, :] - gc[None, :, :], axis=2)
    out = {}
    for tol in tols:
        out['precision@%g' % tol] = float((dmat.min(axis=1) <= tol).mean())
        out['recall@%g' % tol] = float((dmat.min(axis=0) <= tol).mean())
    # greedy mutual matching at the loosest tolerance for a distance figure
    tol = max(tols)
    order = np.dstack(np.unravel_index(np.argsort(dmat, axis=None), dmat.shape))[0]
    used_p, used_g, dists = set(), set(), []
    for i, j in order:
        if dmat[i, j] > tol:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        dists.append(dmat[i, j])
    out['matched_corners'] = len(dists)
    out['mean_corner_dist'] = float(np.mean(dists)) if dists else None
    return out


def boundary_chamfer(pred_poly, gt_poly, step=0.05):
    """Symmetric boundary distance stats (metres)."""
    def sample(poly):
        pts = []
        for g in _geoms(poly):
            ring = g.exterior
            n = max(int(ring.length / step), 8)
            pts += [ring.interpolate(i * ring.length / n) for i in range(n)]
        return pts

    def dist(poly, pt):
        return min(g.exterior.distance(pt) for g in _geoms(poly))

    d_pg = [dist(gt_poly, pt) for pt in sample(pred_poly)]
    d_gp = [dist(pred_poly, pt) for pt in sample(gt_poly)]
    both = np.asarray(d_pg + d_gp)
    return {'mean': float(both.mean()),
            'median': float(np.median(both)),
            'p90': float(np.percentile(both, 90))}


def evaluate_rooms(pred_t, gt_rooms, args):
    """One full room-geometry round against a GT room set.

    Greedy match, per-room IoU/corners/chamfer, unmatched diagnoses,
    aggregated corner stats and total-outline figures.
    Returns (pairs, rooms_block, corners_block, total_block)."""
    pairs = match_rooms(pred_t, gt_rooms, args.match_iou)
    per_room, corner_rows = [], []
    for i, (gname, iou) in sorted(pairs.items(), key=lambda kv: -kv[1][1]):
        pp, gp = pred_t[i], gt_rooms[gname]
        cm = corner_metrics(pp, gp, args.corner_tol)
        corner_rows.append(cm)
        per_room.append({
            'gt': gname, 'pred': i, 'iou': round(iou, 3),
            'gt_area_m2': round(gp.area, 2), 'pred_area_m2': round(pp.area, 2),
            'corners': cm, 'chamfer': boundary_chamfer(pp, gp),
        })
    n_p, n_g, n_m = len(pred_t), len(gt_rooms), len(pairs)
    precision = n_m / n_p if n_p else 0.
    recall = n_m / n_g if n_g else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.

    # Merge/split diagnosis: a GT room that failed to match is usually not
    # absent but swallowed by a bigger prediction (客厅+餐厅 merged into one),
    # and an unmatched prediction is decomposed into the GT rooms it covers.
    matched_gt = {v[0] for v in pairs.values()}
    unmatched_gt = []
    for nm in gt_rooms:
        if nm in matched_gt:
            continue
        gp = gt_rooms[nm]
        covs = [(pp.intersection(gp).area / gp.area, i)
                for i, pp in enumerate(pred_t)]
        cov, i = max(covs)
        unmatched_gt.append({'gt': nm, 'best_pred': i if cov > 0.05 else None,
                             'coverage': round(cov, 2)})
    unmatched_pred = []
    for i in range(n_p):
        if i in pairs:
            continue
        pp = pred_t[i]
        comp = sorted(((pp.intersection(gp).area / pp.area, nm)
                       for nm, gp in gt_rooms.items()), reverse=True)
        comp = [{'gt': nm, 'frac': round(f, 2)} for f, nm in comp if f > 0.05]
        unmatched_pred.append({'pred': i, 'area_m2': round(pp.area, 1),
                               'covers': comp})

    rooms_block = {
        'n_pred': n_p, 'n_gt': n_g, 'n_matched': n_m,
        'precision': round(precision, 3), 'recall': round(recall, 3),
        'f1': round(f1, 3),
        'mean_iou': round(float(np.mean([v[1] for v in pairs.values()])), 3)
            if pairs else 0.,
        'per_room': per_room,
        'unmatched_gt': unmatched_gt,
        'unmatched_pred': unmatched_pred,
    }

    # corner metrics aggregated over matched rooms (corner-count weighted)
    corners_agg = {}
    for tol in args.corner_tol:
        for kind in ('precision', 'recall'):
            key = '%s@%g' % (kind, tol)
            corners_agg[key] = round(float(np.mean([c[key] for c in corner_rows])), 3) \
                if corner_rows else None
    dists = [c['mean_corner_dist'] for c in corner_rows
             if c['mean_corner_dist'] is not None]
    ns = [c['matched_corners'] for c in corner_rows
          if c['mean_corner_dist'] is not None]
    corners_agg['matched_corners'] = int(np.sum(ns)) if ns else 0
    corners_agg['mean_corner_dist'] = \
        round(float(np.average(dists, weights=ns)), 3) if ns else None

    gt_union = unary_union(list(gt_rooms.values()))
    pred_union = unary_union(pred_t)
    total_block = {
        'outline_iou': round(pred_union.intersection(gt_union).area /
                             pred_union.union(gt_union).area, 3),
        'pred_area_m2': round(pred_union.area, 1),
        'gt_area_m2': round(gt_union.area, 1),
    }
    return pairs, rooms_block, corners_agg, total_block


# Prediction opening types -> GT door/window/opening vocabulary.  The pipeline
# emits 'door' (leaf openings) and 'passage' (wide multi-room openings, i.e.
# 门洞/垭口); it never emits windows.
_DW_PRED_TYPE = {'door': 'door', 'passage': 'opening'}


def eval_doors_windows(pred_openings, transform, gt_items, tol):
    """Position-level door/window comparison against doors_windows.json.

    Each predicted opening's centre is mapped into the GT frame and greedily
    matched to the nearest GT item that shares its orientation and lies within
    `tol` metres (nearest pair first, one-to-one).  A match is a position hit
    regardless of type; whether the type also agrees is tracked separately.

    Windows are never predicted (the pipeline emits doors and passages only),
    so window recall is expected to be ~0 -- recall is broken out per GT type
    so that expected gap does not masquerade as a door/opening failure.
    """
    preds = []
    for op in pred_openings:
        c = transform(op['center'][None, :])[0]
        seg = transform(op['seg'])
        raw_type = op['raw'].get('type')
        preds.append({
            'raw_type': raw_type,
            'type': _DW_PRED_TYPE.get(raw_type, raw_type),
            'center': c,
            'seg': seg,                       # 2x2 endpoints in GT frame
            'orient': _seg_orient(seg),
            'width_m': op['raw'].get('width_m'),
        })

    # candidate pairs (same orientation, within tol), greedy nearest-first
    cand = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gt_items):
            if p['orient'] != g['orient']:
                continue
            d = float(np.hypot(*(p['center'] - g['center'])))
            if d <= tol:
                cand.append((d, i, j))
    cand.sort()
    p_used, g_used, matched = set(), set(), []
    for d, i, j in cand:
        if i in p_used or j in g_used:
            continue
        p_used.add(i)
        g_used.add(j)
        p, g = preds[i], gt_items[j]
        matched.append({
            'pred_type': p['raw_type'], 'gt_type': g['type'],
            'gt_subtype': g['subtype'], 'gt_room': g['room'],
            'type_ok': p['type'] == g['type'],
            'dist_m': round(d, 3),
            'width_iou': round(_seg_width_iou(p['seg'], g['seg'], g['orient']), 3),
            'pred_width_m': (round(float(p['width_m']), 3)
                             if p['width_m'] is not None else None),
            'gt_width_m': (round(float(g['width_m']), 3)
                           if g['width_m'] is not None else None),
            'pred_center_m': [round(float(v), 3) for v in p['center']],
            'gt_center_m': [round(float(v), 3) for v in g['center']],
            'pred_seg_m': np.round(p['seg'], 3).tolist(),
            'gt_seg_m': np.round(g['seg'], 3).tolist(),
        })

    false_pos = [{'pred_type': preds[i]['raw_type'],
                  'center_m': [round(float(v), 3) for v in preds[i]['center']],
                  'seg_m': np.round(preds[i]['seg'], 3).tolist(),
                  'width_m': preds[i]['width_m']}
                 for i in range(len(preds)) if i not in p_used]
    missed = [{'gt_type': gt_items[j]['type'], 'subtype': gt_items[j]['subtype'],
               'room': gt_items[j]['room'],
               'center_m': [round(float(v), 3) for v in gt_items[j]['center']],
               'seg_m': np.round(gt_items[j]['seg'], 3).tolist(),
               'width_m': gt_items[j]['width_m']}
              for j in range(len(gt_items)) if j not in g_used]

    tp, fp, fn = len(matched), len(false_pos), len(missed)
    prec = tp / (tp + fp) if tp + fp else 0.
    rec = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.
    dists = [m['dist_m'] for m in matched]
    ious = [m['width_iou'] for m in matched]

    by_type = {}
    for t in ('door', 'window', 'opening'):
        by_type[t] = {
            'matched': sum(1 for m in matched if m['gt_type'] == t),
            'total': sum(1 for g in gt_items if g['type'] == t),
        }
    type_ok = sum(1 for m in matched if m['type_ok'])

    return {
        'match_tol_m': tol,
        'n_pred': len(preds), 'n_gt': len(gt_items),
        'gt_counts': {t: by_type[t]['total'] for t in by_type},
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(prec, 3), 'recall': round(rec, 3), 'f1': round(f1, 3),
        'mean_center_dist_m': round(float(np.mean(dists)), 3) if dists else None,
        'mean_width_iou': round(float(np.mean(ious)), 3) if ious else None,
        'by_gt_type': by_type,
        'type_agreement': {'ok': type_ok, 'of': tp},
        'matched': matched, 'false_positives': false_pos, 'missed': missed,
    }


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

def _setup_cjk_font():
    """Pick an installed CJK-capable font so room names render."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Hiragino Sans GB', 'PingFang SC', 'Arial Unicode MS',
                 'Noto Sans CJK SC', 'Songti SC'):
        if name in have:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False


def _draw_gt_rooms(ax, gt_rooms):
    """Blue GT room outlines + names, shared background of every panel."""
    ax.set_aspect('equal')
    for name, gp in gt_rooms.items():
        xs, ys = gp.exterior.xy
        ax.plot(xs, ys, color='#1f3d7a', lw=2)
        c = gp.representative_point()
        ax.text(c.x, c.y, name, color='#1f3d7a', fontsize=9,
                ha='center', va='center')


def _draw_rooms_panel(ax, pred_polys, pairs):
    ax.set_title('rooms: GT (blue outline) vs prediction (orange fill)')
    for i, pp in enumerate(pred_polys):
        xs, ys = pp.exterior.xy
        ax.fill(xs, ys, color='#ff7f0e', alpha=0.30, lw=1.2,
                edgecolor='#b25406')
        c = pp.representative_point()
        label = 'p%d' % i
        if i in pairs:
            label += '\nIoU %.2f' % pairs[i][1]
        ax.text(c.x, c.y + 0.35, label, color='#7a3a00', fontsize=8,
                ha='center', va='center')


def _draw_doors_windows_panel(ax, dw_result):
    """Door/window POSITION matching: GT segment (thick) vs predicted opening
    segment (thin dashed) with a connector to show the centre offset."""
    ax.set_title('doors/windows (position): green=hit  yellow=hit, type differs\n'
                 'red=false positive  red dashed=missed door/opening  '
                 'gray dashed=missed window (never predicted)', fontsize=10)
    for m in dw_result['matched']:
        col = '#2ca02c' if m['type_ok'] else '#f5b800'
        g = np.asarray(m['gt_seg_m'])
        p = np.asarray(m['pred_seg_m'])
        ax.plot(g[:, 0], g[:, 1], color=col, lw=6, alpha=0.9,
                solid_capstyle='butt')
        ax.plot(p[:, 0], p[:, 1], color=col, lw=2, ls=(0, (2, 2)))
        pc, gc = m['pred_center_m'], m['gt_center_m']
        ax.plot([pc[0], gc[0]], [pc[1], gc[1]], color=col, lw=1)
        ax.text(gc[0], gc[1], 'IoU %.2f' % m['width_iou'], color=col,
                fontsize=7, ha='center', va='bottom')
    for f in dw_result['false_positives']:
        s = np.asarray(f['seg_m'])
        ax.plot(s[:, 0], s[:, 1], color='#d62728', lw=6, alpha=0.9,
                solid_capstyle='butt')
    for e in dw_result['missed']:
        col = '#999999' if e['gt_type'] == 'window' else '#d62728'
        s = np.asarray(e['seg_m'])
        ax.plot(s[:, 0], s[:, 1], color=col, lw=3, ls='--', alpha=0.9)


def save_overlay(out_path, gt_rooms, pred_polys, pairs, dw_result=None):
    _setup_cjk_font()

    # rooms is always shown; the door/window panel only when there is
    # something to draw (doors_windows.json present).
    panels = ['rooms']
    if dw_result and any(dw_result.get(k) for k in
                         ('matched', 'false_positives', 'missed')):
        panels.append('doors_windows')

    fig, axes = plt.subplots(1, len(panels), figsize=(11 * len(panels), 11),
                             squeeze=False)
    axes = axes[0]
    # GT z grows toward the top of the floor-plan image, which matches a
    # normal (y-up) matplotlib axis -- no flipping needed.
    for ax in axes:
        _draw_gt_rooms(ax, gt_rooms)
    for ax, kind in zip(axes, panels):
        if kind == 'rooms':
            _draw_rooms_panel(ax, pred_polys, pairs)
        elif kind == 'doors_windows':
            _draw_doors_windows_panel(ax, dw_result)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _format_rooms(L, rm, gt_col_w=10):
    """Append the room-level lines (match table + unmatched diagnoses)."""
    L.append('pred %d / gt %d   matched %d   P %.3f  R %.3f  F1 %.3f   mean IoU %.3f' %
             (rm['n_pred'], rm['n_gt'], rm['n_matched'],
              rm['precision'], rm['recall'], rm['f1'], rm['mean_iou']))
    L.append('%-*s %-6s %-6s %-9s %-9s %s' %
             (gt_col_w, 'gt room', 'pred', 'IoU', 'areaGT', 'areaPred',
              'chamfer mean'))
    for row in rm['per_room']:
        L.append('%-*s p%-5d %-6.3f %-9.2f %-9.2f %.3f m' %
                 (gt_col_w, row['gt'], row['pred'], row['iou'],
                  row['gt_area_m2'], row['pred_area_m2'],
                  row['chamfer']['mean']))
    for row in rm['unmatched_gt']:
        hint = ('%.0f%% covered by p%d (likely merged)'
                % (row['coverage'] * 100, row['best_pred'])
                if row['best_pred'] is not None else 'no prediction overlaps')
        L.append('GT not matched: %-8s %s' % (row['gt'], hint))
    for row in rm['unmatched_pred']:
        comp = ' + '.join('%s %.0f%%' % (c['gt'], c['frac'] * 100)
                          for c in row['covers']) or \
            'no GT overlap (a room missing from the GT data?)'
        L.append('pred not matched: p%-3d %5.1f m2  covers: %s'
                 % (row['pred'], row['area_m2'], comp))


def _format_corners(L, c):
    """Append the aggregated corner-metric lines."""
    for k in sorted(k for k in c if k.startswith('precision')):
        tol = k.split('@')[1]
        L.append('  @%sm  P %.3f  R %.3f' % (tol, c[k], c['recall@' + tol]))
    if c.get('mean_corner_dist') is not None:
        L.append('  matched corner mean dist %.3f m (n=%d)' %
                 (c['mean_corner_dist'], c['matched_corners']))


def _total_line(t):
    return ('total outline IoU %.3f   area pred %.1f m2 / gt %.1f m2' %
            (t['outline_iou'], t['pred_area_m2'], t['gt_area_m2']))


def summary_text(result):
    """The human-readable metrics summary (console and {name}_eval.txt)."""
    r = result
    L = ['=== registration ===']
    reg = r['registration']
    L.append('rotation %d deg  mirror %s  scale %.3f  union IoU %.3f' %
             (reg['rotation_deg'], reg['mirror'], reg['scale'],
              reg['union_iou']))
    L.append('')
    L.append('=== rooms (%s GT) ===' % r.get('gt_geometry', 'inner'))
    _format_rooms(L, r['rooms'])
    L.append('')
    L.append(_total_line(r['total']))
    L.append('')
    L.append('=== corners (matched rooms) ===')
    _format_corners(L, r['corners'])
    if 'doors_windows' in r:
        dw = r['doors_windows']
        gc = dw['gt_counts']
        bt = dw['by_gt_type']
        ta = dw['type_agreement']
        md = ('%.2f m' % dw['mean_center_dist_m']
              if dw['mean_center_dist_m'] is not None else 'n/a')
        mi = ('%.3f' % dw['mean_width_iou']
              if dw['mean_width_iou'] is not None else 'n/a')
        L.append('')
        L.append('=== doors/windows (position level) ===')
        L.append('match tol %.2f m' % dw['match_tol_m'])
        L.append('pred %d   gt %d (door %d, window %d, opening %d)' %
                 (dw['n_pred'], dw['n_gt'], gc['door'], gc['window'],
                  gc['opening']))
        L.append('matched %d   P %.3f  R %.3f  F1 %.3f   '
                 'mean center dist %s   mean width IoU %s' %
                 (dw['tp'], dw['precision'], dw['recall'], dw['f1'], md, mi))
        L.append('by GT type:  door %d/%d   window %d/%d   opening %d/%d' %
                 (bt['door']['matched'], bt['door']['total'],
                  bt['window']['matched'], bt['window']['total'],
                  bt['opening']['matched'], bt['opening']['total']))
        L.append('type agreement on matched: %d/%d' % (ta['ok'], ta['of']))
        for m in dw['matched']:
            L.append('  %s  pred %-7s -> GT %-7s %-6s room=%s  '
                     'dist %.2f m  IoU %.2f  (w pred %.2f / gt %.2f m)' %
                     ('hit ' if m['type_ok'] else 'hit~', m['pred_type'],
                      m['gt_type'], m['gt_subtype'], m['gt_room'], m['dist_m'],
                      m['width_iou'],
                      m['pred_width_m'] if m['pred_width_m'] is not None else 0.0,
                      m['gt_width_m'] if m['gt_width_m'] is not None else 0.0))
        for f in dw['false_positives']:
            L.append('  FP    pred %-7s center=(%.2f, %.2f)' %
                     (f['pred_type'], f['center_m'][0], f['center_m'][1]))
        for e in dw['missed']:
            L.append('  miss  GT %-7s %-6s room=%s center=(%.2f, %.2f)' %
                     (e['gt_type'], e['subtype'], e['room'],
                      e['center_m'][0], e['center_m'][1]))
    return '\n'.join(L) + '\n'


def print_summary(result):
    print('\n' + summary_text(result), end='')


# --------------------------------------------------------------------------- #

def get_args_parser():
    p = argparse.ArgumentParser('Floor-plan evaluation against realsee GT')
    p.add_argument('--pred', required=True, help='{name}_aligned_polys.json')
    p.add_argument('--gt_dir', required=True,
                   help='folder with room_layout.json (+ optional '
                        'rooms_centerline.json / doors_windows.json)')
    p.add_argument('--output_dir', default='infer_out')
    p.add_argument('--pred_units', choices=['m', 'mm'], default='m',
                   help='unit of the original .ply (MVS scenes are metres)')
    p.add_argument('--gt_geometry', choices=['centerline', 'inner'],
                   default='centerline',
                   help='GT polygon convention: wall centerlines '
                        '(rooms_centerline.json, same convention as the '
                        'prediction — no half-wall-thickness ceiling effect) '
                        'or inner wall surfaces (room_layout.json).  Falls '
                        'back to inner when no centerline file exists.')
    p.add_argument('--scale', type=float, default=None,
                   help='lock registration scale (e.g. 1.0)')
    p.add_argument('--scale_range', type=float, nargs=2, default=[0.90, 1.10])
    p.add_argument('--scale_step', type=float, default=0.01)
    p.add_argument('--raster_res', type=float, default=0.05,
                   help='coarse registration raster, metres/pixel')
    p.add_argument('--raster_res_fine', type=float, default=0.01)
    p.add_argument('--match_iou', type=float, default=0.5)
    p.add_argument('--corner_tol', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    p.add_argument('--dw_match_tol', type=float, default=0.6,
                   help='door/window position match tolerance, metres '
                        '(centre distance; matched pairs must share orientation)')
    p.add_argument('--no_doors_windows_eval', action='store_true')
    return p


def main():
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.basename(args.pred).replace('_aligned_polys.json', '') \
                                      .replace('_polys.json', '')

    gt_geom, gt_rooms = args.gt_geometry, None
    if gt_geom == 'centerline':
        gt_rooms = load_gt_rooms_centerline(args.gt_dir)
        if gt_rooms is None:
            print('no rooms_centerline.json in %s; falling back to '
                  'inner-surface GT (room_layout.json)' % args.gt_dir)
            gt_geom = 'inner'
    if gt_rooms is None:
        gt_rooms = load_gt_rooms(args.gt_dir)
    print('GT rooms: %d (%s geometry), total area %.1f m2' %
          (len(gt_rooms), gt_geom, sum(p.area for p in gt_rooms.values())))

    unit_scale = 0.001 if args.pred_units == 'mm' else 1.0
    pred_rooms, pred_openings, _ = load_pred(args.pred, unit_scale)
    print('pred rooms: %d, openings: %d' % (len(pred_rooms), len(pred_openings)))

    reg = register(pred_rooms, gt_rooms, args)
    transform = reg['transform']
    print('registration: rot %d deg, mirror %s, scale %.3f, IoU %.3f' %
          (reg['rotation_deg'], reg['mirror'], reg['scale'], reg['iou']))

    pred_t = [Polygon(transform(np.asarray(p.exterior.coords))).buffer(0)
              for p in pred_rooms]

    # ---- rooms ----
    pairs, rooms_block, corners_agg, total = evaluate_rooms(pred_t, gt_rooms, args)

    result = {
        'pred': args.pred,
        'gt_dir': args.gt_dir,
        'gt_geometry': gt_geom,
        'registration': {
            'rotation_deg': reg['rotation_deg'], 'mirror': reg['mirror'],
            'scale': round(reg['scale'], 4),
            'translation_m': [round(float(v), 3) for v in reg['translation']],
            'union_iou': round(reg['iou'], 3),
        },
        'rooms': rooms_block,
        'corners': corners_agg,
        'total': total,
    }

    # ---- doors / windows / openings (position level) ----
    # doors_windows.json (if present, referenced by floorplan.json's
    # doors_windows.local_path) gives door/window/opening positions on the
    # wall centrelines; score the predicted openings' positions against them.
    # This is the single opening evaluation (the old connectivity-level path
    # over openings_gt.json has been retired).
    if not args.no_doors_windows_eval:
        gt_items = load_gt_doors_windows(args.gt_dir)
        if gt_items:
            result['doors_windows'] = eval_doors_windows(
                pred_openings, transform, gt_items, args.dw_match_tol)
            print('doors/windows GT: %d items -> P %.3f R %.3f F1 %.3f' %
                  (result['doors_windows']['n_gt'],
                   result['doors_windows']['precision'],
                   result['doors_windows']['recall'],
                   result['doors_windows']['f1']))
        elif gt_items is None:
            print('no doors_windows.json found; skipping door/window eval')
        else:
            print('doors_windows.json has no items; skipping door/window eval')

    out_json = os.path.join(args.output_dir, '%s_eval.json' % name)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    out_txt = os.path.join(args.output_dir, '%s_eval.txt' % name)
    with open(out_txt, 'w') as f:
        f.write('pred:   %s\ngt_dir: %s (%s geometry)\n\n'
                % (args.pred, args.gt_dir, gt_geom))
        f.write(summary_text(result))

    save_overlay(os.path.join(args.output_dir, '%s_eval_overlay.png' % name),
                 gt_rooms, pred_t, pairs, result.get('doors_windows'))

    print_summary(result)
    print('\nsaved: %s, %s, %s' % (out_json, out_txt,
          os.path.join(args.output_dir, '%s_eval_overlay.png' % name)))


if __name__ == '__main__':
    main()
