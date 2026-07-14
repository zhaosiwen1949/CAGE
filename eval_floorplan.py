"""Evaluate a predicted floor plan against a realsee ground-truth room layout.

Compares ``{name}_aligned_polys.json`` (from align_floorplan.py) with the
ground-truth layout in a folder like ``data/custom/xinghewan_floorplan/``:

  room_layout.json        per-pano wall lines (metres, y-up, one global
                          frame); the horizontal (ceiling-level) lines of
                          each pano trace the room polygon on the floor
                          plane (x, z).  INNER wall surfaces.
  rooms_centerline.json   wall-CENTERLINE room polygons (same convention as
                          the prediction) — the default GT (--gt_geometry).
  openings_gt.json        hand-annotated room-connectivity list
                          (door/passage), optional.

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
                  and area, opening connectivity P/R/F1.
  5. Merged-GT:   when one prediction swallowed several GT rooms (>=95%
                  coverage each), unify those GT rooms and score all room/
                  corner/chamfer metrics again ("accuracy if we accept the
                  merges"); pred-side splits are not unified.
  6. Outputs:     {name}_eval.json, {name}_eval.txt (the console summary),
                  {name}_eval_overlay.png.

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
from shapely.geometry import LineString, Point, Polygon
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


def load_gt_openings(gt_dir, gt_rooms):
    """openings_gt.json -> (evaluable pair list, skipped entries)."""
    path = os.path.join(gt_dir, 'openings_gt.json')
    if not os.path.exists(path):
        return [], []
    with open(path) as f:
        data = json.load(f)
    evaluable, skipped = [], []
    for entry in data['openings']:
        ok = entry.get('evaluable', True) and all(r in gt_rooms for r in entry['rooms'])
        (evaluable if ok else skipped).append(entry)
    return evaluable, skipped


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
    aggregated corner stats and total-outline figures.  Used twice: for the
    original GT and for the merged-GT re-evaluation.
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


def find_merge_groups(pred_t, gt_rooms, pairs, cov_thr=0.95):
    """Prediction rooms that swallowed several GT rooms -> {pred_i: [names]}.

    A GT room belongs to prediction j's group when j covers >= cov_thr of
    its area; the GT room 1:1-matched to j (if any) joins the group too.
    Only groups with two or more GT rooms count as a merge (a single fully
    covered room is just a normal match)."""
    groups = {}
    for j, pp in enumerate(pred_t):
        members = [nm for nm, gp in gt_rooms.items()
                   if pp.intersection(gp).area / gp.area >= cov_thr]
        if j in pairs and pairs[j][0] not in members:
            members.append(pairs[j][0])
        if len(members) >= 2:
            members.sort(key=lambda nm: -gt_rooms[nm].area)  # biggest first
            groups[j] = members
    return groups


def eval_openings(pred_openings, transform, pairs, room_sets,
                  gt_rooms, gt_open, args):
    """Connectivity-level opening comparison, strict + lenient.

    Strict: map each prediction's two room indices through the 1:1 room
    matching to GT names and compare unordered pairs with the annotation.
    Predictions touching unmatched rooms are 'unjudgeable' there.

    Lenient: resolve each prediction room to the SET of GT rooms it covers
    (>50% of the GT room area) so that openings on the boundary of a merged
    prediction (e.g. 客厅+餐厅 as one room) still get credit; GT pairs whose
    two rooms were merged into the same prediction are structurally
    undetectable and reported apart instead of as plain misses.
    """
    idx2name = {i: name for i, (name, _) in pairs.items()}
    gt_pairs = {frozenset(e['rooms']): e for e in gt_open}

    # ---- strict pass ----
    matched, false_pos, unjudgeable = [], [], []
    seen = set()
    for op in pred_openings:
        raw = op['raw']
        rms = raw.get('rooms', [])
        names = [idx2name.get(r) for r in rms]
        center = transform(op['center'][None, :])[0]
        rec = {'pred': raw, 'rooms_pred': rms, 'rooms_gt': names,
               'center_m': [round(float(c), 3) for c in center]}
        if len(names) != 2 or None in names:
            unjudgeable.append(rec)
            continue
        key = frozenset(names)
        if key in gt_pairs:
            # center should sit near both rooms' shared wall: report the
            # larger of its distances to the two room polygons.
            off = max(gt_rooms[n].distance(Point(center)) for n in names)
            rec['gt_type'] = gt_pairs[key]['type']
            rec['center_off_m'] = round(float(off), 3)
            matched.append(rec)
            seen.add(key)
        else:
            false_pos.append(rec)

    missed = [e for k, e in gt_pairs.items() if k not in seen]
    tp, fp, fn = len(matched), len(false_pos), len(missed)
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.

    # ---- lenient pass ----
    len_hits, len_fp, len_na = [], [], []
    remaining = set(gt_pairs) - seen
    for rec in list(unjudgeable) + list(false_pos):
        s1, s2 = (room_sets.get(i, set()) for i in rec['rooms_pred'][:2]) \
            if len(rec['rooms_pred']) == 2 else (set(), set())
        if not s1 or not s2:
            len_na.append(rec)
            continue
        compat = [k for k in remaining
                  if len(k) == 2 and
                  ((min(k) in s1 and max(k) in s2) or
                   (min(k) in s2 and max(k) in s1))]
        if compat:
            key = compat[0]
            remaining.discard(key)
            # annotate the original record (it stays in its strict list too)
            # so the overlay can recolor it as a lenient hit.
            rec['resolved_pair'] = sorted(key)
            rec['gt_type'] = gt_pairs[key]['type']
            len_hits.append(rec)
        else:
            len_fp.append(rec)

    # GT pairs both of whose rooms sit inside one predicted room can never
    # be detected as an opening -- that error already shows up as a room
    # merge, so list them separately.
    undetectable, len_missed = [], []
    for k in remaining:
        merged = any(set(k) <= s for s in room_sets.values())
        (undetectable if merged else len_missed).append(gt_pairs[k])

    ltp = tp + len(len_hits)
    lfp = len(len_fp)
    lfn = len(len_missed)
    lprec = ltp / (ltp + lfp) if ltp + lfp else 0.
    lrec = ltp / (ltp + lfn) if ltp + lfn else 0.
    lf1 = 2 * lprec * lrec / (lprec + lrec) if lprec + lrec else 0.

    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(precision, 3), 'recall': round(recall, 3),
            'f1': round(f1, 3),
            'matched': matched, 'false_positives': false_pos,
            'missed': missed, 'unjudgeable': unjudgeable,
            'lenient': {
                'tp': ltp, 'fp': lfp, 'fn': lfn,
                'precision': round(lprec, 3), 'recall': round(lrec, 3),
                'f1': round(lf1, 3),
                'extra_hits': len_hits, 'false_positives': len_fp,
                'missed': len_missed, 'undetectable_merged': undetectable,
                'unjudgeable': len_na,
            }}


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


def save_overlay(out_path, gt_rooms, pred_polys, pairs, transform,
                 openings_result):
    _setup_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(22, 11))

    # GT z grows toward the top of the floor-plan image, which matches a
    # normal (y-up) matplotlib axis -- no flipping needed.
    for ax in axes:
        ax.set_aspect('equal')
        for name, gp in gt_rooms.items():
            xs, ys = gp.exterior.xy
            ax.plot(xs, ys, color='#1f3d7a', lw=2)
            c = gp.representative_point()
            ax.text(c.x, c.y, name, color='#1f3d7a', fontsize=9,
                    ha='center', va='center')

    ax = axes[0]
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

    ax = axes[1]
    ax.set_title('openings: green=hit  yellow-green=lenient hit (merged room)  '
                 'red=false positive  gray=unjudgeable  purple dashed=missed')
    def _draw(rec, color):
        seg = transform(np.asarray([rec['pred_seg'][0], rec['pred_seg'][1]]))
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=6, alpha=0.9,
                solid_capstyle='butt')
    for rec in openings_result['matched']:
        _draw(rec, '#2ca02c')
    for rec in openings_result['false_positives']:
        _draw(rec, '#9acd32' if rec.get('resolved_pair') else '#d62728')
    for rec in openings_result['unjudgeable']:
        _draw(rec, '#9acd32' if rec.get('resolved_pair') else '#999999')
    lenient = openings_result.get('lenient', {})
    resolved = {frozenset(r['resolved_pair'])
                for r in lenient.get('extra_hits', [])}
    for e in openings_result['missed']:
        if frozenset(e['rooms']) in resolved:
            continue
        a, b = (gt_rooms[r].representative_point() for r in e['rooms'])
        ax.plot([a.x, b.x], [a.y, b.y], color='#9467bd', lw=2, ls='--')

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
    if 'openings' in r:
        o = r['openings']
        L.append('')
        L.append('=== openings (connectivity level) ===')
        L.append('strict (1:1 matched rooms only):')
        L.append('TP %d  FP %d  FN %d   P %.3f  R %.3f  F1 %.3f' %
                 (o['tp'], o['fp'], o['fn'], o['precision'], o['recall'],
                  o['f1']))
        for rec in o['matched']:
            L.append('  hit   %s-%s  (%s, center off %.2f m)' %
                     (rec['rooms_gt'][0], rec['rooms_gt'][1], rec['gt_type'],
                      rec['center_off_m']))
        lo = o['lenient']
        L.append('lenient (merged prediction rooms resolved by coverage):')
        L.append('TP %d  FP %d  FN %d   P %.3f  R %.3f  F1 %.3f' %
                 (lo['tp'], lo['fp'], lo['fn'], lo['precision'], lo['recall'],
                  lo['f1']))
        for rec in lo['extra_hits']:
            L.append('  hit*  %s-%s  (%s, via merged room)' %
                     (rec['resolved_pair'][0], rec['resolved_pair'][1],
                      rec['gt_type']))
        for rec in lo['false_positives']:
            names = [gt if gt is not None else '?'
                     for gt in rec['rooms_gt']]
            L.append('  FP    %s-%s (pred rooms %s)' %
                     (names[0], names[1], rec['rooms_pred']))
        for e in lo['missed']:
            L.append('  miss  %s-%s (%s)' %
                     (e['rooms'][0], e['rooms'][1], e['type']))
        for e in lo['undetectable_merged']:
            L.append('  n/e   %s-%s (rooms merged into one prediction; '
                     'shows up as a room-level error instead)' %
                     (e['rooms'][0], e['rooms'][1]))
        for rec in lo['unjudgeable']:
            sides = ['%s' % gt if gt is not None else 'p%d?' % i
                     for gt, i in zip(rec['rooms_gt'], rec['rooms_pred'])]
            L.append('  n/a   %s (a side covers no GT room)' % '-'.join(sides))
    if 'merged_eval' in r:
        m = r['merged_eval']
        L.append('')
        L.append('=== merged-GT re-evaluation '
                 '(GT rooms unified per swallowing prediction) ===')
        for g in m['merge_groups']:
            L.append('merge: p%-3d <- %s' % (g['pred'], ' + '.join(g['gt'])))
        L.append('')
        w = max([10] + [len(row['gt']) for row in m['rooms']['per_room']])
        _format_rooms(L, m['rooms'], gt_col_w=w)
        L.append('')
        L.append(_total_line(m['total']))
        L.append('corners (matched rooms, merged GT):')
        _format_corners(L, m['corners'])
    return '\n'.join(L) + '\n'


def print_summary(result):
    print('\n' + summary_text(result), end='')


# --------------------------------------------------------------------------- #

def get_args_parser():
    p = argparse.ArgumentParser('Floor-plan evaluation against realsee GT')
    p.add_argument('--pred', required=True, help='{name}_aligned_polys.json')
    p.add_argument('--gt_dir', required=True,
                   help='folder with room_layout.json (+ openings_gt.json)')
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
    p.add_argument('--no_openings_eval', action='store_true')
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

    # ---- merged-GT re-evaluation ----
    # When one prediction swallowed several GT rooms (p15 = 客厅+餐厅+阳台B),
    # unify those GT rooms and score everything again: "how accurate is the
    # plan if we accept the merges?"  Registration is reused (merging does
    # not move any geometry).  Pred splits are NOT unified (kept as round-1
    # diagnoses).
    groups = find_merge_groups(pred_t, gt_rooms, pairs)
    if groups:
        merged_gt, consumed, merge_info = dict(gt_rooms), set(), []
        for j, members in sorted(groups.items()):
            members = [nm for nm in members if nm not in consumed]
            if len(members) < 2:
                continue
            consumed.update(members)
            mname = '+'.join(members)
            for nm in members:
                merged_gt.pop(nm)
            merged_gt[mname] = unary_union(
                [gt_rooms[nm] for nm in members]).buffer(0)
            merge_info.append({'pred': j, 'gt': members, 'merged_name': mname})
        _, rooms2, corners2, total2 = evaluate_rooms(pred_t, merged_gt, args)
        result['merged_eval'] = {'merge_groups': merge_info, 'rooms': rooms2,
                                 'corners': corners2, 'total': total2}

    # ---- openings ----
    if not args.no_openings_eval:
        gt_open_path = os.path.join(args.gt_dir, 'openings_gt.json')
        gt_open, gt_open_skipped = load_gt_openings(args.gt_dir, gt_rooms)
        if gt_open:
            # Which GT rooms does each prediction "contain" (>50% of the GT
            # room's area)?  Drives the lenient opening matching.
            room_sets = {
                i: {nm for nm, gp in gt_rooms.items()
                    if pp.intersection(gp).area / gp.area > 0.5}
                for i, pp in enumerate(pred_t)}
            openings_result = eval_openings(pred_openings, transform, pairs,
                                            room_sets, gt_rooms, gt_open, args)
            # stash segment endpoints for the overlay drawing
            for rec in (openings_result['matched'] +
                        openings_result['false_positives'] +
                        openings_result['unjudgeable']):
                idx = next(i for i, op in enumerate(pred_openings)
                           if op['raw'] is rec['pred'])
                rec['pred_seg'] = pred_openings[idx]['seg'].tolist()
            result['openings'] = openings_result
            result['openings_gt_skipped'] = gt_open_skipped
        elif not os.path.exists(gt_open_path):
            print('no openings_gt.json found; skipping opening eval')
        else:
            print('openings_gt.json has no evaluable entries (all rooms '
                  'missing from GT?); skipping opening eval')

    out_json = os.path.join(args.output_dir, '%s_eval.json' % name)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    out_txt = os.path.join(args.output_dir, '%s_eval.txt' % name)
    with open(out_txt, 'w') as f:
        f.write('pred:   %s\ngt_dir: %s (%s geometry)\n\n'
                % (args.pred, args.gt_dir, gt_geom))
        f.write(summary_text(result))

    empty = {'matched': [], 'false_positives': [], 'unjudgeable': [], 'missed': []}
    save_overlay(os.path.join(args.output_dir, '%s_eval_overlay.png' % name),
                 gt_rooms, pred_t, pairs, transform,
                 result.get('openings', empty))

    print_summary(result)
    print('\nsaved: %s, %s, %s' % (out_json, out_txt,
          os.path.join(args.output_dir, '%s_eval_overlay.png' % name)))


if __name__ == '__main__':
    main()
