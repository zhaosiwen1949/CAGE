"""
Floor-plan corners (2D) -> original point-cloud 3D world coordinates.

`infer_pointcloud.py` writes `{name}_polys.json`, where each room's `world_mm` /
`world_m` corners live in the FLOOR-PLANE frame *after* the up-axis reorder and
yaw correction that were applied before density projection. They are therefore
NOT in the original `.ply` coordinate frame.

This script inverts that transform: it reads the JSON and re-expresses every room
corner in the original point cloud's 3D world frame, so the polygons overlay on
the raw `.ply`. It emits two floors of geometry (floor + ceiling), producing
closed 3D wall rings.

Inverse of infer_pointcloud.load_input (must stay numerically reciprocal):
    world_mm(2D)  --undo yaw-->  --attach height-->  --undo reorder-->  (x, y, z)

Everything needed is already in the JSON (applied_yaw_deg + vertical min/max), so
this script depends only on numpy + the standard library -- no torch, no model,
and it does NOT need the original `.ply`.
"""

import argparse
import glob
import json
import os

import numpy as np


# Forward column permutation used by infer_pointcloud.reorder_up_axis:
#   reordered = orig[:, PERM[up_axis]]  (up-axis moved to column 2)
PERM = {
    'z': [0, 1, 2],
    'y': [0, 2, 1],
    'x': [1, 2, 0],
}


def get_args_parser():
    parser = argparse.ArgumentParser('CAGE floor-plan -> 3D world coordinates',
                                     add_help=False)
    parser.add_argument('--input', required=True, type=str,
                        help="A single *_polys.json file or a directory containing them")
    parser.add_argument('--output_dir', default=None, type=str,
                        help="Where to write results (default: alongside each input file)")
    parser.add_argument('--up_axis', default=None, type=str, choices=('x', 'y', 'z'),
                        help="Which axis pointed vertically up in the ORIGINAL point "
                             "cloud. Default: read from the JSON's normalization.up_axis "
                             "(recorded by infer_pointcloud.py); falls back to 'y'. "
                             "Passing it here overrides the JSON.")
    parser.add_argument('--floor', default=None, type=float,
                        help="Override the floor height (original .ply units). "
                             "By default it is recovered from the JSON vertical range.")
    parser.add_argument('--ceiling', default=None, type=float,
                        help="Override the ceiling height (original .ply units).")
    parser.add_argument('--units', default='mm', type=str, choices=('mm', 'm'),
                        help="Which stored corners to read: world_mm or world_m.")
    return parser


def inverse_perm(up_axis):
    """Return inv such that orig = reordered_full[:, inv] undoes reorder_up_axis."""
    p = PERM[up_axis]
    inv = [0, 0, 0]
    for k, src in enumerate(p):
        inv[src] = k
    return inv


def recover_height_range(norm, floor_override=None, ceiling_override=None):
    """Recover (floor, ceiling) heights in original up-axis units from the JSON.

    Both extents live in the `ps` frame where ps[:, 2] = -height, so a high ps_z
    is a low height (floor) and vice versa: floor = -max_z, ceiling = -min_z.

    Preference order:
      1) `coords_pct_low/high` (robust percentiles, recorded by newer infer runs)
         -- excludes floor noise / furniture tops, no padding to undo.
      2) `min_coords/max_coords` extremes -- padded by +/-0.1*range, undone via
         R = (max-min)/1.2 ; raw_min = min+0.1R ; raw_max = max-0.1R.
    Per-axis overrides win over both.
    """
    pct_lo = norm.get('coords_pct_low')
    pct_hi = norm.get('coords_pct_high')
    if pct_lo is not None and pct_hi is not None:
        lo_z = float(np.asarray(pct_lo)[2])
        hi_z = float(np.asarray(pct_hi)[2])
        floor = -hi_z
        ceiling = -lo_z
        source = 'percentile'
    else:
        mn = float(np.asarray(norm['min_coords'])[2])
        mx = float(np.asarray(norm['max_coords'])[2])
        r = (mx - mn) / 1.2
        raw_min = mn + 0.1 * r
        raw_max = mx - 0.1 * r
        floor = -raw_max
        ceiling = -raw_min
        source = 'minmax'

    if floor_override is not None:
        floor = float(floor_override)
        source = 'override'
    if ceiling_override is not None:
        ceiling = float(ceiling_override)
        source = 'override'
    return floor, ceiling, source


def undo_yaw(xy, yaw_deg):
    """Rotate 2D floor-plane points by -yaw_deg about the origin.

    Exact inverse of infer_pointcloud.rotate_floor_plane, which applies +yaw via
    a standard rotation. Here angle = -yaw_deg:
        r0 = x*c - y*s ; r1 = x*s + y*c
    """
    r = np.deg2rad(-yaw_deg)
    c, s = np.cos(r), np.sin(r)
    xy = np.asarray(xy, dtype=np.float64)
    r0 = xy[:, 0] * c - xy[:, 1] * s
    r1 = xy[:, 0] * s + xy[:, 1] * c
    return np.stack([r0, r1], axis=1)


def corners_to_3d(world_xy, yaw_deg, up_axis, height):
    """Map floor-plane 2D corners to original 3D world coordinates at `height`.

    1) undo yaw -> reordered-frame floor-plane columns (0, 1)
    2) attach `height` as the reordered column 2 (the up axis)
    3) undo the column permutation -> original (x, y, z)
    Returns an (N, 3) float64 array.
    """
    r01 = undo_yaw(world_xy, yaw_deg)                 # (N, 2)
    h = np.full((len(r01), 1), float(height), dtype=np.float64)
    reordered_full = np.concatenate([r01, h], axis=1)  # (N, 3): (r0, r1, height)
    inv = inverse_perm(up_axis)
    return reordered_full[:, inv]


def write_obj(obj_path, rooms_3d):
    """Write floor/ceiling rings + wall quads as a Wavefront OBJ.

    rooms_3d: list of (floor_pts (N,3), ceiling_pts (N,3)). Per room we emit N
    floor verts then N ceiling verts, one wall quad per edge, and an n-gon face
    for the floor and the ceiling. OBJ indices are global and 1-based.
    """
    lines = ['# CAGE floor plan -> 3D world geometry (floor + ceiling wall rings)']
    vert_offset = 0
    for ridx, (floor_pts, ceil_pts) in enumerate(rooms_3d):
        n = len(floor_pts)
        if n < 3:
            continue
        lines.append('o room_{}'.format(ridx))
        for p in floor_pts:
            lines.append('v {:.6f} {:.6f} {:.6f}'.format(p[0], p[1], p[2]))
        for p in ceil_pts:
            lines.append('v {:.6f} {:.6f} {:.6f}'.format(p[0], p[1], p[2]))
        f0 = vert_offset + 1            # first floor vertex (1-based)
        c0 = vert_offset + n + 1        # first ceiling vertex
        # wall quads: floor edge (i -> i+1) lifted to the ceiling
        for i in range(n):
            j = (i + 1) % n
            lines.append('f {} {} {} {}'.format(f0 + i, f0 + j, c0 + j, c0 + i))
        # floor and ceiling faces (n-gons)
        lines.append('f ' + ' '.join(str(f0 + i) for i in range(n)))
        lines.append('f ' + ' '.join(str(c0 + i) for i in range(n)))
        vert_offset += 2 * n
    with open(obj_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def process_file(json_path, output_dir, up_axis_arg, floor_override, ceiling_override, units):
    with open(json_path, 'r') as f:
        data = json.load(f)

    norm = data.get('normalization', {})
    yaw = float(norm.get('applied_yaw_deg', 0.0))

    # up_axis: CLI override wins; otherwise use the value recorded in the JSON by
    # infer_pointcloud.py; fall back to 'y' for legacy JSON that lacks it.
    if up_axis_arg is not None:
        up_axis, up_src = up_axis_arg, 'cli'
    elif norm.get('up_axis') in ('x', 'y', 'z'):
        up_axis, up_src = norm['up_axis'], 'json'
    else:
        up_axis, up_src = 'y', 'default'

    floor, ceiling, h_src = recover_height_range(norm, floor_override, ceiling_override)

    corner_key = 'world_mm' if units == 'mm' else 'world_m'

    rooms_3d = []
    for room in data.get('rooms', []):
        world_xy = np.asarray(room[corner_key], dtype=np.float64)
        if world_xy.size == 0:
            floor_pts = ceil_pts = np.zeros((0, 3))
        else:
            floor_pts = corners_to_3d(world_xy, yaw, up_axis, floor)
            ceil_pts = corners_to_3d(world_xy, yaw, up_axis, ceiling)
        room['world_3d'] = {
            'floor': floor_pts.tolist(),
            'ceiling': ceil_pts.tolist(),
        }
        rooms_3d.append((floor_pts, ceil_pts))

    data['frame_info'] = {
        'up_axis': up_axis,
        'up_axis_source': up_src,
        'applied_yaw_deg': yaw,
        'floor': floor,
        'ceiling': ceiling,
        'height_source': h_src,
        'percentile': norm.get('percentile'),
        'units': units,
    }

    name = os.path.splitext(os.path.basename(json_path))[0]
    if name.endswith('_polys'):
        name = name[:-len('_polys')]
    out_json = os.path.join(output_dir, '{}_world3d.json'.format(name))
    out_obj = os.path.join(output_dir, '{}_world3d.obj'.format(name))
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=2)
    write_obj(out_obj, rooms_3d)

    print('{}: {} rooms, up_axis={} ({}), floor={:.1f} ceiling={:.1f} '
          '[{}] ({} units) -> {}, {}'.format(
              os.path.basename(json_path), len(rooms_3d), up_axis, up_src,
              floor, ceiling, h_src, units,
              os.path.basename(out_json), os.path.basename(out_obj)))


def main(args):
    if os.path.isdir(args.input):
        json_files = sorted(glob.glob(os.path.join(args.input, '*_polys.json')))
    else:
        json_files = [args.input]
    if len(json_files) == 0:
        raise FileNotFoundError('No *_polys.json files found at: {}'.format(args.input))

    for json_path in json_files:
        output_dir = args.output_dir or os.path.dirname(os.path.abspath(json_path))
        os.makedirs(output_dir, exist_ok=True)
        process_file(json_path, output_dir, args.up_axis,
                     args.floor, args.ceiling, args.units)

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CAGE floor-plan -> 3D world coordinates',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
