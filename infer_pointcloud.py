"""
Point cloud -> floor plan inference for CAGE.

Standalone inference script: takes a Structured3D-style point cloud (.ply),
projects it to a density map, runs the trained RoomFormer model, and writes the
reconstructed room polygons as a floor-plan image, a density overlay, and JSON
coordinates (both pixel-space and real-world scale).

NOTE: this script intentionally does NOT import `engine` or `eval`. Both of those
modules import `s3d_floorplan_eval` and call `MCSSOptions().parse()` at import
time, which parses argv and requires ground-truth evaluation data. The prediction
+ post-processing logic from `engine.evaluate_floor` (non-semantic branch,
engine.py:290-399) is therefore inlined here.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon

REPO = os.path.dirname(os.path.abspath(__file__))
# Make the data_preprocess utilities importable. stru3d_utils.py itself runs
# `sys.path.append('../data_preprocess')` at import time; inserting the absolute
# path first makes its `from common_utils import ...` resolve regardless of cwd.
sys.path.insert(0, os.path.join(REPO, 'data_preprocess'))
sys.path.insert(0, os.path.join(REPO, 'data_preprocess', 'stru3d'))

from common_utils import read_scene_pc                      # noqa: E402
from stru3d_utils import generate_density                   # noqa: E402
from models import build_model                              # noqa: E402
from util.edge_utils import (                               # noqa: E402
    remove_short_edges,
    get_corners_from_edges,
    merge_points,
    remove_rooms_with_iou,
    refine_rooms,
)
from util.plot_utils import plot_floorplan_with_regions, plot_room_map  # noqa: E402


def get_args_parser():
    # Model hyper-parameters mirror eval.py so the architecture matches the
    # checkpoint. Dataset/evaluation-only flags are dropped; inference I/O added.
    parser = argparse.ArgumentParser('CAGE point-cloud inference', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)

    # backbone
    parser.add_argument('--backbone', default='swinv2_L_192_22k', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--lr_backbone', default=2e-4, type=float)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str,
                        choices=('sine', 'learned'))
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float)
    parser.add_argument('--num_feature_levels', default=4, type=int)

    # transformer
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=800, type=int,
                        help="num_polys * max. number of corner per poly")
    parser.add_argument('--num_polys', default=20, type=int,
                        help="Maximum number of room polygons")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--query_pos_type', default='sine', type=str,
                        choices=('static', 'sine', 'none'))
    parser.add_argument('--with_poly_refine', default=True, action='store_true')
    parser.add_argument('--masked_attn', default=False, action='store_true')
    parser.add_argument('--semantic_classes', default=-1, type=int,
                        help="-1 = non-semantic floorplan (geometry only)")

    # aux
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true')
    parser.add_argument('--use_angle_loss', default=True, type=bool)

    # inference I/O
    parser.add_argument('--input', required=True, type=str,
                        help="A single .ply file or a directory containing .ply files")
    parser.add_argument('--checkpoint', default='checkpoint/CAGE_stru3d_swinv2.pth',
                        help='model checkpoint to load')
    parser.add_argument('--output_dir', default='infer_out', type=str,
                        help='directory to write results into')
    parser.add_argument('--device', default='cuda', help='cuda / cpu')
    parser.add_argument('--up_axis', default='y', type=str, choices=('x', 'y', 'z'),
                        help="Which axis of the input point cloud points vertically up. "
                             "generate_density projects the first two columns (the floor "
                             "plane) and treats the 3rd as height; columns are reordered so "
                             "the up-axis lands in the 3rd. Structured3D itself is z-up.")

    # yaw alignment: the cloud may be rotated about the vertical axis, so the
    # rendered density map is not axis-aligned. We estimate and undo that yaw.
    parser.add_argument('--no_align', dest='align', action='store_false',
                        help="Disable automatic yaw alignment of the floor plane.")
    parser.set_defaults(align=True)
    parser.add_argument('--rotation_deg', default=None, type=float,
                        help="Manually rotate the floor plane by this many degrees "
                             "(overrides automatic estimation).")
    parser.add_argument('--align_search_deg', default=45.0, type=float,
                        help="Half-range (deg) of the yaw search; walls have 90-deg "
                             "symmetry so +/-45 covers all orientations.")
    parser.add_argument('--align_step_deg', default=0.5, type=float,
                        help="Angular step (deg) of the yaw search.")

    # robust height/extent estimation: percentile bounds exclude outliers (floor
    # noise, furniture tops) so polys_to_3d.py can recover floor/ceiling without
    # min/max spikes. Recorded into the output JSON.
    parser.add_argument('--pct_low', default=2.0, type=float,
                        help="Lower percentile (0-100) for robust coordinate extent.")
    parser.add_argument('--pct_high', default=98.0, type=float,
                        help="Upper percentile (0-100) for robust coordinate extent.")

    return parser


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


def _projection_sharpness(coords_1d, bins=256):
    """Peakiness of a 1D point distribution: sum of squared normalized histogram.

    Maximal when points concentrate into a few bins, i.e. when walls parallel to
    this axis collapse onto shared coordinates.
    """
    h, _ = np.histogram(coords_1d, bins=bins)
    total = h.sum()
    if total == 0:
        return 0.0
    h = h.astype(np.float64) / total
    return float(np.sum(h * h))


def estimate_yaw(plane_xy, search_deg=45.0, step_deg=0.5, bins=256, max_points=100000):
    """Estimate the yaw (rotation about the vertical axis) that best axis-aligns
    the floor plane, using the Manhattan-world assumption.

    For each candidate angle we rotate the 2D points and score how sharply they
    project onto the x and y axes; the best angle is the one whose rotation makes
    walls parallel to the image axes. Returns the angle to ROTATE BY to correct.
    """
    pts = plane_xy
    if len(pts) > max_points:
        # subsample for speed; orientation is a global property
        idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
        pts = pts[idx]

    best_angle, best_score = 0.0, -1.0
    angles = np.arange(-search_deg, search_deg + 1e-9, step_deg)
    for a in angles:
        r = np.deg2rad(a)
        c, s = np.cos(r), np.sin(r)
        x = pts[:, 0] * c - pts[:, 1] * s
        y = pts[:, 0] * s + pts[:, 1] * c
        score = _projection_sharpness(x, bins) + _projection_sharpness(y, bins)
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


def load_input(ply_path, up_axis='y', align=True, rotation_deg=None,
               search_deg=45.0, step_deg=0.5, pct_low=2.0, pct_high=98.0):
    """Read a .ply point cloud and build the model input density map.

    Replicates exactly the tensor the eval pipeline feeds the model:
    data_preprocess (generate_density -> export_density as uint8 PNG) followed by
    datasets/poly_data.py (img / 255). Before projection, the floor plane is
    yaw-corrected so the rendered density map is axis-aligned.

    Records into `norm` the up_axis and robust percentile extents so the inverse
    script (polys_to_3d.py) can recover the original frame and floor/ceiling.
    """
    points = read_scene_pc(ply_path)
    xyz = points[:, :3].astype(np.float32)
    xyz = reorder_up_axis(xyz, up_axis)

    applied_yaw = 0.0
    if rotation_deg is not None:
        applied_yaw = float(rotation_deg)
    elif align:
        applied_yaw = estimate_yaw(xyz[:, :2], search_deg=search_deg, step_deg=step_deg)
    if applied_yaw != 0.0:
        xyz = rotate_floor_plane(xyz, applied_yaw)

    density, norm = generate_density(xyz, width=256, height=256)   # density in [0, 1]
    norm['applied_yaw_deg'] = applied_yaw
    norm['up_axis'] = up_axis

    # Robust extent via percentiles, in the SAME frame as min/max_coords:
    # generate_density builds `ps` with ps[:,0:2] = xyz[:,0:2] and ps[:,2] = -height.
    # Percentiles (unlike min/max) are not padded and reject outliers, giving
    # polys_to_3d.py a cleaner floor/ceiling than the raw extremes.
    ps = xyz.astype(np.float64).copy()
    ps[:, 2] = -ps[:, 2]
    norm['coords_pct_low'] = np.percentile(ps, pct_low, axis=0)
    norm['coords_pct_high'] = np.percentile(ps, pct_high, axis=0)
    norm['percentile'] = {'low': float(pct_low), 'high': float(pct_high)}

    density_u8 = (density * 255).astype(np.uint8)                  # cf. export_density
    img = (1 / 255.) * torch.as_tensor(np.expand_dims(density_u8, 0)).float()  # (1,256,256)
    return img, density_u8, norm


@torch.no_grad()
def predict_polys(model, img, device):
    """Run the model and reconstruct room polygons.

    Inlined non-semantic branch of engine.evaluate_floor (engine.py:290-399).
    Returns a list of room polygons (each np.int32 array of shape (N, 2) in
    256x256 pixel space).
    """
    outputs, _ = model([img.to(device)])

    pred_logits = torch.sigmoid(outputs['pred_logits'])   # (1, num_polys, queries_per_poly)
    pred_corners = outputs['pred_coords']                 # (1, num_polys, queries_per_poly, 4)
    fg_mask = pred_logits > 0.5                            # select valid corners

    # single scene
    fg_mask_per_scene = fg_mask[0]
    pred_logits_per_scene = pred_logits[0]
    pred_corners_per_scene = pred_corners[0]

    room_polys = []
    for j in range(fg_mask_per_scene.shape[0]):
        fg_mask_per_room = fg_mask_per_scene[j]
        pred_logits_per_room = pred_logits_per_scene[j][fg_mask_per_room].cpu().numpy()
        valid_corners_per_room = pred_corners_per_scene[j][fg_mask_per_room]

        if len(valid_corners_per_room) > 0:
            corners = (valid_corners_per_room * 255).cpu().numpy()
            corners, pred_logits_per_room = remove_short_edges(corners, pred_logits_per_room)
            corners = np.around(corners).astype(np.int32)
            corners = get_corners_from_edges(corners, pred_logits_per_room, 10)
            corners = np.around(corners).astype(np.int32)
            corners = merge_points(corners, 2)

            if len(corners) >= 4:
                room_polys.append(corners)

    # shapely refinement (engine.py:374-399)
    shapely_polygons = []
    for np_array in room_polys:
        pts = [tuple(point) for point in np_array]
        if len(pts) > 0 and pts[0] != pts[-1]:
            pts.append(pts[0])
        shapely_polygons.append(Polygon(pts))

    try:
        shapely_polygons = remove_rooms_with_iou(shapely_polygons)
        polygon_list, _ = refine_rooms(shapely_polygons, False)
        room_polys = [np.array(p.exterior.coords, dtype=np.int32)[:-1] for p in polygon_list]
    except Exception:
        # keep the un-refined polygons on failure, mirroring engine's bare except
        pass

    return room_polys


def pixel_to_world(poly_px, norm):
    """Inverse-project pixel-space polygon corners to real-world coordinates.

    generate_density maps real x/y -> [0,1] via (p - min)/(max - min), then to
    pixels by *image_res (256). engine decodes corners with *255, so we invert
    with /255 to match the actual pixel values. The 255 vs 256 discrepancy is a
    sub-0.4% scale ambiguity inherent to the codebase. Units are the original
    .ply units (Structured3D = millimetres).
    """
    mn = np.asarray(norm['min_coords'], dtype=np.float64)
    mx = np.asarray(norm['max_coords'], dtype=np.float64)
    poly = np.asarray(poly_px, dtype=np.float64)
    wx = mn[0] + (poly[:, 0] / 255.) * (mx[0] - mn[0])
    wy = mn[1] + (poly[:, 1] / 255.) * (mx[1] - mn[1])
    return np.stack([wx, wy], axis=1)


def save_outputs(name, room_polys, density_u8, norm, output_dir):
    # 1) floor-plan visualization
    floorplan_map = plot_floorplan_with_regions([np.array(r) for r in room_polys], scale=1000)
    cv2.imwrite(os.path.join(output_dir, '{}_pred_floorplan.png'.format(name)), floorplan_map)

    # 2) raw density map
    cv2.imwrite(os.path.join(output_dir, '{}_density.png'.format(name)), density_u8)

    # 3) predicted polygons overlaid on the density map (engine.py:459-469)
    density_map = np.repeat(density_u8[:, :, None].astype(np.float32), 3, axis=2)
    pred_room_map = np.zeros([256, 256, 3])
    for room_poly in room_polys:
        pred_room_map = plot_room_map(room_poly, pred_room_map)
    pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
    cv2.imwrite(os.path.join(output_dir, '{}_pred_room_map.png'.format(name)), pred_room_map)

    # 4) polygon coordinates (pixel + real-world scale)
    rooms = []
    for r in room_polys:
        r = np.asarray(r)
        world_mm = pixel_to_world(r, norm)
        rooms.append({
            'pixel': r.astype(int).tolist(),
            'world_mm': world_mm.tolist(),
            'world_m': (world_mm / 1000.).tolist(),
        })
    result = {
        'num_rooms': len(rooms),
        'rooms': rooms,
        'normalization': {
            'min_coords': np.asarray(norm['min_coords']).tolist(),
            'max_coords': np.asarray(norm['max_coords']).tolist(),
            'image_res': np.asarray(norm['image_res']).tolist(),
            'applied_yaw_deg': norm.get('applied_yaw_deg', 0.0),
            'up_axis': norm.get('up_axis'),
            'coords_pct_low': (np.asarray(norm['coords_pct_low']).tolist()
                               if 'coords_pct_low' in norm else None),
            'coords_pct_high': (np.asarray(norm['coords_pct_high']).tolist()
                                if 'coords_pct_high' in norm else None),
            'percentile': norm.get('percentile'),
        },
    }
    with open(os.path.join(output_dir, '{}_polys.json'.format(name)), 'w') as f:
        json.dump(result, f, indent=2)


def main(args):
    device = torch.device(args.device)

    # build model
    model = build_model(args, train=False)
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys
                       if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    model.eval()

    # collect input .ply files
    if os.path.isdir(args.input):
        ply_files = sorted(glob.glob(os.path.join(args.input, '*.ply')))
    else:
        ply_files = [args.input]
    if len(ply_files) == 0:
        raise FileNotFoundError('No .ply files found at: {}'.format(args.input))

    os.makedirs(args.output_dir, exist_ok=True)

    for ply_path in ply_files:
        name = os.path.splitext(os.path.basename(ply_path))[0]
        print('Processing {} ...'.format(ply_path))
        img, density_u8, norm = load_input(
            ply_path, up_axis=args.up_axis, align=args.align,
            rotation_deg=args.rotation_deg, search_deg=args.align_search_deg,
            step_deg=args.align_step_deg,
            pct_low=args.pct_low, pct_high=args.pct_high)
        print('  applied yaw correction: {:.2f} deg'.format(norm['applied_yaw_deg']))
        room_polys = predict_polys(model, img, device)
        print('  -> {} room polygons'.format(len(room_polys)))
        save_outputs(name, room_polys, density_u8, norm, args.output_dir)

    print('Done. Results written to: {}'.format(args.output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CAGE point-cloud inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
