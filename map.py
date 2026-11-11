"""
generate_pointcloud.py
======================
Generates dense point clouds from TUM RGB-D sequences using
ORB-SLAM3 camera trajectories.

Produces TWO point clouds for comparison:
  1. Without mask  — includes ghost points from dynamic objects
  2. With mask     — clean static map, dynamic pixels removed

Usage:
    python3 generate_pointcloud.py \
        --dataset    /home/fyp/pytorch-liteflownet/data/fr3_walking_xyz \
        --traj_no_mask  /home/fyp/pytorch-liteflownet/ORB_SLAM3/w_xyz/no_mask/CameraTrajectory.txt \
        --traj_mask     /home/fyp/pytorch-liteflownet/ORB_SLAM3/w_xyz/best/CameraTrajectory.txt \
        --masks      /home/fyp/pytorch-liteflownet/results_fr3_walking_xyz3/masks \
        --output_dir ./pointclouds \
        --every_n    5

Requirements:
    pip install open3d numpy opencv-python
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("[ERROR] open3d not installed. Run: pip install open3d")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# TUM Camera Intrinsics
# ─────────────────────────────────────────────────────────────────────────────
TUM_INTRINSICS = {
    "fr1": dict(fx=517.3, fy=516.5, cx=318.6, cy=255.3),
    "fr2": dict(fx=520.9, fy=521.0, cx=325.1, cy=249.7),
    "fr3": dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6),
}
DEPTH_SCALE = 5000.0


# ─────────────────────────────────────────────────────────────────────────────
# Load trajectory (TUM format: ts tx ty tz qx qy qz qw)
# ─────────────────────────────────────────────────────────────────────────────
def load_trajectory(traj_file):
    """Returns dict: {timestamp_str: 4x4 transformation matrix}"""
    from scipy.spatial.transform import Rotation
    poses = {}
    with open(traj_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            ts = parts[0]
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), \
                              float(parts[6]), float(parts[7])
            # Convert quaternion to rotation matrix
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = [tx, ty, tz]
            poses[ts] = T
    return poses


# ─────────────────────────────────────────────────────────────────────────────
# Load TUM rgb/depth associations
# ─────────────────────────────────────────────────────────────────────────────
def load_associations(dataset_dir, max_diff=0.02):
    """Match RGB frames to nearest depth frames by timestamp."""
    def parse(txt):
        entries = {}
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                entries[float(parts[0])] = parts[1]
        return entries

    rgb_entries   = parse(os.path.join(dataset_dir, 'rgb.txt'))
    depth_entries = parse(os.path.join(dataset_dir, 'depth.txt'))
    depth_ts_list = sorted(depth_entries.keys())

    matched = []
    for rgb_t, rgb_path in sorted(rgb_entries.items()):
        closest_d = min(depth_ts_list, key=lambda x: abs(x - rgb_t))
        if abs(closest_d - rgb_t) > max_diff:
            continue
        matched.append({
            'ts':         f"{rgb_t:.6f}",
            'rgb_path':   os.path.join(dataset_dir, rgb_path),
            'depth_path': os.path.join(dataset_dir, depth_entries[closest_d]),
        })
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Build point cloud from one frame
# ─────────────────────────────────────────────────────────────────────────────
def frame_to_pointcloud(rgb, depth, pose, intr,
                         mask=None, max_depth=4.0):
    """
    Projects RGB-D frame into world coordinates using camera pose.
    If mask provided, dynamic pixels (white) are skipped.
    Returns Nx3 points and Nx3 colours.
    """
    fx, fy = intr['fx'], intr['fy']
    cx, cy = intr['cx'], intr['cy']
    H, W   = depth.shape

    depth_m = depth.astype(np.float32) / DEPTH_SCALE

    # Create pixel grid
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))

    # Valid pixel mask: depth > 0 and depth < max_depth
    valid = (depth_m > 0) & (depth_m < max_depth)

    # Apply dynamic mask — skip white (dynamic) pixels
    if mask is not None:
        valid = valid & (mask < 127)

    # Back-project to camera coordinates
    Z = depth_m[valid]
    X = (uu[valid] - cx) * Z / fx
    Y = (vv[valid] - cy) * Z / fy

    # Camera coordinates → homogeneous
    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)  # Nx4

    # Transform to world coordinates using pose
    pts_world = (pose @ pts_cam.T).T[:, :3]  # Nx3

    # Get colours
    rgb_valid = rgb[valid]  # Nx3 BGR
    colours   = rgb_valid[:, ::-1].astype(np.float64) / 255.0  # Nx3 RGB

    return pts_world, colours


# ─────────────────────────────────────────────────────────────────────────────
# Build full point cloud
# ─────────────────────────────────────────────────────────────────────────────
def build_pointcloud(associations, poses, intr, masks_dir=None,
                     every_n=5, max_depth=4.0):
    """
    Builds dense point cloud from all frames with known poses.
    every_n: use every Nth frame (reduce memory/time)
    """
    all_points  = []
    all_colours = []
    used_frames = 0

    for i, frame in enumerate(associations):
        if i % every_n != 0:
            continue

        ts = frame['ts']

        # Find matching pose (nearest timestamp)
        if not poses:
            continue
        pose_ts = min(poses.keys(), key=lambda x: abs(float(x) - float(ts)))
        if abs(float(pose_ts) - float(ts)) > 0.05:
            continue  # no close pose found

        pose = poses[pose_ts]

        # Load RGB and depth
        rgb   = cv2.imread(frame['rgb_path'])
        depth = cv2.imread(frame['depth_path'], cv2.IMREAD_UNCHANGED)

        if rgb is None or depth is None:
            continue

        # Load mask if available
        mask = None
        if masks_dir is not None:
            mask_file = os.path.join(masks_dir, f"{ts}.png")
            if os.path.exists(mask_file):
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        # Build points for this frame
        pts, cols = frame_to_pointcloud(rgb, depth, pose, intr,
                                         mask=mask, max_depth=max_depth)
        if len(pts) > 0:
            all_points.append(pts)
            all_colours.append(cols)
            used_frames += 1

        if used_frames % 20 == 0:
            print(f"  Processed {used_frames} frames "
                  f"({len(all_points)*every_n}/{len(associations)} total)...")

    if not all_points:
        print("[WARN] No points generated!")
        return None

    points  = np.vstack(all_points)
    colours = np.vstack(all_colours)

    # Limit total points to prevent OOM
    max_pts = 3_000_000
    if len(points) > max_pts:
        idx     = np.random.choice(len(points), max_pts, replace=False)
        points  = points[idx]
        colours = colours[idx]
        print(f"  Subsampled to {max_pts:,} points (memory limit)")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colours)

    # Voxel downsample to reduce size
    pcd = pcd.voxel_down_sample(voxel_size=0.02)

    print(f"  Total points: {len(pcd.points):,}")
    return pcd


# ─────────────────────────────────────────────────────────────────────────────
# Visualise side by side
# ─────────────────────────────────────────────────────────────────────────────
def render_pcd_matplotlib(pcd, title, output_path, color):
    """
    Renders point cloud with 4 views matching reference paper style.
    Main view = perspective-like front view (most informative).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    # Clip outliers for cleaner view
    for axis in range(3):
        p5  = np.percentile(pts[:, axis], 2)
        p95 = np.percentile(pts[:, axis], 98)
        keep = (pts[:, axis] > p5) & (pts[:, axis] < p95)
        pts  = pts[keep]
        cols = cols[keep]

    # Subsample
    n   = min(len(pts), 200000)
    idx = np.random.choice(len(pts), n, replace=False)
    pts  = pts[idx]
    cols = cols[idx]

    fig = plt.figure(figsize=(22, 10), facecolor='#111111')
    fig.suptitle(title, color='white', fontsize=15, fontweight='bold', y=0.98)

    # ── Main view: perspective front (like reference image) ───────────────
    ax_main = fig.add_subplot(1, 2, 1, projection='3d')
    ax_main.set_facecolor('#111111')
    ax_main.scatter(pts[:,0], pts[:,2], -pts[:,1],
                    c=cols, s=0.4, alpha=0.7, depthshade=True)
    ax_main.set_xlabel('X', color='grey', fontsize=8)
    ax_main.set_ylabel('Z', color='grey', fontsize=8)
    ax_main.set_zlabel('Y', color='grey', fontsize=8)
    ax_main.tick_params(colors='grey', labelsize=6)
    ax_main.set_title('Perspective View', color='white', fontsize=11, pad=10)
    # Set viewing angle similar to reference
    ax_main.view_init(elev=15, azim=-60)
    ax_main.set_box_aspect([1,1,0.6])
    for pane in [ax_main.xaxis.pane, ax_main.yaxis.pane, ax_main.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#333')

    # ── Right panel: 3 orthographic views ─────────────────────────────────
    gs_right = fig.add_gridspec(3, 2, left=0.52, right=0.98,
                                 top=0.92, bottom=0.05,
                                 hspace=0.35, wspace=0.3)

    views_2d = [
        ('Top-Down (X-Z)', pts[:,0], pts[:,2],  'X (m)', 'Z (m)'),
        ('Front (X-Y)',    pts[:,0], -pts[:,1], 'X (m)', 'Y (m)'),
        ('Side (Z-Y)',     pts[:,2], -pts[:,1], 'Z (m)', 'Y (m)'),
    ]

    for row, (vname, x, y, xl, yl) in enumerate(views_2d):
        ax = fig.add_subplot(gs_right[row, :])
        ax.set_facecolor('#111111')
        ax.scatter(x, y, c=cols, s=0.2, alpha=0.6)
        ax.set_title(vname, color='white', fontsize=9)
        ax.set_xlabel(xl, color='grey', fontsize=7)
        ax.set_ylabel(yl, color='grey', fontsize=7)
        ax.tick_params(colors='grey', labelsize=6)
        ax.set_aspect('equal', adjustable='datalim')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#111111')
    plt.close()
    print(f"[✓] View saved → {output_path}")


def visualize_comparison(pcd_no_mask, pcd_with_mask, output_dir):
    """Save both point clouds and generate comparison images."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as PLY files
    no_mask_ply   = os.path.join(output_dir, 'pointcloud_no_mask.ply')
    with_mask_ply = os.path.join(output_dir, 'pointcloud_with_mask.ply')

    if pcd_no_mask:
        o3d.io.write_point_cloud(no_mask_ply, pcd_no_mask)
        print(f"[✓] Saved → {no_mask_ply}")

    if pcd_with_mask:
        o3d.io.write_point_cloud(with_mask_ply, pcd_with_mask)
        print(f"[✓] Saved → {with_mask_ply}")

    # Render views using matplotlib (WSL compatible)
    print("\nGenerating comparison images...")
    if pcd_no_mask:
        render_pcd_matplotlib(pcd_no_mask,
                              "Point Cloud WITHOUT Mask (ghosted dynamic objects)",
                              os.path.join(output_dir, "view_no_mask.png"),
                              color='red')
    if pcd_with_mask:
        render_pcd_matplotlib(pcd_with_mask,
                              "Point Cloud WITH Mask (clean static map)",
                              os.path.join(output_dir, "view_with_mask.png"),
                              color='green')

    # Side-by-side comparison
    img1_path = os.path.join(output_dir, "view_no_mask.png")
    img2_path = os.path.join(output_dir, "view_with_mask.png")

    if os.path.exists(img1_path) and os.path.exists(img2_path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img1 = mpimg.imread(img1_path)
        img2 = mpimg.imread(img2_path)

        fig, axes = plt.subplots(2, 1, figsize=(20, 14), facecolor='#1a1a1a')
        fig.suptitle("Point Cloud Comparison — Without Mask vs With Mask",
                     color='white', fontsize=16, fontweight='bold')

        axes[0].imshow(img1); axes[0].axis('off')
        axes[0].set_title("WITHOUT Mask — dynamic objects cause ghost points",
                           color='#ff6666', fontsize=12, pad=8)

        axes[1].imshow(img2); axes[1].axis('off')
        axes[1].set_title("WITH Mask — clean static environment map",
                           color='#66ff66', fontsize=12, pad=8)

        plt.tight_layout()
        comp_path = os.path.join(output_dir, "comparison.png")
        plt.savefig(comp_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        print(f"[✓] Comparison saved → {comp_path}")




# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Dense point cloud comparison — with vs without dynamic mask")
    parser.add_argument('--dataset',       required=True,
                        help='TUM dataset folder')
    parser.add_argument('--traj_no_mask',  required=True,
                        help='CameraTrajectory.txt WITHOUT masks')
    parser.add_argument('--traj_mask',     required=True,
                        help='CameraTrajectory.txt WITH masks')
    parser.add_argument('--masks',         required=True,
                        help='Masks folder (PNG files named by timestamp)')
    parser.add_argument('--output_dir',    default='./pointclouds',
                        help='Output directory for PLY files and images')
    parser.add_argument('--sequence',      default='fr3',
                        choices=['fr1','fr2','fr3'],
                        help='TUM sequence (determines intrinsics)')
    parser.add_argument('--every_n',       type=int, default=5,
                        help='Use every Nth frame (default 5, lower=denser but slower)')
    parser.add_argument('--max_depth',     type=float, default=4.0,
                        help='Max depth in metres (default 4.0)')
    args = parser.parse_args()

    intr = TUM_INTRINSICS[args.sequence]

    print(f"\n{'='*55}")
    print(f"  Dense Point Cloud Comparison")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Sequence : {args.sequence.upper()}")
    print(f"  Every N  : {args.every_n} frames")
    print(f"  Max depth: {args.max_depth}m")
    print(f"{'='*55}\n")

    # Check scipy available
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("[ERROR] scipy not installed. Run: pip install scipy")
        sys.exit(1)

    # Load data
    print("Loading associations...")
    associations = load_associations(args.dataset)
    print(f"  Found {len(associations)} RGB-D pairs")

    print("\nLoading trajectories...")
    poses_no_mask  = load_trajectory(args.traj_no_mask)
    poses_with_mask = load_trajectory(args.traj_mask)
    print(f"  No-mask trajectory : {len(poses_no_mask)} poses")
    print(f"  Masked trajectory  : {len(poses_with_mask)} poses")

    # Build point clouds
    print("\n[1/2] Building point cloud WITHOUT mask...")
    pcd_no_mask = build_pointcloud(
        associations, poses_no_mask, intr,
        masks_dir=None,
        every_n=args.every_n,
        max_depth=args.max_depth
    )

    print("\n[2/2] Building point cloud WITH mask...")
    pcd_with_mask = build_pointcloud(
        associations, poses_with_mask, intr,
        masks_dir=args.masks,
        every_n=args.every_n,
        max_depth=args.max_depth
    )

    # Save and visualize
    print("\nSaving outputs...")
    visualize_comparison(pcd_no_mask, pcd_with_mask, args.output_dir)

    print(f"\n[✓] All done! Results in: {os.path.abspath(args.output_dir)}")
    print("\nTo view interactively:")
    print(f"  python3 -c \"import open3d as o3d; "
          f"o3d.visualization.draw_geometries("
          f"[o3d.io.read_point_cloud('{args.output_dir}/pointcloud_with_mask.ply')])\"")


if __name__ == "__main__":
    main()
