"""
plot_trajectory.py
==================
Plots camera trajectory comparison using ORB-SLAM3's own alignment.
Reuses the align() function from evaluate_ate_scale.py for correct alignment.

Usage — Single dataset:
    python3 plot_trajectory.py \
        --gt        /path/to/groundtruth.txt \
        --no_mask   /path/to/no_mask/CameraTrajectory.txt \
        --with_mask /path/to/with_mask/CameraTrajectory.txt \
        --name      "fr3/w/xyz" \
        --output    ./trajectory_xyz.png

Usage — Multiple datasets grid:
    python3 plot_trajectory.py --multi --output ./trajectory_all.png
    (edit DATASETS list inside script first)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-DATASET CONFIG — edit paths for your setup
# ─────────────────────────────────────────────────────────────────────────────
BASE      = '/home/fyp/pytorch-liteflownet'
SLAM_BASE = f'{BASE}/ORB_SLAM3'
DATA_BASE = f'{BASE}/data'

DATASETS = [
    {
        'name':      'fr3/w/half',
        'gt':        f'{DATA_BASE}/fr3_walking_halfsphere/groundtruth.txt',
        'no_mask':   f'{SLAM_BASE}/w_half/no_mask/CameraTrajectory.txt',
        'with_mask': f'{SLAM_BASE}/w_half/best/CameraTrajectory.txt',
    },
    {
        'name':      'fr3/w/rpy',
        'gt':        f'{DATA_BASE}/fr3_walking_rpy/groundtruth.txt',
        'no_mask':   f'{SLAM_BASE}/w_rpy/no_mask/CameraTrajectory.txt',
        'with_mask': f'{SLAM_BASE}/w_rpy/best/CameraTrajectory.txt',
    },
    {
        'name':      'fr3/w/static',
        'gt':        f'{DATA_BASE}/fr3_walking_static/groundtruth.txt',
        'no_mask':   f'{SLAM_BASE}/w_static/no_mask/CameraTrajectory.txt',
        'with_mask': f'{SLAM_BASE}/w_static/best/CameraTrajectory.txt',
    },
    {
        'name':      'fr3/w/xyz',
        'gt':        f'{DATA_BASE}/fr3_walking_xyz/groundtruth.txt',
        'no_mask':   f'{SLAM_BASE}/w_xyz/no_mask/CameraTrajectory.txt',
        'with_mask': f'{SLAM_BASE}/w_xyz/best/CameraTrajectory.txt',
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Alignment — same Horn method as ORB-SLAM3 evaluate_ate_scale.py
# ─────────────────────────────────────────────────────────────────────────────
def align(model, data):
    """Horn's closed-form alignment (identical to ORB-SLAM3 evaluate_ate_scale.py)."""
    model_zerocentered = model - model.mean(1)
    data_zerocentered  = data  - data.mean(1)

    W = np.zeros((3, 3))
    for col in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, col], data_zerocentered[:, col])

    U, d, Vh = np.linalg.svd(W.T)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = np.matrix(U) * np.matrix(S) * np.matrix(Vh)

    dots = norms = 0.0
    rotmodel = rot * np.matrix(model_zerocentered)
    for col in range(data_zerocentered.shape[1]):
        dots  += np.dot(np.array(data_zerocentered)[:, col],
                        np.array(rotmodel)[:, col])
        norms += np.linalg.norm(np.array(model_zerocentered)[:, col]) ** 2

    scale  = float(dots / norms)
    trans  = np.array(data.mean(1)) - scale * np.array(rot * np.matrix(model.mean(1)))
    return rot, trans, scale


# ─────────────────────────────────────────────────────────────────────────────
# File readers
# ─────────────────────────────────────────────────────────────────────────────
def read_trajectory(filepath):
    traj = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            traj[float(p[0])] = [float(x) for x in p[1:]]
    return traj


def associate(first, second, max_diff=0.02):
    matches = []
    second_keys = np.array(sorted(second.keys()))
    for ts1 in sorted(first.keys()):
        j = np.argmin(np.abs(second_keys - ts1))
        if abs(second_keys[j] - ts1) < max_diff:
            matches.append((ts1, second_keys[j]))
    return matches


def plot_segments(ax, stamps, xyz, style, color, label, zorder=3):
    """Plot trajectory with gap detection."""
    if len(stamps) < 2:
        return
    interval = np.median([stamps[i+1]-stamps[i] for i in range(len(stamps)-1)])
    x, y, last = [], [], stamps[0]
    for i, ts in enumerate(stamps):
        if ts - last < 2 * interval:
            x.append(xyz[0, i])
            y.append(xyz[1, i])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label, zorder=zorder)
            label = ''
            x, y = [], []
        last = ts
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label, zorder=zorder)


# ─────────────────────────────────────────────────────────────────────────────
# Core plot function
# ─────────────────────────────────────────────────────────────────────────────
def plot_traj_aligned(ax, gt_file, est_file, title, show_legend=False):
    """Plot aligned trajectory using Horn's method — same as ATE evaluation."""
    if not os.path.exists(est_file):
        ax.text(0.5, 0.5, f'File not found:\n{os.path.basename(est_file)}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8, color='red')
        ax.set_title(title, fontsize=9, fontweight='bold')
        return

    if not os.path.exists(gt_file):
        ax.text(0.5, 0.5, 'GT not found', ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='red')
        return

    gt_dict  = read_trajectory(gt_file)
    est_dict = read_trajectory(est_file)
    matches  = associate(gt_dict, est_dict, max_diff=0.02)

    if len(matches) < 2:
        ax.text(0.5, 0.5, 'No timestamp matches',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        return

    # Matched 3xN matrices for alignment
    gt_xyz  = np.matrix([[gt_dict[a][0],  gt_dict[a][1],  gt_dict[a][2]]
                          for a, b in matches]).T
    est_xyz = np.matrix([[est_dict[b][0], est_dict[b][1], est_dict[b][2]]
                          for a, b in matches]).T

    rot, trans, scale = align(est_xyz, gt_xyz)

    # Apply alignment to matched points (for difference lines)
    est_aligned = scale * np.array(rot * est_xyz) + trans

    # Apply alignment to FULL estimated trajectory
    est_stamps = sorted(est_dict.keys())
    gt_stamps  = sorted(gt_dict.keys())

    est_full = np.matrix([[est_dict[s][0], est_dict[s][1], est_dict[s][2]]
                           for s in est_stamps]).T
    gt_full  = np.matrix([[gt_dict[s][0],  gt_dict[s][1],  gt_dict[s][2]]
                           for s in gt_stamps]).T

    est_full_aligned = scale * np.array(rot * est_full) + trans

    # Difference lines
    gt_arr  = np.array(gt_xyz)
    est_arr = np.array(est_aligned)
    label_d = 'difference'
    for i in range(gt_arr.shape[1]):
        ax.plot([gt_arr[0,i], est_arr[0,i]],
                [gt_arr[1,i], est_arr[1,i]],
                '-', color='red', linewidth=0.5, alpha=0.4,
                label=label_d, zorder=1)
        label_d = ''

    # Full trajectories
    plot_segments(ax, gt_stamps,  np.array(gt_full),         '-', 'black', 'ground truth', zorder=4)
    plot_segments(ax, est_stamps, est_full_aligned,           '-', 'blue',  'estimated',    zorder=3)

    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.set_xlabel('x (m)', fontsize=7)
    ax.set_ylabel('y (m)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.2, linewidth=0.5)

    if show_legend:
        ax.legend(loc='best', fontsize=7,
                  facecolor='white', edgecolor='grey', framealpha=0.8)

    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')


# ─────────────────────────────────────────────────────────────────────────────
# Single dataset — 1×2
# ─────────────────────────────────────────────────────────────────────────────
def plot_single_dataset(gt, no_mask, with_mask, name, output):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    fig.suptitle(f"Camera Trajectory Comparison — {name}",
                 fontsize=14, fontweight='bold', y=1.02)

    plot_traj_aligned(axes[0], gt, no_mask,
                      f"Baseline (No Mask)\n{name}",        show_legend=True)
    plot_traj_aligned(axes[1], gt, with_mask,
                      f"Proposed System (With Mask)\n{name}", show_legend=True)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-dataset grid — 2×N
# ─────────────────────────────────────────────────────────────────────────────
def plot_multi_dataset(datasets, output):
    n     = len(datasets)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 10), facecolor='white')
    fig.suptitle("Camera Trajectory — Baseline vs Proposed System",
                 fontsize=14, fontweight='bold', y=1.01)

    for col, ds in enumerate(datasets):
        plot_traj_aligned(axes[0, col], ds['gt'], ds['no_mask'],
                          ds['name'], show_legend=(col == 0))
        plot_traj_aligned(axes[1, col], ds['gt'], ds['with_mask'],
                          ds['name'], show_legend=(col == 0))

    axes[0, 0].set_ylabel("Baseline (No Mask)\ny (m)",    fontsize=9, fontweight='bold')
    axes[1, 0].set_ylabel("Proposed System\ny (m)", fontsize=9, fontweight='bold')

    handles = [
        mpatches.Patch(color='black', label='ground truth'),
        mpatches.Patch(color='blue',  label='estimated'),
        mpatches.Patch(color='red',   label='difference'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Multi-dataset plot saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Plot aligned camera trajectory comparison')
    parser.add_argument('--gt',        help='Ground truth file')
    parser.add_argument('--no_mask',   help='Trajectory WITHOUT mask')
    parser.add_argument('--with_mask', help='Trajectory WITH mask')
    parser.add_argument('--name',      default='fr3/w/xyz',
                        help='Dataset name for title')
    parser.add_argument('--output',    default='./trajectory.png',
                        help='Output image path')
    parser.add_argument('--multi',     action='store_true',
                        help='Plot all datasets in DATASETS list as grid')
    args = parser.parse_args()

    if args.multi:
        print("Generating multi-dataset trajectory grid...")
        plot_multi_dataset(DATASETS, args.output)
    else:
        if not all([args.gt, args.no_mask, args.with_mask]):
            print("[ERROR] Provide --gt --no_mask --with_mask for single mode")
            parser.print_help()
            return
        plot_single_dataset(
            gt=args.gt,
            no_mask=args.no_mask,
            with_mask=args.with_mask,
            name=args.name,
            output=args.output
        )


if __name__ == "__main__":
    main()
