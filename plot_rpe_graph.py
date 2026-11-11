"""
plot_rpe_overtime.py
====================
Plots RPE over time using IDENTICAL logic to ORB-SLAM3's evaluate_rpe.py.
Uses transform44(), evaluate_trajectory(), compute_distance(), compute_angle()
from the official TUM evaluation script to ensure values match exactly.

Usage — All 4 walking sequences:
    python3 plot_rpe_overtime.py --multi --output ./rpe_overtime_all.png

Usage — Single sequence:
    python3 plot_rpe_overtime.py \
        --gt    /path/to/groundtruth.txt \
        --traj  /path/to/CameraTrajectory.txt \
        --name  "fr3/w/xyz" \
        --output ./rpe_xyz.png
"""

import os
import argparse
import random
import numpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit paths for your setup
# ─────────────────────────────────────────────────────────────────────────────
BASE      = '/home/fyp/pytorch-liteflownet'
SLAM_BASE = f'{BASE}/ORB_SLAM3'
DATA_BASE = f'{BASE}/data'

DATASETS = [
    {
        'name':  'fr3/w/xyz',
        'gt':    f'{DATA_BASE}/fr3_walking_xyz/groundtruth.txt',
        'traj':  f'{SLAM_BASE}/w_xyz/best/CameraTrajectory.txt',
        'color': '#2980B9',
    },
    {
        'name':  'fr3/w/rpy',
        'gt':    f'{DATA_BASE}/fr3_walking_rpy/groundtruth.txt',
        'traj':  f'{SLAM_BASE}/w_rpy/best/CameraTrajectory.txt',
        'color': '#E74C3C',
    },
    {
        'name':  'fr3/w/static',
        'gt':    f'{DATA_BASE}/fr3_walking_static/groundtruth.txt',
        'traj':  f'{SLAM_BASE}/w_static/best/CameraTrajectory.txt',
        'color': '#27AE60',
    },
    {
        'name':  'fr3/w/half',
        'gt':    f'{DATA_BASE}/fr3_walking_halfsphere/groundtruth.txt',
        'traj':  f'{SLAM_BASE}/w_half/best/CameraTrajectory.txt',
        'color': '#8E44AD',
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# EXACT FUNCTIONS FROM ORB-SLAM3 evaluate_rpe.py
# ─────────────────────────────────────────────────────────────────────────────
_EPS = numpy.finfo(float).eps * 4.0

def transform44(l):
    t = l[1:4]
    q = numpy.array(l[4:8], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.array((
            (1.0, 0.0, 0.0, t[0]),
            (0.0, 1.0, 0.0, t[1]),
            (0.0, 0.0, 1.0, t[2]),
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=numpy.float64)
    q *= numpy.sqrt(2.0 / nq)
    q  = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1,1]-q[2,2],     q[0,1]-q[2,3],     q[0,2]+q[1,3], t[0]),
        (    q[0,1]+q[2,3], 1.0-q[0,0]-q[2,2],     q[1,2]-q[0,3], t[1]),
        (    q[0,2]-q[1,3],     q[1,2]+q[0,3], 1.0-q[0,0]-q[1,1], t[2]),
        (              0.0,               0.0,               0.0,   1.0)
    ), dtype=numpy.float64)


def read_trajectory(filename, matrix=True):
    with open(filename) as f:
        data = f.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    lst   = [[float(v.strip()) for v in line.split(" ") if v.strip() != ""]
             for line in lines if len(line) > 0 and line[0] != "#"]
    lst_ok = []
    for l in lst:
        if len(l) < 8:
            continue
        if l[4:8] == [0, 0, 0, 0]:
            continue
        if any(numpy.isnan(v) for v in l):
            continue
        lst_ok.append(l)
    if matrix:
        traj = dict([(l[0], transform44(l[0:])) for l in lst_ok])
    else:
        traj = dict([(l[0], l[1:8]) for l in lst_ok])
    return traj


def find_closest_index(L, t):
    beginning  = 0
    difference = abs(L[0] - t)
    best       = 0
    end        = len(L)
    while beginning < end:
        middle = int((end + beginning) / 2)
        if abs(L[middle] - t) < difference:
            difference = abs(L[middle] - t)
            best       = middle
        if t == L[middle]:
            return middle
        elif L[middle] > t:
            end = middle
        else:
            beginning = middle + 1
    return best


def ominus(a, b):
    return numpy.dot(numpy.linalg.inv(a), b)


def compute_distance(transform):
    return numpy.linalg.norm(transform[0:3, 3])


def compute_angle(transform):
    return numpy.arccos(
        min(1, max(-1, (numpy.trace(transform[0:3, 0:3]) - 1) / 2))
    )


def evaluate_trajectory(traj_gt, traj_est,
                         param_max_pairs=10000,
                         param_fixed_delta=True,
                         param_delta=1.0,
                         param_delta_unit='f',
                         param_offset=0.0,
                         param_scale=1.0):
    """Exact copy of evaluate_trajectory from ORB-SLAM3 evaluate_rpe.py"""
    stamps_gt  = list(traj_gt.keys());  stamps_gt.sort()
    stamps_est = list(traj_est.keys()); stamps_est.sort()

    stamps_est_return = []
    for t_est in stamps_est:
        t_gt         = stamps_gt[find_closest_index(stamps_gt, t_est + param_offset)]
        t_est_return = stamps_est[find_closest_index(stamps_est, t_gt - param_offset)]
        t_gt_return  = stamps_gt[find_closest_index(stamps_gt, t_est_return + param_offset)]
        if t_est_return not in stamps_est_return:
            stamps_est_return.append(t_est_return)

    if len(stamps_est_return) < 2:
        raise Exception("Not enough overlapping timestamps.")

    if param_delta_unit == 'f':
        index_est = list(range(len(traj_est)))
    elif param_delta_unit == 's':
        index_est = list(traj_est.keys()); index_est.sort()
    else:
        index_est = list(range(len(traj_est)))

    if param_fixed_delta:
        pairs = []
        for i in range(len(stamps_est)):
            j = find_closest_index(index_est, index_est[i] + param_delta)
            if j != len(stamps_est) - 1:
                pairs.append((i, j))
        if param_max_pairs != 0 and len(pairs) > param_max_pairs:
            pairs = random.sample(pairs, param_max_pairs)
    else:
        pairs = [(random.randint(0, len(traj_est)-1),
                  random.randint(0, len(traj_est)-1))
                 for _ in range(param_max_pairs)]

    gt_interval         = numpy.median([s-t for s,t in zip(stamps_gt[1:], stamps_gt[:-1])])
    gt_max_time_diff    = 2 * gt_interval

    result = []
    for i, j in pairs:
        stamp_est_0 = stamps_est[i]
        stamp_est_1 = stamps_est[j]
        stamp_gt_0  = stamps_gt[find_closest_index(stamps_gt, stamp_est_0 + param_offset)]
        stamp_gt_1  = stamps_gt[find_closest_index(stamps_gt, stamp_est_1 + param_offset)]

        if (abs(stamp_gt_0 - (stamp_est_0 + param_offset)) > gt_max_time_diff or
            abs(stamp_gt_1 - (stamp_est_1 + param_offset)) > gt_max_time_diff):
            continue

        error44 = ominus(
            ominus(traj_est[stamp_est_1], traj_est[stamp_est_0]) * param_scale,
            ominus(traj_gt[stamp_gt_1],   traj_gt[stamp_gt_0])
        )

        trans = compute_distance(error44)
        rot   = compute_angle(error44)
        result.append([stamp_est_0, stamp_est_1, stamp_gt_0, stamp_gt_1, trans, rot])

    if len(result) < 2:
        raise Exception("Couldn't find matching timestamp pairs!")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Smooth helper
# ─────────────────────────────────────────────────────────────────────────────
def smooth(values, window=10):
    if len(values) <= window:
        return values
    kernel = numpy.ones(window) / window
    return numpy.convolve(values, kernel, mode='valid')


# ─────────────────────────────────────────────────────────────────────────────
# Multi-sequence plot — side by side 1:1 ratio
# ─────────────────────────────────────────────────────────────────────────────
def plot_multi(datasets, output):
    random.seed(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor='white')
    fig.suptitle(
        "RPE over Frame Index — Proposed System (Walking Sequences)",
        fontsize=13, fontweight='bold', y=1.01
    )

    ax_t = axes[0]
    ax_r = axes[1]

    for ds in datasets:
        if not os.path.exists(ds['traj']) or not os.path.exists(ds['gt']):
            print(f"[SKIP] {ds['name']} — file not found")
            continue

        traj_gt  = read_trajectory(ds['gt'])
        traj_est = read_trajectory(ds['traj'])

        try:
            result = evaluate_trajectory(
                traj_gt, traj_est,
                param_max_pairs=10000,
                param_fixed_delta=True,
                param_delta=1.0,
                param_delta_unit='f'
            )
        except Exception as e:
            print(f"[SKIP] {ds['name']}: {e}")
            continue

        stamps      = numpy.array(result)[:, 0]
        trans_error = numpy.array(result)[:, 4]
        rot_error   = numpy.array(result)[:, 5] * 180.0 / numpy.pi  # to degrees

        # Sort by timestamp
        order       = numpy.argsort(stamps)
        stamps      = stamps[order]
        trans_error = trans_error[order]
        rot_error   = rot_error[order]

        # Normalise time to frame index
        frames = numpy.arange(len(stamps))

        col  = ds['color']
        name = ds['name']
        w    = 12

        rmse_t = numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error))
        rmse_r = numpy.sqrt(numpy.dot(rot_error,   rot_error)   / len(rot_error))

        # Translational — smoothed line only, no raw shadow
        sm_t = smooth(trans_error, w)
        ax_t.plot(frames[w-1:], sm_t, color=col, linewidth=1.8,
                  label=f"{name}  RMSE={rmse_t:.4f} m")

        # Rotational — smoothed line only, no raw shadow
        sm_r = smooth(rot_error, w)
        ax_r.plot(frames[w-1:], sm_r, color=col, linewidth=1.8,
                  label=f"{name}  RMSE={rmse_r:.4f}°")

    # ── Format translational ──────────────────────────────────────────────
    ax_t.set_xlabel("Frame Index", fontsize=11)
    ax_t.set_ylabel("Translational RPE (m)", fontsize=11)
    ax_t.set_title("Translational RPE", fontsize=12, fontweight='bold', pad=8)
    ax_t.legend(loc='upper right', fontsize=8.5, framealpha=0.85,
                edgecolor='#cccccc')
    ax_t.grid(True, alpha=0.3, linewidth=0.5)
    ax_t.set_facecolor('#F8F9FA')
    ax_t.spines['top'].set_visible(False)
    ax_t.spines['right'].set_visible(False)
    ax_t.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    # ── Format rotational ─────────────────────────────────────────────────
    ax_r.set_xlabel("Frame Index", fontsize=11)
    ax_r.set_ylabel("Rotational RPE (°)", fontsize=11)
    ax_r.set_title("Rotational RPE", fontsize=12, fontweight='bold', pad=8)
    ax_r.legend(loc='upper right', fontsize=8.5, framealpha=0.85,
                edgecolor='#cccccc')
    ax_r.grid(True, alpha=0.3, linewidth=0.5)
    ax_r.set_facecolor('#F8F9FA')
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['right'].set_visible(False)
    ax_r.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Single sequence plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_single(gt_file, traj_file, name, output, color='#2980B9'):
    random.seed(0)

    traj_gt  = read_trajectory(gt_file)
    traj_est = read_trajectory(traj_file)

    try:
        result = evaluate_trajectory(
            traj_gt, traj_est,
            param_max_pairs=10000,
            param_fixed_delta=True,
            param_delta=1.0,
            param_delta_unit='f'
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    stamps      = numpy.array(result)[:, 0]
    trans_error = numpy.array(result)[:, 4]
    rot_error   = numpy.array(result)[:, 5] * 180.0 / numpy.pi

    order       = numpy.argsort(stamps)
    stamps      = stamps[order]
    trans_error = trans_error[order]
    rot_error   = rot_error[order]
    frames      = numpy.arange(len(stamps))

    rmse_t = numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error))
    rmse_r = numpy.sqrt(numpy.dot(rot_error,   rot_error)   / len(rot_error))
    w      = 12

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor='white')
    fig.suptitle(f"RPE over Frame Index — {name} (Proposed System)",
                 fontsize=13, fontweight='bold', y=1.01)

    # Translational
    sm_t = smooth(trans_error, w)
    axes[0].plot(frames[w-1:], sm_t, color=color, linewidth=1.8,
                 label=f"RMSE = {rmse_t:.4f} m  |  Mean = {numpy.mean(trans_error):.4f} m")
    axes[0].axhline(numpy.mean(trans_error), color=color,
                    linewidth=1.0, linestyle=':', alpha=0.5)
    axes[0].set_xlabel("Frame Index", fontsize=11)
    axes[0].set_ylabel("Translational RPE (m)", fontsize=11)
    axes[0].set_title("Translational RPE", fontsize=12,
                       fontweight='bold', pad=8)
    axes[0].legend(fontsize=9, framealpha=0.85)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor('#F8F9FA')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Rotational
    sm_r = smooth(rot_error, w)
    axes[1].plot(frames[w-1:], sm_r, color=color, linewidth=1.8,
                 label=f"RMSE = {rmse_r:.4f}°  |  Mean = {numpy.mean(rot_error):.4f}°")
    axes[1].axhline(numpy.mean(rot_error), color=color,
                    linewidth=1.0, linestyle=':', alpha=0.5)
    axes[1].set_xlabel("Frame Index", fontsize=11)
    axes[1].set_ylabel("Rotational RPE (°)", fontsize=11)
    axes[1].set_title("Rotational RPE", fontsize=12,
                       fontweight='bold', pad=8)
    axes[1].legend(fontsize=9, framealpha=0.85)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor('#F8F9FA')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Plot RPE over frame index using ORB-SLAM3 evaluation logic')
    parser.add_argument('--gt',     help='Ground truth file (single mode)')
    parser.add_argument('--traj',   help='Trajectory file (single mode)')
    parser.add_argument('--name',   default='fr3/w/xyz')
    parser.add_argument('--color',  default='#2980B9')
    parser.add_argument('--output', default='./rpe_overtime.png')
    parser.add_argument('--multi',  action='store_true',
                        help='Plot all 4 walking sequences')
    args = parser.parse_args()

    if args.multi or (not args.gt and not args.traj):
        print("Plotting all walking sequences...")
        plot_multi(DATASETS, args.output)
    else:
        if not args.gt or not args.traj:
            print("[ERROR] Provide --gt and --traj")
            return
        plot_single(args.gt, args.traj, args.name, args.output, args.color)


if __name__ == "__main__":
    main()
