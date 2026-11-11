"""
ablation_B_flow_plus_sceneflow.py
==================================
Ablation B: OPTICAL FLOW + SCENE FLOW — no YOLO
================================================
Pipeline:
    1. Preprocess frames (bilateral + CLAHE)
    2. LiteFlowNet optical flow
    3. Scene flow (3D motion from depth + flow)
    4. RANSAC ego-motion removal → residual flow
    5. Combine residual flow magnitude AND scene flow dZ
       to threshold dynamic pixels
       (same dual-signal logic as full system confirm_motion,
        but applied per-pixel, not per-instance)

NO YOLO segmentation.
NO per-instance reasoning.
Both optical flow AND scene flow signals used.

Purpose:
    Shows improvement over Ablation A (flow only) when scene flow is added.
    Shows what is STILL missing without YOLO instance reasoning.
    Middle point in the ablation table.

Usage:
    python3 ablation_B_flow_plus_sceneflow.py \
        --tum_dir  /path/to/fr3_walking_xyz \
        --sequence fr3 \
        --output_dir ./ablation_B_results

After running:
    Feed masks to ORB-SLAM3, save KeyFrameTrajectory_B.txt
    evo_ape tum groundtruth.txt KeyFrameTrajectory_B.txt -a --correct_scale
"""

import os, sys, argparse, numpy as np, cv2, struct, tempfile, subprocess
from pathlib import Path
from tqdm import tqdm

LITEFLOWNET_DIR = '/home/fyp/pytorch-liteflownet'
DEPTH_SCALE     = 5000.0

TUM_INTRINSICS = {
    "fr1": dict(fx=517.3, fy=516.5, cx=318.6, cy=255.3),
    "fr2": dict(fx=520.9, fy=521.0, cx=325.1, cy=249.7),
    "fr3": dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6),
}


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING (same as full system)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_frame(img):
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL FLOW (same as full system)
# ─────────────────────────────────────────────────────────────────────────────
def read_flo(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('f', f.read(4))[0]
        assert abs(magic - 202021.25) < 0.01
        W = struct.unpack('i', f.read(4))[0]
        H = struct.unpack('i', f.read(4))[0]
        return np.frombuffer(f.read(), dtype=np.float32).reshape(H, W, 2).copy()

def compute_optical_flow(rgb1_path, rgb2_path):
    img1 = preprocess_frame(cv2.imread(str(rgb1_path)))
    img2 = preprocess_frame(cv2.imread(str(rgb2_path)))
    tmp1 = tempfile.mktemp(suffix='.png')
    tmp2 = tempfile.mktemp(suffix='.png')
    flo  = tempfile.mktemp(suffix='.flo')
    cv2.imwrite(tmp1, img1); cv2.imwrite(tmp2, img2)
    subprocess.run(
        ['python', 'run.py', '--model', 'default',
         '--one', tmp1, '--two', tmp2, '--out', flo],
        cwd=LITEFLOWNET_DIR, check=True, capture_output=True)
    os.remove(tmp1); os.remove(tmp2)
    flow = read_flo(flo); os.remove(flo)
    return flow


# ─────────────────────────────────────────────────────────────────────────────
# SCENE FLOW (same as full system)
# ─────────────────────────────────────────────────────────────────────────────
def depth_to_pointcloud(d, fx, fy, cx, cy):
    H, W = d.shape
    uu,vv = np.meshgrid(np.arange(W,dtype=np.float32),
                         np.arange(H,dtype=np.float32))
    Z = d.astype(np.float32)/DEPTH_SCALE
    return np.stack([(uu-cx)*Z/fx, (vv-cy)*Z/fy, Z], axis=-1)

def warp_depth(d_next, flow):
    H, W = d_next.shape
    uu,vv = np.meshgrid(np.arange(W,dtype=np.float32),
                         np.arange(H,dtype=np.float32))
    return cv2.remap(
        d_next.astype(np.float32),
        (uu+flow[...,0]).astype(np.float32),
        (vv+flow[...,1]).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def compute_scene_flow(d_t, d_t1, flow, fx, fy, cx, cy):
    P_t  = depth_to_pointcloud(d_t, fx, fy, cx, cy)
    dw   = warp_depth(d_t1, flow)
    P_t1 = depth_to_pointcloud(
        np.clip(dw,0,65535).astype(np.uint16), fx, fy, cx, cy)
    sf   = P_t1 - P_t
    vm   = (d_t > 0) & (dw > 0)
    sf[~vm] = 0
    return sf, vm


# ─────────────────────────────────────────────────────────────────────────────
# EGO-MOTION REMOVAL (same as full system)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_ego(flow, vm):
    H, W = vm.shape
    ys, xs = np.where(vm)
    if len(xs) < 100:
        bu = np.median(flow[vm,0]) if vm.any() else 0.0
        bv = np.median(flow[vm,1]) if vm.any() else 0.0
        r  = flow.copy(); r[...,0]-=bu; r[...,1]-=bv; return r
    n   = min(4000, len(xs))
    idx = np.random.choice(len(xs), n, replace=False)
    xs_s, ys_s = xs[idx], ys[idx]
    u_s = flow[ys_s,xs_s,0]; v_s = flow[ys_s,xs_s,1]
    src = np.stack([xs_s,ys_s],1).astype(np.float32)
    dst = np.stack([xs_s+u_s,ys_s+v_s],1).astype(np.float32)
    A, inl = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=1.0, maxIters=2000, confidence=0.99)
    if A is None:
        bu = np.median(flow[vm,0]); bv = np.median(flow[vm,1])
        r  = flow.copy(); r[...,0]-=bu; r[...,1]-=bv; return r
    uu,vv = np.meshgrid(np.arange(W,dtype=np.float32),
                         np.arange(H,dtype=np.float32))
    pu = A[0,0]*uu+A[0,1]*vv+A[0,2]-uu
    pv = A[1,0]*uu+A[1,1]*vv+A[1,2]-vv
    r  = np.empty_like(flow)
    r[...,0]=flow[...,0]-pu; r[...,1]=flow[...,1]-pv
    return r


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION B: DYNAMIC MASK — OPTICAL FLOW + SCENE FLOW, NO YOLO
#
# Uses same dual-signal logic as full system confirm_motion():
#   - Strong flow (>20% pixels above threshold) → dynamic
#   - Weak flow (>0.5%) AND high dZ (>10%) → dynamic
# But applied per-pixel over the whole image, not per-instance.
# ─────────────────────────────────────────────────────────────────────────────
def make_dynamic_mask_flow_sceneflow(flow, scene_flow, valid_mask,
                                      flow_thresh=1.5,
                                      dz_thresh=0.05,
                                      min_blob_area=500):
    """
    Ablation B: optical flow + scene flow, no YOLO.

    Per-pixel dual-signal logic (mirrors confirm_motion thresholds):
      - residual flow > flow_thresh px  → definitely dynamic
      - residual flow > 0.5px AND dZ > dz_thresh m → dynamic (borderline)
      - otherwise → static

    Then same morphological cleanup as Ablation A.

    Returns: H×W uint8 mask — 255=dynamic, 0=static
    """
    H, W = valid_mask.shape

    # Ego-motion removal
    residual = estimate_ego(flow, valid_mask)
    res_mag  = np.sqrt(residual[...,0]**2 + residual[...,1]**2)
    res_mag[~valid_mask] = 0

    # Scene flow dZ — smoothed to reduce depth sensor noise
    # (same Gaussian blur as confirm_motion in full system)
    dZ_raw  = np.abs(scene_flow[...,2]).astype(np.float32)
    dZ_s    = cv2.GaussianBlur(dZ_raw, (5,5), sigmaX=2)
    dZ_s[~valid_mask] = 0

    # Dual-signal dynamic detection (per-pixel version of confirm_motion)
    # Signal 1: strong flow alone → dynamic
    strong_flow = res_mag > flow_thresh

    # Signal 2: weak flow + dZ confirmation → dynamic
    weak_flow_with_dz = (res_mag > 0.5) & (dZ_s > dz_thresh)

    # Combine
    dynamic_pixels = (strong_flow | weak_flow_with_dz).astype(np.uint8) * 255
    dynamic_pixels[~valid_mask] = 0

    # Remove tiny noise blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dynamic_pixels, connectivity=8)
    cleaned = np.zeros_like(dynamic_pixels)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_blob_area:
            cleaned[labels == label_id] = 255

    # Morphological closing — fill holes in dynamic regions
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    # Dilation — same kernel as full system (fair comparison)
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    final_mask = cv2.dilate(closed, kernel_dil, iterations=2)

    return final_mask


# ─────────────────────────────────────────────────────────────────────────────
# TUM LOADER + MASK FILL
# ─────────────────────────────────────────────────────────────────────────────
def load_tum_pairs(tum_dir, max_pairs=9999):
    tum_dir = Path(tum_dir)
    def parse(txt):
        out=[]
        for line in open(txt):
            line=line.strip()
            if not line or line.startswith('#'): continue
            out.append(str(tum_dir/line.split()[1]))
        return out
    rt,dt = tum_dir/"rgb.txt", tum_dir/"depth.txt"
    if rt.exists() and dt.exists():
        rgbs,deps = parse(rt),parse(dt)
    else:
        rgbs=[str(p) for p in sorted((tum_dir/"rgb").glob("*.png"))]
        deps=[str(p) for p in sorted((tum_dir/"depth").glob("*.png"))]
    n=min(len(rgbs),len(deps),max_pairs+1)
    return [(rgbs[i],rgbs[i+1],deps[i],deps[i+1]) for i in range(n-1)]


def _fill_missing_masks(masks_dir, rgb_txt):
    """
    Guarantee every RGB frame timestamp has a mask file.
    Uses nearest-neighbour fill from existing masks.
    """
    import shutil
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not Path(rgb_txt).exists():
        print(f"  [fill] WARNING: rgb.txt not found at {rgb_txt}")
        return

    # Parse all RGB timestamps in order
    rgb_ts = []
    with open(rgb_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rgb_ts.append(line.split()[0])

    if not rgb_ts:
        print("  [fill] WARNING: no timestamps found in rgb.txt")
        return

    existing = set(f.stem for f in masks_dir.glob("*.png"))
    missing  = [ts for ts in rgb_ts if ts not in existing]

    if not missing:
        print(f"  [fill] All {len(rgb_ts)} masks present ✓")
        return

    print(f"  [fill] Filling {len(missing)} missing masks "
          f"from {len(existing)} existing...")

    filled = 0
    for ts in missing:
        idx = rgb_ts.index(ts)
        # Search outward from current position
        for offset in range(1, len(rgb_ts)):
            found = False
            for candidate_idx in [max(0, idx-offset),
                                   min(len(rgb_ts)-1, idx+offset)]:
                candidate = rgb_ts[candidate_idx]
                if candidate in existing:
                    src = masks_dir / f"{candidate}.png"
                    dst = masks_dir / f"{ts}.png"
                    if src.exists():
                        shutil.copy(str(src), str(dst))
                        existing.add(ts)
                        filled += 1
                        found = True
                        break
            if found:
                break

    print(f"  [fill] Filled {filled} masks. "
          f"Total: {len(list(masks_dir.glob('*.png')))}/{len(rgb_ts)}")


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ONE PAIR
# ─────────────────────────────────────────────────────────────────────────────
def process_pair(r1, r2, d1, d2, intr, out_dir,
                  flow_thresh, dz_thresh, idx=0):
    rgb1 = cv2.imread(str(r1))
    dt   = cv2.imread(str(d1), cv2.IMREAD_UNCHANGED)
    dt1  = cv2.imread(str(d2), cv2.IMREAD_UNCHANGED)
    if rgb1 is None or dt is None or dt1 is None: return

    H, W = rgb1.shape[:2]
    if dt.shape[:2]  != (H,W):
        dt  = cv2.resize(dt,  (W,H), interpolation=cv2.INTER_NEAREST)
    if dt1.shape[:2] != (H,W):
        dt1 = cv2.resize(dt1, (W,H), interpolation=cv2.INTER_NEAREST)

    fx,fy,cx,cy = intr['fx'],intr['fy'],intr['cx'],intr['cy']

    # Optical flow
    flow = compute_optical_flow(str(r1), str(r2))

    # Scene flow — ENABLED in Ablation B
    sf, vm = compute_scene_flow(dt, dt1, flow, fx, fy, cx, cy)

    # Dynamic mask — flow + scene flow, no YOLO
    dm = make_dynamic_mask_flow_sceneflow(
        flow, sf, vm,
        flow_thresh=flow_thresh,
        dz_thresh=dz_thresh)

    # Save masks named by timestamp
    masks_dir = Path(out_dir) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    ts1 = Path(r1).stem
    ts2 = Path(r2).stem
    cv2.imwrite(str(masks_dir/f"{ts1}.png"), dm)
    cv2.imwrite(str(masks_dir/f"{ts2}.png"), dm)

    dyn_pct = dm.sum()//255 / (H*W) * 100
    print(f"  [B] pair {idx:04d} | dynamic: {dyn_pct:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Ablation B: Optical Flow + Scene Flow (no YOLO)")
    p.add_argument("--tum_dir",     required=True)
    p.add_argument("--sequence",    default="fr3",
                   choices=["fr1","fr2","fr3"])
    p.add_argument("--output_dir",  default="./ablation_B_results")
    p.add_argument("--max_pairs",   type=int,   default=9999)
    p.add_argument("--flow_thresh", type=float, default=1.5,
                   help="Residual flow threshold px (default: 1.5)")
    p.add_argument("--dz_thresh",   type=float, default=0.05,
                   help="Scene flow dZ threshold metres (default: 0.05)")
    args = p.parse_args()

    intr  = TUM_INTRINSICS[args.sequence]
    pairs = load_tum_pairs(args.tum_dir, args.max_pairs)
    if not pairs:
        print("[ERROR] No pairs found."); sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  ABLATION B — Optical Flow + Scene Flow")
    print(f"  YOLO:         DISABLED")
    print(f"  Scene flow:   ENABLED")
    print(f"  Flow thresh:  {args.flow_thresh}px")
    print(f"  dZ thresh:    {args.dz_thresh}m")
    print(f"  Pairs:        {len(pairs)}")
    print(f"  Output:       {args.output_dir}")
    print(f"{'='*55}\n")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for i, (r1, r2, d1, d2) in enumerate(
            tqdm(pairs, desc="Ablation B — Flow + SceneFlow")):
        process_pair(r1, r2, d1, d2, intr, args.output_dir,
                     args.flow_thresh, args.dz_thresh, i)

    # Final fill — guarantee every RGB frame has a mask
    rgb_txt   = str(Path(args.tum_dir) / "rgb.txt")
    masks_dir = Path(args.output_dir) / "masks"
    _fill_missing_masks(masks_dir, rgb_txt)
    total = len(list(masks_dir.glob("*.png")))
    rgb_total = sum(1 for l in open(rgb_txt)
                    if l.strip() and not l.startswith('#'))
    if total < rgb_total:
        print(f"[WARNING] Only {total}/{rgb_total} masks generated.")
        print(f"  Some frames may still be missing — check rgb.txt path.")
    else:
        print(f"[✓] All {total} masks present — ready for ORB-SLAM3")

    print(f"\n[✓] Ablation B complete — {total} masks → {args.output_dir}/masks/")
    print(f"[→] Run ORB-SLAM3 with these masks")
    print(f"[→] Save trajectory as KeyFrameTrajectory_B.txt")
    print(f"[→] evo_ape tum groundtruth.txt KeyFrameTrajectory_B.txt "
          f"--align --correct_scale -p")

if __name__ == "__main__":
    main()
