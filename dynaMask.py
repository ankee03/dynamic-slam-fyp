"""
Scene Flow + YOLOv8 Dynamic Mask Pipeline for TUM RGB-D / ORB-SLAM3
====================================================================

Pipeline:
    1. Preprocess frames (bilateral filter + CLAHE)
    2. LiteFlowNet optical flow
    3. Compute scene flow (3D motion per pixel)
    4. YOLOv8-seg → categorise into RIGID / NON-RIGID instances
    5. RANSAC affine ego-motion removal → residual flow
    6. Per-instance motion confirmation (residual flow + scene flow dZ)
    7. Final dynamic mask = confirmed dynamic non-rigid instances
    8. Visualisation: instance seg + dynamic/static labels + final mask

Visualisation rows:
    Row 1: Original Frame t | Frame t+1
    Row 2: Optical Flow t   | Optical Flow t+1
    Row 3: YOLOv8 instances (coloured, rigid=blue, non-rigid=orange,
                              dynamic=red, static non-rigid=green)
         | Scene Flow
    Row 4: Final Dynamic Mask (white=dynamic) | Overlay on RGB
"""

import os, sys, argparse, numpy as np, cv2, struct, tempfile, subprocess
import matplotlib; matplotlib.use('Agg')  # non-interactive — no popups
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
import random

LITEFLOWNET_DIR = '/home/fyp/pytorch-liteflownet'
DEPTH_SCALE     = 5000.0

TUM_INTRINSICS = {
    "fr1": dict(fx=517.3, fy=516.5, cx=318.6, cy=255.3),
    "fr2": dict(fx=520.9, fy=521.0, cx=325.1, cy=249.7),
    "fr3": dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6),
}

# ── Class categorisation ──────────────────────────────────────────────────────
NON_RIGID_CLASSES = {
    # Living things
    'person', 'cat', 'dog', 'bird', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe',
    # Vehicles
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat',
    'airplane', 'train',
    # Carried / movable objects
    'backpack', 'handbag', 'suitcase', 'umbrella',
    # Sports / throwable
    'sports ball', 'frisbee', 'skateboard', 'surfboard', 'skis',
    # Movable furniture / objects (can be pushed, carried or moved by human)
    'chair', 'bottle', 'cup', 'bowl', 'book',
    'keyboard', 'mouse', 'remote', 'cell phone',
    'scissors', 'vase', 'clock',
}

RIGID_CLASSES = {
    # Fixed infrastructure — cannot be moved without tools
    'dining table', 'bed', 'toilet', 'sink', 'refrigerator',
    'oven', 'microwave', 'couch', 'bench', 'potted plant',
    'fire hydrant', 'stop sign', 'parking meter',
    # Screens / monitors — fixed to desk/wall
    'tv', 'laptop', 'monitor', 'computer',
}

# Colours for visualisation (BGR)
COL_RIGID        = (180,  80,  80)   # blue-ish  — rigid, skip
COL_NONRIGID     = (  0, 165, 255)   # orange    — non-rigid, unconfirmed
COL_DYNAMIC      = (  0,   0, 255)   # red       — confirmed dynamic
COL_STATIC_NR    = (  0, 200,   0)   # green     — non-rigid but static

# ── YOLOv8 singleton ──────────────────────────────────────────────────────────
_yolo_model = None
def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO('yolov8n-seg.pt')
        print("[✓] YOLOv8n-seg loaded")
    return _yolo_model


# ─────────────────────────────────────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_frame(img):
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# 2. OPTICAL FLOW
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
    subprocess.run(['python','run.py','--model','default',
                    '--one',tmp1,'--two',tmp2,'--out',flo],
                   cwd=LITEFLOWNET_DIR, check=True)
    os.remove(tmp1); os.remove(tmp2)
    flow = read_flo(flo); os.remove(flo)
    u, v = flow[...,0], flow[...,1]
    print(f"  [flow] u:{u.min():.2f}/{u.max():.2f}  v:{v.min():.2f}/{v.max():.2f}")
    return flow, flow_to_color_hsv(flow)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCENE FLOW
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
    return cv2.remap(d_next.astype(np.float32),
                     (uu+flow[...,0]).astype(np.float32),
                     (vv+flow[...,1]).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def compute_scene_flow(d_t, d_t1, flow, fx, fy, cx, cy):
    P_t  = depth_to_pointcloud(d_t, fx, fy, cx, cy)
    dw   = warp_depth(d_t1, flow)
    P_t1 = depth_to_pointcloud(np.clip(dw,0,65535).astype(np.uint16), fx, fy, cx, cy)
    sf   = P_t1 - P_t
    vm   = (d_t > 0) & (dw > 0)
    sf[~vm] = 0
    return sf, vm


# ─────────────────────────────────────────────────────────────────────────────
# 4. EGO-MOTION REMOVAL
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
    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                   ransacReprojThreshold=1.0,
                                   maxIters=2000, confidence=0.99)
    if A is None:
        bu = np.median(flow[vm,0]); bv = np.median(flow[vm,1])
        r  = flow.copy(); r[...,0]-=bu; r[...,1]-=bv; return r
    print(f"  [ego] RANSAC inlier={inl.sum()/len(inl):.1%}")
    uu,vv = np.meshgrid(np.arange(W,dtype=np.float32),
                         np.arange(H,dtype=np.float32))
    pu = A[0,0]*uu+A[0,1]*vv+A[0,2]-uu
    pv = A[1,0]*uu+A[1,1]*vv+A[1,2]-vv
    r  = np.empty_like(flow)
    r[...,0]=flow[...,0]-pu; r[...,1]=flow[...,1]-pv
    return r


# ─────────────────────────────────────────────────────────────────────────────
# 5. YOLOV8 DETECTION → instances with rigid/non-rigid category
# ─────────────────────────────────────────────────────────────────────────────
def detect_instances(rgb, conf_thresh=0.25):
    """
    Run YOLOv8-seg. Returns list of dicts:
    {
      'class':    str,
      'conf':     float,
      'category': 'non-rigid' | 'rigid' | 'unknown',
      'mask':     H×W uint8 binary,
      'color':    (B,G,R) unique instance colour
    }
    """
    model   = get_yolo()
    H, W    = rgb.shape[:2]
    results = model(rgb, conf=conf_thresh, verbose=False)[0]

    instances = []
    if results.masks is None:
        print("  [yolo] no masks detected")
        return instances

    # Generate unique colours for each instance
    np.random.seed(42)
    colors = [(int(c[0]),int(c[1]),int(c[2]))
              for c in np.random.randint(50,230,(len(results.boxes),3))]

    for i, (cls_id, conf) in enumerate(zip(results.boxes.cls, results.boxes.conf)):
        cls_name = model.names[int(cls_id)]

        if cls_name in NON_RIGID_CLASSES:
            category = 'non-rigid'
        elif cls_name in RIGID_CLASSES:
            category = 'rigid'
        else:
            category = 'non-rigid'  # unknown → treat as potentially dynamic

        mask_data   = results.masks.data[i].cpu().numpy()
        mask_resized= cv2.resize(mask_data,(W,H),interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_resized > 0.5).astype(np.uint8)*255

        bbox = results.boxes.xyxy[i].cpu().numpy()  # x1,y1,x2,y2
        instances.append({
            'class':    cls_name,
            'conf':     float(conf),
            'category': category,
            'mask':     binary_mask,
            'color':    colors[i],
            'bbox_y':   int(bbox[1]),
            'bbox_h':   int(bbox[3] - bbox[1]),
        })
        print(f"  [yolo] {cls_name}({conf:.2f}) [{category}]  "
              f"px={binary_mask.sum()//255}")

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# 6. MOTION CONFIRMATION PER INSTANCE
# ─────────────────────────────────────────────────────────────────────────────
def confirm_motion(obj_mask, res_mag, scene_flow, valid_mask,
                   flow_thresh=1.5, dz_thresh=0.05):
    """
    Returns (is_dynamic, flow_ratio, dz_ratio).

    Motion detection strategy:
      - PRIMARY: residual optical flow (reliable, ego-motion removed)
      - SECONDARY: scene flow dZ only used to boost borderline cases
        (dZ alone is too noisy from depth sensor — keyboard/mouse show
         high dZ even when stationary)

    Logic:
      - flow > 20%  → definitely dynamic (strong motion)
      - flow 5-20% AND dZ > 10% → dynamic (weak motion confirmed by depth)
      - flow < 5%   → static (camera noise level)
    """
    mask_bool = (obj_mask > 0) & valid_mask
    if mask_bool.sum() < 10:
        return False, 0.0, 0.0

    # Signal 1: residual optical flow (primary signal)
    flow_ratio = float(((res_mag > flow_thresh) & mask_bool).sum() / mask_bool.sum())

    # Signal 2: scene flow dZ (secondary — only used to confirm borderline)
    dZ_s     = cv2.GaussianBlur(np.abs(scene_flow[...,2]).astype(np.float32),(5,5),sigmaX=2)
    dz_ratio = float(((dZ_s > dz_thresh) & mask_bool).sum() / mask_bool.sum())

    # Strong flow → definitely dynamic
    if flow_ratio > 0.20:
        is_dynamic = True
    # Weak flow + dZ confirmation → dynamic (catches slow/sitting movers)
    elif flow_ratio > 0.005 and dz_ratio > 0.05:
        is_dynamic = True
    # Zero flow → static regardless of dZ (depth sensor noise)
    else:
        is_dynamic = False

    return is_dynamic, flow_ratio, dz_ratio


# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD INSTANCE VISUALISATION IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def draw_instance_vis(rgb, instances, alpha=0.5):
    """
    Draws coloured instance masks with dynamic/static labels (like reference image).
    Red   = DYNAMIC
    Green = STATIC non-rigid
    Blue  = RIGID (skipped)
    """
    H, W  = rgb.shape[:2]
    overlay = np.zeros_like(rgb, dtype=np.float32)
    vis     = rgb.copy().astype(np.float32)

    for inst in instances:
        mask = inst['mask'] > 0
        if not mask.any():
            continue

        # Colour by status
        if inst['category'] == 'rigid':
            col = COL_RIGID
        elif inst.get('is_dynamic') is True:
            col = COL_DYNAMIC
        else:
            col = COL_STATIC_NR

        overlay[mask] = col

    # Blend
    result = cv2.addWeighted(vis.astype(np.uint8), 1-alpha,
                              overlay.astype(np.uint8), alpha, 0)

    # Draw labels on top
    for inst in instances:
        mask = inst['mask'] > 0
        if not mask.any():
            continue

        if inst['category'] == 'rigid':
            col    = COL_RIGID
            status = 'rigid'
        elif inst.get('is_dynamic') is True:
            col    = COL_DYNAMIC
            status = 'dynamic'
        else:
            col    = COL_STATIC_NR
            status = 'static'

        # Position label at top of mask
        ys, xs = np.where(mask)
        x = int(xs.mean())
        y = max(int(ys.min()) - 5, 15)

        # Background box for readability
        label = f"{inst['class']} [{status}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(result, (x-2, y-th-2), (x+tw+2, y+2), (0,0,0), -1)
        cv2.putText(result, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    return result


def draw_debug_panels(rgb, instances):
    """
    Generates 3 debug images saved separately (not shown in popup):
      1. All instances with unique colours + confidence
      2. Non-rigid (red) / Rigid (blue) categorisation
      3. Per-instance dynamic (red) / static (green) with flow/dz stats
    """
    H, W = rgb.shape[:2]

    # ── Panel 1: All instances unique colours ─────────────────────────────
    p1 = rgb.copy().astype(np.float32)
    ov1 = np.zeros_like(rgb, dtype=np.float32)
    for inst in instances:
        mask = inst['mask'] > 0
        if not mask.any(): continue
        col = inst['color']
        ov1[mask] = col
    p1 = cv2.addWeighted(p1.astype(np.uint8), 0.5, ov1.astype(np.uint8), 0.5, 0)
    for inst in instances:
        mask = inst['mask'] > 0
        if not mask.any(): continue
        ys,xs = np.where(mask)
        x,y = int(xs.mean()), max(int(ys.min())-5, 15)
        label = f"{inst['class']} {inst['conf']:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(p1,(x-2,y-th-2),(x+tw+2,y+2),(0,0,0),-1)
        cv2.putText(p1, label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,
                    tuple(int(c) for c in inst['color']),1,cv2.LINE_AA)
    cv2.putText(p1,"YOLOv8-seg  (all instances)",(5,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)

    # ── Panel 2: Non-rigid red / Rigid blue ───────────────────────────────
    p2  = rgb.copy().astype(np.float32)
    ov2 = np.zeros_like(rgb, dtype=np.float32)
    for inst in instances:
        mask = inst['mask'] > 0
        if not mask.any(): continue
        col = (180,80,80) if inst['category']=='rigid' else (0,80,220)
        ov2[mask] = col
    p2 = cv2.addWeighted(p2.astype(np.uint8),0.45,ov2.astype(np.uint8),0.55,0)
    for inst in instances:
        mask = inst['mask'] > 0
        if not mask.any(): continue
        ys,xs = np.where(mask)
        x,y = int(xs.mean()), max(int(ys.min())-5,15)
        tag = '[R]' if inst['category']=='rigid' else '[NR]'
        col = (180,80,80) if inst['category']=='rigid' else (0,80,220)
        label = f"{tag} {inst['class']}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.4,1)
        cv2.rectangle(p2,(x-2,y-th-2),(x+tw+2,y+2),(0,0,0),-1)
        cv2.putText(p2,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1,cv2.LINE_AA)
    cv2.putText(p2,"Non-rigid (blue) / Rigid (red)",(5,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)

    # ── Panel 3: Dynamic red / Static green with stats ────────────────────
    p3  = rgb.copy().astype(np.float32)
    ov3 = np.zeros_like(rgb, dtype=np.float32)
    for inst in instances:
        mask = inst['mask'] > 0
        if not mask.any(): continue
        if inst['category']=='rigid': col=(180,80,80)
        elif inst.get('is_dynamic'):  col=(0,0,220)
        else:                          col=(0,180,0)
        ov3[mask]=col
    p3 = cv2.addWeighted(p3.astype(np.uint8),0.45,ov3.astype(np.uint8),0.55,0)
    for inst in instances:
        mask = inst['mask'] > 0
        if not mask.any(): continue
        ys,xs = np.where(mask)
        x,y = int(xs.mean()), max(int(ys.min())-5,15)
        if inst['category']=='rigid':
            label = f"{inst['class']} [rigid]"
            col=(180,80,80)
        elif inst.get('is_dynamic'):
            fr  = inst.get('flow_ratio',0)
            dzr = inst.get('dz_ratio',0)
            label = f"{inst['class']} [DYN] f={fr:.0%} dz={dzr:.0%}"
            col=(0,0,220)
        else:
            label = f"{inst['class']} [static]"
            col=(0,180,0)
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.35,1)
        cv2.rectangle(p3,(x-2,y-th-2),(x+tw+2,y+2),(0,0,0),-1)
        cv2.putText(p3,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1,cv2.LINE_AA)
    cv2.putText(p3,"Per-instance: red=DYN, green=static",(5,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)

    return p1, p2, p3


# ─────────────────────────────────────────────────────────────────────────────
# 8. DYNAMIC MASK PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def make_dynamic_mask(scene_flow, valid_mask, flow, depth_t, rgb,
                       debug_dir=None, pair_idx=0):
    """
    Full pipeline returning:
      final_mask  : H×W uint8 — white = dynamic pixels
      instances   : list of instance dicts with is_dynamic set
      inst_vis    : H×W×3 BGR — coloured instance visualisation
    """
    H, W = valid_mask.shape

    # Step 1: YOLOv8 detection
    instances = detect_instances(rgb)

    # Step 2: Ego-motion removal
    residual = estimate_ego(flow, valid_mask)
    res_mag  = np.sqrt(residual[...,0]**2 + residual[...,1]**2)
    res_mag[~valid_mask] = 0
    if valid_mask.any():
        print(f"  [mask] residual p50={np.percentile(res_mag[valid_mask],50):.3f} "
              f"p90={np.percentile(res_mag[valid_mask],90):.3f} px")

    # Step 3: Per-instance motion confirmation (non-rigid only)
    final_mask = np.zeros((H,W), dtype=np.uint8)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

    # Objects that should NEVER be flagged dynamic via motion alone
    # They only move when physically carried — but flow/dZ cant distinguish
    # camera-induced apparent motion from real motion for small objects
    NEVER_MOTION_DYNAMIC = {
        'keyboard', 'mouse', 'bottle', 'cup', 'bowl',
        'remote', 'cell phone', 'book', 'scissors',
        'clock', 'vase',
    }

    # Pass 1: confirm dynamic PERSONS first
    person_dynamic_mask = np.zeros((H,W), dtype=np.uint8)
    for inst in instances:
        if inst['category'] == 'rigid':
            inst['is_dynamic'] = False
            inst['flow_ratio'] = 0.0
            inst['dz_ratio']   = 0.0
            print(f"  [mask] {inst['class']} [RIGID] → skip")
            continue

        # Skip motion check for small objects — too noisy
        if inst['class'] in NEVER_MOTION_DYNAMIC:
            inst['is_dynamic'] = False
            inst['flow_ratio'] = 0.0
            inst['dz_ratio']   = 0.0
            print(f"  [mask] {inst['class']} → deferred (small object, skip motion check)")
            continue

        is_dyn, fr, dzr = confirm_motion(
            inst['mask'], res_mag, scene_flow, valid_mask,
            flow_thresh=1.5, dz_thresh=0.05)

        inst['is_dynamic'] = is_dyn
        inst['flow_ratio'] = fr
        inst['dz_ratio']   = dzr

        status = "DYNAMIC ✓" if is_dyn else "static ✗"
        print(f"  [mask] {inst['class']} [non-rigid] → {status}  "
              f"flow={fr:.1%} dz={dzr:.1%}")

        if is_dyn:
            dilated = cv2.dilate(inst['mask'], k_dil, iterations=2)
            final_mask = cv2.bitwise_or(final_mask, dilated)
            if inst['class'] == 'person':
                # Tight contact zone — only immediate surroundings
                person_dynamic_mask = cv2.bitwise_or(
                    person_dynamic_mask,
                    cv2.dilate(inst['mask'], k_dil, iterations=4)
                )

    # Build all-person mask (dynamic or not) for chair sitting check
    all_person_mask = np.zeros((H,W), dtype=np.uint8)
    for inst in instances:
        if inst['class'] == 'person':
            all_person_mask = cv2.bitwise_or(all_person_mask, inst['mask'])

    # Pass 2: non-person objects
    # Chairs: dynamic if person is sitting on them (vertical overlap in lower half)
    # Other objects: dynamic only if directly touching dynamic person
    DIRECT_CONTACT_CLASSES = {'backpack', 'handbag', 'suitcase', 'umbrella',
                               'sports ball', 'frisbee', 'skateboard'}
    PROXIMITY_EXCLUDE = {'keyboard', 'mouse', 'bottle', 'cup', 'bowl',
                         'remote', 'cell phone'}

    for inst in instances:
        if inst['class'] == 'person' or inst['category'] == 'rigid':
            continue
        if inst.get('is_dynamic'):
            continue

        if inst['class'] == 'chair':
            # A person is sitting on a chair if:
            # 1. Their masks overlap significantly
            # 2. The overlap is in the UPPER half of the chair
            #    (standing person would only overlap at chair side, not top)
            chair_mask = inst['mask']
            H_c, W_c   = chair_mask.shape

            # Upper 60% of chair bounding box = seat area
            y_top  = inst.get('bbox_y', 0)
            y_bot  = y_top + inst.get('bbox_h', H_c)
            seat_zone = np.zeros_like(chair_mask)
            seat_top  = y_top
            seat_bot  = y_top + int((y_bot - y_top) * 0.65)
            seat_zone[seat_top:seat_bot, :] = chair_mask[seat_top:seat_bot, :]

            seat_person_overlap = cv2.bitwise_and(seat_zone, all_person_mask)
            seat_dynamic_overlap = cv2.bitwise_and(seat_zone, person_dynamic_mask)

            if seat_person_overlap.sum() > 300 and seat_dynamic_overlap.sum() > 150:
                inst['is_dynamic'] = True
                print(f"  [mask] chair → DYNAMIC (person sitting on seat area)")
                final_mask = cv2.bitwise_or(final_mask,
                                            cv2.dilate(inst['mask'], k_dil, iterations=2))
            else:
                print(f"  [mask] chair → static (no sitting overlap in seat zone)")

        elif inst['class'] in PROXIMITY_EXCLUDE:
            # These need direct physical contact — skip proximity check
            print(f"  [mask] {inst['class']} → static (proximity excluded)")

        else:
            # Other objects: dynamic if touching contact zone
            overlap = cv2.bitwise_and(inst['mask'], person_dynamic_mask)
            if overlap.sum() > 0:
                inst['is_dynamic'] = True
                print(f"  [mask] {inst['class']} → DYNAMIC (contact with person)")
                final_mask = cv2.bitwise_or(final_mask,
                                             cv2.dilate(inst['mask'], k_dil, iterations=2))
            else:
                print(f"  [mask] {inst['class']} → static (not near person)")

    # Step 4: Build instance visualisation
    inst_vis = draw_instance_vis(rgb, instances)

    # Debug panels — save separately, not shown in popup
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        p = f"{debug_dir}/pair{pair_idx:04d}"
        if valid_mask.any():
            rn = np.clip(res_mag/(np.percentile(res_mag[valid_mask],99)+1e-6),0,1)
        else:
            rn = res_mag
        cv2.imwrite(f"{p}_residual.png",
                    cv2.applyColorMap((rn*255).astype(np.uint8),cv2.COLORMAP_JET))
        cv2.imwrite(f"{p}_inst_vis.png",  inst_vis)
        cv2.imwrite(f"{p}_dyn_mask.png",  final_mask)

        # Save the 3 debug panels
        p1, p2, p3 = draw_debug_panels(rgb, instances)
        cv2.imwrite(f"{p}_debug_all_instances.png",   p1)
        cv2.imwrite(f"{p}_debug_rigid_nonrigid.png",  p2)
        cv2.imwrite(f"{p}_debug_dyn_static.png",      p3)
        print(f"  [debug] saved 3 panel images to {debug_dir}")

    print(f"  [mask] final dynamic pixels={final_mask.sum()//255} "
          f"({final_mask.sum()//255/(H*W)*100:.1f}%)")

    return final_mask, instances, inst_vis


# ─────────────────────────────────────────────────────────────────────────────
# 9. COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def flow_to_color_hsv(flow):
    u,v  = flow[...,0].astype(np.float32), flow[...,1].astype(np.float32)
    ang  = np.arctan2(v,u)+np.pi
    mag  = np.sqrt(u**2+v**2)
    hue  = (ang/(2*np.pi)*179).astype(np.uint8)
    sat  = np.clip(mag/(np.percentile(mag,99)+1e-6)*255,0,255).astype(np.uint8)
    return cv2.cvtColor(np.stack([hue,sat,np.full_like(sat,255)],-1),
                        cv2.COLOR_HSV2BGR)

def visualize_scene_flow(sf, vm, flow=None):
    f   = flow if flow is not None else sf[...,:2]
    u,v = f[...,0].astype(np.float32), f[...,1].astype(np.float32)
    ang = np.arctan2(v,u)+np.pi
    mag = np.sqrt(u**2+v**2)
    hue = (ang/(2*np.pi)*179).astype(np.uint8)
    sat = np.clip(mag/(np.percentile(mag,99)+1e-6)*255,0,255).astype(np.uint8)
    fb  = cv2.cvtColor(np.stack([hue,sat,np.full_like(sat,255)],-1),
                        cv2.COLOR_HSV2BGR).astype(np.float32)
    bu  = np.median(u[vm]) if vm.any() else 0.0
    bv  = np.median(v[vm]) if vm.any() else 0.0
    rm  = np.sqrt((u-bu)**2+(v-bv)**2)
    a   = np.power(np.clip(rm/(np.percentile(rm,98)+1e-6),0,1),0.25)[...,np.newaxis]
    vis = (a*fb+(1-a)*255).clip(0,255).astype(np.uint8)
    vis[~vm] = 255
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# 10. DISPLAY + SAVE  (4-row layout)
# ─────────────────────────────────────────────────────────────────────────────
def display_and_save(rgb1, rgb2, fv, sv, dm, inst_vis, instances,
                      out_path, pair_idx=0):
    """
    3-row × 2-col grid:
      Row 1: Frame t            | Frame t+1
      Row 2: Optical Flow       | YOLOv8 Instance Seg
      Row 3: Scene Flow         | Final Dynamic Mask
    """
    fig = plt.figure(figsize=(20, 14), facecolor='#0d0d0d')
    fig.suptitle(f"Scene Flow + YOLOv8 — TUM RGB-D  |  pair {pair_idx:04d}",
                 color='white', fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.05, hspace=0.15,
                           left=0.07, right=0.98, top=0.95, bottom=0.03)

    # Row 0: Originals
    ax00 = fig.add_subplot(gs[0,0]); ax00.imshow(cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB))
    ax01 = fig.add_subplot(gs[0,1]); ax01.imshow(cv2.cvtColor(rgb2, cv2.COLOR_BGR2RGB))

    # Row 1: Optical flow | Scene flow
    ax10 = fig.add_subplot(gs[1,0]); ax10.imshow(cv2.cvtColor(fv, cv2.COLOR_BGR2RGB))
    ax11 = fig.add_subplot(gs[1,1]); ax11.imshow(cv2.cvtColor(sv, cv2.COLOR_BGR2RGB))

    # Row 2: YOLOv8 instance seg | Dynamic mask
    ax20 = fig.add_subplot(gs[2,0]); ax20.imshow(cv2.cvtColor(inst_vis, cv2.COLOR_BGR2RGB))
    ax21 = fig.add_subplot(gs[2,1]); ax21.imshow(dm, cmap='hot')

    # Titles
    titles = ["Frame t","Frame t+1",
              "Optical Flow (t→t+1)","Scene Flow",
              "YOLOv8 Instances","Final Dynamic Mask"]
    for ax, ti in zip([ax00,ax01,ax10,ax11,ax20,ax21], titles):
        ax.set_title(ti, color='#cccccc', fontsize=9, pad=4)
        ax.axis('off')

    # Row labels
    for row, lab in enumerate(["Original","Flow & Scene","Seg & Mask"]):
        fig.text(0.01, 0.83 - row*0.30, lab, color='white', fontsize=10,
                 fontweight='bold', rotation=90, va='center')

    # Legend for instance seg
    legend_items = [
        mpatches.Patch(color=np.array(COL_RIGID[::-1])/255,     label='Rigid (skipped)'),
        mpatches.Patch(color=np.array(COL_DYNAMIC[::-1])/255,   label='Dynamic → masked'),
        mpatches.Patch(color=np.array(COL_STATIC_NR[::-1])/255, label='Non-rigid Static'),
    ]
    ax20.legend(handles=legend_items, loc='lower left', fontsize=8,
                framealpha=0.7, facecolor='#222')

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close(fig)
    print(f"[✓] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. TUM LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_tum_pairs(tum_dir, max_pairs=10):
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


# ─────────────────────────────────────────────────────────────────────────────
# 12. PROCESS ONE PAIR
# ─────────────────────────────────────────────────────────────────────────────
def process_pair(r1, r2, d1, d2, intr, out_dir, idx=0):
    rgb1=cv2.imread(str(r1)); rgb2=cv2.imread(str(r2))
    dt  =cv2.imread(str(d1),cv2.IMREAD_UNCHANGED)
    dt1 =cv2.imread(str(d2),cv2.IMREAD_UNCHANGED)
    if rgb1 is None or rgb2 is None: print(f"[SKIP] {r1}"); return
    if dt   is None or dt1  is None: print(f"[SKIP] {d1}"); return

    H,W = rgb1.shape[:2]
    if dt.shape[:2]!=(H,W):
        dt =cv2.resize(dt, (W,H),interpolation=cv2.INTER_NEAREST)
        dt1=cv2.resize(dt1,(W,H),interpolation=cv2.INTER_NEAREST)

    fx,fy,cx,cy = intr['fx'],intr['fy'],intr['cx'],intr['cy']

    flow,fv  = compute_optical_flow(str(r1),str(r2))
    sf,vm    = compute_scene_flow(dt,dt1,flow,fx,fy,cx,cy)
    sv       = visualize_scene_flow(sf,vm,flow)

    dm, instances, inst_vis = make_dynamic_mask(
        sf, vm, flow, dt, rgb1,
        debug_dir=str(Path(out_dir)/"debug"),
        pair_idx=idx)

    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)

    # ── Visualisation outputs ─────────────────────────────────────────────
    cv2.imwrite(str(od/f"pair{idx:04d}_optical_flow.png"),  fv)
    cv2.imwrite(str(od/f"pair{idx:04d}_scene_flow.png"),    sv)
    cv2.imwrite(str(od/f"pair{idx:04d}_instance_seg.png"),  inst_vis)
    np.save(str(od/f"pair{idx:04d}_scene_flow.npy"), sf)

    # ── Dynamic masks — named after RGB timestamps for ORB-SLAM3 ──────────
    # Saved in results/masks/ — filename = rgb frame timestamp (no extension)
    # ORB-SLAM3 matches mask to frame by timestamp filename
    masks_dir = od / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    ts1 = Path(r1).stem   # e.g. 1341846313.553992
    ts2 = Path(r2).stem   # e.g. 1341846313.588650

    cv2.imwrite(str(masks_dir / f"{ts1}.png"), dm)
    cv2.imwrite(str(masks_dir / f"{ts2}.png"), dm)
    print(f"  [masks] saved {ts1}.png + {ts2}.png → {masks_dir}")

    display_and_save(rgb1,rgb2,fv,sv,dm,inst_vis,instances,
                     str(od/f"pair{idx:04d}_summary.png"),idx)
    return sf,vm


# ─────────────────────────────────────────────────────────────────────────────
# 13. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p=argparse.ArgumentParser(
        description="Scene Flow + YOLOv8 dynamic segmentation for ORB-SLAM3")
    p.add_argument("--rgb1");   p.add_argument("--rgb2")
    p.add_argument("--depth1"); p.add_argument("--depth2")
    p.add_argument("--tum_dir")
    p.add_argument("--max_pairs",type=int,default=9999,
                   help="Max frame pairs to process (default: all)")
    p.add_argument("--sequence",default="fr3",choices=["fr1","fr2","fr3"])
    p.add_argument("--output_dir",default="./scene_flow_results")
    args=p.parse_args()

    intr=TUM_INTRINSICS[args.sequence]
    print(f"\n{'='*55}")
    print(f"  Scene Flow + YOLOv8 — TUM {args.sequence.upper()}")
    print(f"  fx={intr['fx']} fy={intr['fy']} cx={intr['cx']} cy={intr['cy']}")
    print(f"  Output → {args.output_dir}")
    print(f"{'='*55}\n")

    if args.tum_dir:
        pairs=load_tum_pairs(args.tum_dir,args.max_pairs)
        if not pairs: print("[ERROR] No pairs."); sys.exit(1)
        print(f"Found {len(pairs)} pair(s).\n")
        for i,(r1,r2,d1,d2) in enumerate(tqdm(pairs,desc="Processing")):
            process_pair(r1,r2,d1,d2,intr,args.output_dir,i)
    elif args.rgb1 and args.depth1:
        process_pair(args.rgb1,args.rgb2,args.depth1,args.depth2,
                     intr,args.output_dir,0)
    else:
        p.print_help(); sys.exit(1)

    print(f"\n[✓] Done! → {os.path.abspath(args.output_dir)}")

if __name__=="__main__":
    main()
