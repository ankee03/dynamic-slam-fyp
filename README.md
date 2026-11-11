# Dynamic Object Segmentation Using Lightweight Deep Learning-Based Optical Flow in Visual SLAM Algorithm

Final Year Project, Universiti Teknikal Malaysia Melaka (UTeM).

A lightweight hybrid pipeline that stops moving objects from corrupting RGB-D SLAM. Dynamic regions are detected with optical flow, scene flow and instance segmentation, then masked out of ORB-SLAM3's feature extraction.

## How it works
1. LiteFlowNet estimates optical flow between consecutive frames.
2. Scene flow lifts the 2D motion into 3D metric motion using depth.
3. RANSAC removes camera ego-motion, leaving residual (independent) motion.
4. YOLOv8n-seg segments objects, each classified as rigid or non-rigid.
5. Dual-signal motion confirmation (residual flow + scene-flow depth change) decides which instances are really moving.
6. Binary dynamic masks, named by RGB timestamp, are loaded by a modified ORB-SLAM3, which discards keypoints on dynamic pixels.

   <img width="2176" height="1409" alt="SEGMENTATION MODEL" src="https://github.com/user-attachments/assets/969de1bd-3cb0-4288-99bb-d620ec7a7cab" />

## Results (ATE RMSE, TUM RGB-D fr3)

| Sequence | Baseline ORB-SLAM3 | Proposed | Improvement |
|---|---|---|---|
| walking_xyz | 0.7166 m | 0.0153 m | 97.9% |
| walking_static | 0.3808 m | 0.0106 m | 97.2% |
| walking_rpy | 0.5604 m | 0.0279 m | 95.0% |
| walking_halfsphere | 0.4997 m | 0.0230 m | 95.4% |

On the sitting sequences (almost no motion) the change stays within about 1 cm of the baseline.

## Repository contents
- `mask.py`: segmentation pipeline that generates the dynamic masks
- `run.py`: LiteFlowNet inference
- `association.py`: builds association files for ORB-SLAM3
- `trajplot.py`, `plot_rpe_graph.py`: ATE and RPE plots
- `visualMap.py`: dense point-cloud comparison, with and without masks
- `ORB_SLAM3/`: ORB-SLAM3 with dynamic mask support (modified files: Frame.h/.cc, Tracking.h/.cc, System.h/.cc, Examples/RGB-D/rgbd_tum.cc)

Not included: TUM datasets, generated masks, compiled binaries. Model weights (LiteFlowNet, YOLOv8n-seg) are downloaded automatically on first run.

## Acknowledgements
- ORB-SLAM3 (Campos et al., 2021)
- LiteFlowNet (Hui et al., 2018) and the pytorch-liteflownet implementation by sniklaus
- YOLOv8 (Ultralytics)
- TUM RGB-D benchmark (Sturm et al., 2012)

These are open-source projects, so their original licences apply to their code.
