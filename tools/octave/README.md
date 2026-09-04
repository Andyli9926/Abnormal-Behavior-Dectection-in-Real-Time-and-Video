# Development environment (GNU Octave)

This project is written for **MATLAB R2023b** and depends on commercial
MathWorks toolboxes:

- Deep Learning Toolbox (+ "Deep Learning Toolbox Model for GoogLeNet Network")
- Computer Vision Toolbox

MATLAB is proprietary and cannot be licensed or installed inside the Cloud Agent
container, so the environment provisions **GNU Octave** instead. Octave is the
open-source, largely MATLAB-compatible interpreter. It lets you edit and run
generic MATLAB `.m` code and the image-processing / motion-detection parts of
this project without a MATLAB license.

## What is installed

`.cursor/install.sh` (run automatically by the Cloud Agent `install` step)
provisions:

| Package             | Purpose                                                        |
| ------------------- | ------------------------------------------------------------- |
| `octave`            | MATLAB-compatible interpreter                                  |
| `octave-image`      | `imopen`, `imclose`, `imfill`, `bwconncomp`, `regionprops`, …  |
| `octave-statistics` | statistics helpers                                             |
| `octave-video`      | `VideoReader` / video I/O                                      |
| `ffmpeg`            | codecs for `octave-video` and video assembly                  |

## Runnable demo

```bash
octave-cli --path tools/octave \
           --eval "gmm_motion_detection_demo('out')"
```

`tools/octave/gmm_motion_detection_demo.m` reproduces the **GMM-based
motion-detection** stage of the project (see `MotionTrackingwithLabel.m`) on a
self-contained synthetic clip — no external dataset and no hardcoded Windows
paths. It exercises the exact processing chain the project relies on:

```
foreground mask -> imopen -> imclose -> imfill('holes') -> blob analysis
```

and writes annotated frames, a montage (`out/motion_detection_montage.png`), and
per-frame stills under `out/frames/`.

## Compatibility limits (what still requires MATLAB)

Octave **cannot** run the following, which genuinely require MATLAB + toolboxes:

- `App/*.mlapp` — App Designer GUIs (proprietary format).
- `*.mlx` live scripts — MATLAB Live Editor format.
- `Train/I3D.m`, `Train/R(2+1)D.m` — use Deep Learning Toolbox (`dlnetwork`,
  `trainNetwork`, GoogLeNet, `randomAffine2d`, nested functions, …). These do not
  even parse under Octave.
- `MotionTrackingwithLabel.m` end-to-end — relies on `vision.ForegroundDetector`,
  `vision.BlobAnalysis`, `vision.KalmanFilter`, `insertObjectAnnotation`, etc.
  from the Computer Vision Toolbox. The demo above re-implements its
  motion-detection core with Octave equivalents.

To run the full project, use MATLAB R2023b on a licensed machine with the
toolboxes listed in the top-level `README.md`.
