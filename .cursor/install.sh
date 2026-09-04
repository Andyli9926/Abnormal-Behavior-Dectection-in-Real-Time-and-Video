#!/usr/bin/env bash
#
# Development-environment bootstrap for the Abnormal Behavior Detection project.
#
# This repository is written for MATLAB R2023b and relies on commercial
# MathWorks toolboxes (Deep Learning Toolbox, Computer Vision Toolbox) plus the
# GoogLeNet add-on. MATLAB itself is proprietary and cannot be licensed or
# installed inside this container, so we provision GNU Octave instead. Octave is
# the open-source, largely MATLAB-compatible interpreter and lets you edit and
# run generic MATLAB `.m` code as well as the image-processing / motion-detection
# parts of this project (see tools/octave/gmm_motion_detection_demo.m).
#
# The script is idempotent: re-running it simply reinstalls/upgrades the same
# packages.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update

# octave           - MATLAB-compatible interpreter
# octave-image     - imopen/imclose/imfill/bwconncomp/regionprops etc.
# octave-statistics- normfit/GMM-style helpers used by the detector demo
# octave-video     - VideoReader/VideoWriter equivalents (VideoReader/videoWriter)
# ffmpeg           - backend codecs for octave-video
apt-get install -y --no-install-recommends \
  octave \
  octave-image \
  octave-statistics \
  octave-video \
  ffmpeg

# Use the non-GUI binary for verification so no X11/Qt display is required.
export QT_QPA_PLATFORM=offscreen

echo "-----------------------------------------------------------------"
octave-cli --version | head -1
echo "Octave packages available:"
octave-cli --eval "pkg list" 2>/dev/null | sed -n '1,40p' || true
echo "Development environment ready."
