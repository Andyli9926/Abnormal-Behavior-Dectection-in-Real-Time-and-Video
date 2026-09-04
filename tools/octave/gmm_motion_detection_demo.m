#!/usr/bin/env octave
%% gmm_motion_detection_demo.m
%
% Environment smoke-test / demo for the Abnormal Behavior Detection project.
%
% The real project targets MATLAB R2023b with the Computer Vision Toolbox and
% Deep Learning Toolbox. Those are proprietary and unavailable in this Linux
% container, so this script reproduces the *GMM-based motion-detection* stage of
% the pipeline (see MotionTrackingwithLabel.m) using only GNU Octave + the
% octave-image package, on a self-contained synthetic video. It exercises the
% same processing chain the project uses:
%
%   foreground mask -> imopen -> imclose -> imfill('holes') -> blob analysis
%
% and writes annotated output frames + a montage so the environment can be
% verified without any MATLAB license, external dataset, or Windows path.
%
% Usage:
%   octave --no-gui --path tools/octave \
%          --eval "gmm_motion_detection_demo('out')"

% ------------------------------ functions ------------------------------------
function gmm_motion_detection_demo(outdir)
  if nargin < 1 || isempty(outdir)
    outdir = fullfile(pwd, "out");
  end
  pkg load image statistics;
  if ~exist(outdir, "dir")
    mkdir(outdir);
  end

  printf("=== GMM motion-detection demo (Octave %s) ===\n", version());
  printf("Output directory: %s\n", outdir);

  % ---- 1. Build a synthetic surveillance-style clip -------------------------
  H = 180; W = 240; nFrames = 60;
  rand("seed", 42); randn("seed", 42);

  % Static textured background (what the GMM should learn and ignore).
  background = uint8(90 + 25 * rand(H, W));

  % Two "people" moving across the scene on different trajectories.
  objA = struct("w", 22, "h", 40, "x0", 10,  "y0", 60,  "vx", 3.3, "vy", 0.4, "val", 210);
  objB = struct("w", 18, "h", 34, "x0", 210, "y0", 110, "vx", -2.6, "vy", -0.7, "val", 40);

  frames = cell(1, nFrames);
  for k = 1:nFrames
    f = double(background) + 8 * randn(H, W);   % per-frame sensor noise
    for obj = {objA, objB}
      o = obj{1};
      x = round(o.x0 + (k - 1) * o.vx);
      y = round(o.y0 + (k - 1) * o.vy);
      f = paint_box(f, x, y, o.w, o.h, o.val);
    end
    frames{k} = uint8(min(max(f, 0), 255));
  end
  printf("Generated %d synthetic frames (%dx%d).\n", nFrames, W, H);

  % ---- 2. GMM-style background model (running Gaussian per pixel) -----------
  % Mirrors vision.ForegroundDetector('NumGaussians',3,'NumTrainingFrames',...).
  % Seed the background with a temporal median over the clip: moving objects
  % wash out, giving a clean, object-free background estimate (no ghosts).
  stack = zeros(H, W, nFrames);
  for k = 1:nFrames
    stack(:, :, k) = double(frames{k});
  end
  alpha = 0.02;                 % learning rate
  mu = median(stack, 3);        % mean model (object-free background)
  v  = 100 * ones(H, W);        % variance model
  kThresh = 2.5;                % Mahalanobis-style threshold (std devs)
  minBlobArea = 150;            % analogous to vision.BlobAnalysis MinimumBlobArea

  se_open  = strel("rectangle", [3 3]);
  se_close = strel("rectangle", [15 15]);

  saveIdx = [15 30 45 60];      % frames to highlight in the montage
  montageTiles = {};
  detCounts = zeros(1, nFrames);
  framesDir = fullfile(outdir, "frames");
  if ~exist(framesDir, "dir"); mkdir(framesDir); end

  for k = 1:nFrames
    frame = double(frames{k});

    % Foreground probability via distance from the learned Gaussian.
    d2 = ((frame - mu) .^ 2) ./ max(v, 1);
    mask = d2 > (kThresh ^ 2);

    % ---- morphological cleanup: identical chain to detectObjects() ----------
    mask = imopen(mask, se_open);
    mask = imclose(mask, se_close);
    mask = imfill(mask, "holes");

    % ---- blob analysis (vision.BlobAnalysis equivalent) ---------------------
    cc = bwconncomp(mask);
    stats = regionprops(cc, "Area", "BoundingBox", "Centroid");
    if isempty(stats)
      keep = stats;
    else
      keep = stats([stats.Area] >= minBlobArea);
    end
    detCounts(k) = numel(keep);

    % Update the background model only where there is no foreground.
    bgUpdate = ~mask;
    mu(bgUpdate) = (1 - alpha) * mu(bgUpdate) + alpha * frame(bgUpdate);
    v(bgUpdate)  = (1 - alpha) * v(bgUpdate) + alpha * ((frame(bgUpdate) - mu(bgUpdate)) .^ 2);

    % Annotate every frame (used to assemble a preview video) and keep a few
    % highlighted stills for the montage.
    rgb = repmat(frames{k}, [1 1 3]);
    for i = 1:numel(keep)
      rgb = draw_rect(rgb, keep(i).BoundingBox, [0 255 0]);
    end
    imwrite(rgb, fullfile(framesDir, sprintf("frame_%03d.png", k)));
    if any(k == saveIdx)
      fn = fullfile(outdir, sprintf("frame_%02d.png", k));
      imwrite(rgb, fn);
      printf("Frame %2d: %d moving object(s) detected -> %s\n", k, numel(keep), fn);
      montageTiles{end + 1} = rgb;
    end
  end

  % ---- 3. Montage of the exported frames -----------------------------------
  if ~isempty(montageTiles)
    montage = build_montage(montageTiles, 2);
    mfn = fullfile(outdir, "motion_detection_montage.png");
    imwrite(montage, mfn);
    printf("Wrote montage: %s\n", mfn);
  end

  printf("Total detections across clip: %d\n", sum(detCounts));
  printf("Mean objects/frame after warm-up: %.2f\n", mean(detCounts(20:end)));
  printf("=== demo complete ===\n");
end

function f = paint_box(f, x, y, w, h, val)
  [H, W] = size(f);
  x1 = max(1, x); y1 = max(1, y);
  x2 = min(W, x + w - 1); y2 = min(H, y + h - 1);
  if x2 >= x1 && y2 >= y1
    f(y1:y2, x1:x2) = val;
  end
end

function rgb = draw_rect(rgb, bbox, color)
  [H, W, ~] = size(rgb);
  x = round(bbox(1)); y = round(bbox(2));
  w = round(bbox(3)); h = round(bbox(4));
  x1 = max(1, x); y1 = max(1, y);
  x2 = min(W, x + w - 1); y2 = min(H, y + h - 1);
  for c = 1:3
    rgb(y1:y2, x1, c) = color(c);
    rgb(y1:y2, x2, c) = color(c);
    rgb(y1, x1:x2, c) = color(c);
    rgb(y2, x1:x2, c) = color(c);
  end
end

function m = build_montage(tiles, cols)
  n = numel(tiles);
  rows = ceil(n / cols);
  [h, w, ~] = size(tiles{1});
  pad = 4;
  m = uint8(zeros(rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3));
  for i = 1:n
    r = floor((i - 1) / cols);
    c = mod(i - 1, cols);
    y0 = r * (h + pad) + pad + 1;
    x0 = c * (w + pad) + pad + 1;
    m(y0:y0 + h - 1, x0:x0 + w - 1, :) = tiles{i};
  end
end
