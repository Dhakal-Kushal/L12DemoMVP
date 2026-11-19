# Satellite Damage Assessment Pipeline

The current script implements CLIP scene validation followed by before-image OWLv2 building detection, score-ranked non-maximum suppression (NMS), padding, and SAM 2 roof segmentation with an OpenCV rectangle fitted to each mask. Stage 2 filters boxes by size, keeps detections in descending confidence order, and suppresses a candidate if its IoU with any retained box exceeds `STAGE2_NMS_IOU_THRESHOLD` (default 0.50). It compares unpadded boxes, then pads each retained box by 10% per side. Matching polygons are projected onto the after image. Stage 3 scores each after-building crop against its matching before crop with AnomalyDINO and projects their heatmaps onto the full after scene.

See [designReport.md](designReport.md) for the current architecture and proposed production deployment.

## Repository structure

```text
pipeline_stage1_2.py
pipeline/
    __init__.py
    config.py
    images.py
    runs.py
    geometry.py
    heatmaps.py
    stage1.py
    stage2.py
    stage3.py
```

`pipeline_stage1_2.py` coordinates the stages and remains the command-line entry point. Edit shared settings in `pipeline/config.py`. Image lookup lives in `images.py`, NMS and roof orientation in `geometry.py`, CLIP validation in `stage1.py`, OWLv2 detection and export in `stage2.py`, and paired anomaly assessment in `stage3.py`.

Stages 1, 2, and 3 run in sequence from `main`. A future Stage 4 can be added as `pipeline/stage4.py` and called after Stage 3 returns its report path.

## Setup after cloning

Git shares source code and dependency declarations. It does not install Python packages or copy another developer's virtual environment. Each collaborator creates their own environment and installs `requirements.txt`.

Use **Python 3.11**, matching the development environment. From the cloned repository directory, run the following in PowerShell. These commands call the environment's Python directly, so PowerShell script activation is unnecessary.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

On Linux or macOS with Python 3.11 installed:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

The requirements pin the direct dependencies to versions observed in the current development environment. They are not a complete lock of transitive dependencies, and a fresh installation on every platform has not been validated. Record and review dependency changes together when Stage 3 introduces additional packages; avoid replacing this list with a freeze of an unrelated global environment.

For how pip installs requirements files, see the [pip user guide](https://pip.pypa.io/en/stable/user_guide/#requirements-files).

## CPU and GPU setup

The script defaults to `DEVICE = "cpu"`; no GPU is required. Standard PyTorch packages differ by operating system and installation index. For an explicitly CPU-only Windows/Linux setup, install PyTorch first from its CPU index, then install the remaining requirements as above:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an NVIDIA GPU, use the [official PyTorch installer](https://pytorch.org/get-started/locally/) to select a compatible CUDA build inside this same virtual environment. Ensure its base PyTorch version satisfies the requirements pin; if a different release is needed, agree on and validate a dependency update instead of silently mixing versions. A CUDA-enabled build can still run the script on CPU.

Verify GPU availability before setting `DEVICE = "cuda"` in `pipeline/config.py`:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The second line should print `True` for CUDA operation. Merely changing `DEVICE` does not add CUDA support to a CPU-only PyTorch installation.

## Input data and running

The sample images in `inputImages/` are tracked by Git and included when cloning the repository. Add additional shareable image pairs to this folder as needed; keep private imagery outside the repository.

Name files `before<N>` and `after<N>` with a supported extension: `.png`, `.jpg`, `.jpeg`, `.tif`, or `.tiff`. At startup, select a complete pair by its numeric ID. Press Enter to use the displayed default, which is the configured `PAIR_ID` when available or the lowest available ID otherwise. Before/after images must have identical dimensions and represent spatially aligned tiles.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe pipeline_stage1_2.py
```

Enter a target query at the prompt, or press Enter for the default building query. On the first run, Hugging Face downloads the the selected Stage 1, detector, segmenter, and Stage 3 checkpoint files as their stages are reached, requiring internet access and local storage. Package installation does not download these weights. Later runs normally reuse the local Hugging Face cache. Weights are not committed to Git.

### Stage 1 scene screening

Stage 1 checks target relevance on the before image and ground visibility on both images. A clear after image can pass even when the building has become rubble. The gate rejects when a competing irrelevant scene beats the requested target on the before image, or when the cloud prompt beats the strongest visible-ground prompt on either image, by at least `STAGE1_REJECT_MARGIN` raw logits. The starting margin of 2.0 is provisional and model dependent. A smaller adverse lead passes with an uncertainty message. The console records the model, each image's verdict and reason. Scene scores do not measure cloud area.

Stage 1 defaults to RemoteCLIP ViT-B/32 with pinned weights and OpenCLIP preprocessing. Run `python -m pipeline.evaluate_stage1` to screen the three example pairs with the configured model. Add `--compare-models` to evaluate both RemoteCLIP and standard CLIP. To select standard CLIP for the pipeline, set `STAGE1_MODEL = "openai/clip-vit-base-patch32"` in `pipeline/config.py`.

For accuracy evaluation, supply `--manifest labels.csv --split development`. The CSV columns are `before,after,query,split,usable,damaged`. Image paths are relative to the CSV; labels use 0 or 1. Use distinct scene pairs in the development and heldout splits. Include damaged but visible buildings, intact buildings, white roofs, partial clouds, severe obstruction, and irrelevant scenes. Tune `--margin` on development pairs, then evaluate the chosen setting with `--split heldout`. The evaluator reports rejected usable pairs (including the damaged subset), admitted unusable pairs, and inference time excluding model loading. Optional `before_cloud_fraction` and `after_cloud_fraction` columns accept independently measured fractions in [0, 1]; coverage of 20% or more gives an evaluation warning only.

### Stage 2 detector and roof outlines

In `pipeline/config.py`, the default detector is `STAGE2_DETECTOR = "owlv2"` with `STAGE2_OUTLINE_METHOD = "sam2"`. Set the outline method to `"otsu"` for the original grayscale baseline. Set the detector to `"grounding_dino"` to try Grounding DINO Tiny with the same tiled detection pipeline. Its box and text cutoffs are configured separately as `STAGE2_DINO_BOX_THRESHOLD` and `STAGE2_DINO_TEXT_THRESHOLD` (initial values 0.25).

SAM 2.1 Hiera Tiny encodes the full before image once and takes unpadded detector boxes as prompts. Stage 2 fits a rectangle to the largest exterior mask contour in source coordinates. Empty, degenerate, out-of-bounds and poorly aligned fits fall back to the detector rectangle; the exported box includes an `outline_fallback` reason. The before and after exports share the same final geometry. Stage 3 scores the padded rectangular crop and uses the outline for heatmap display.

Stage 2 detects on the full before image and on overlapping tiles (default 512 pixels with at least 25% overlap). It skips tile proposals touching an internal crop edge, translates the remaining boxes into scene coordinates, and runs box NMS across all views. After fitting the rotated roof outlines, a second NMS pass drops lower-confidence outlines whose polygon IoU with a retained outline exceeds `STAGE2_OUTLINE_IOU_THRESHOLD` (default 0.70). The detector confidence cutoff defaults to 0.22. A final containment check flags outlines when at least 95% of the smaller outline lies inside the larger one (`STAGE2_CONTAINMENT_THRESHOLD`). It drops a surrounding box if it encloses two candidates sharing at most 10% of the smaller candidate's area. For other nested pairs it keeps the higher-scoring outline. Building IDs are assigned after filtering. To compare with a full-image pass, set `STAGE2_TILE_SIZE = 0` in `pipeline/config.py`.

Stage 2 runs only if both images pass Stage 1. A successful Stage 2 writes:

- `boxedOutput/<run_id>/before<N>_boxed.png`
- `boxedOutput/<run_id>/after<N>_boxed.png`
- `stage2json/<run_id>/boxes.json`
- `stage2json/latest.json` pointing to the latest successful Stage 2 export

Outputs are ignored by Git. Stage 2 generates one shared run ID (`pair<N>_<date>_<time>_<microseconds>`) used by both stages. Repeated pipeline runs preserve earlier results. The latest pointer updates atomically only after Stage 2 writes all its outputs; rejected or failed runs leave it pointing to the previous successful Stage 2 run. This pointer does not indicate Stage 3 success. Both boxed scenes and the heatmap display the run ID and matching `B<id>` building labels. Compare outputs with the same run ID.

## Stage 3: paired building assessment and scene heatmap

After Stage 2 successfully writes its JSON, the entry script passes that exact output path to Stage 3. Stage 1 rejection or Stage 2 failure stops the chain. Stage 3 loads both original images and crops matching rectangles using the rounded, clipped `bbox_xyxy` coordinates. Model inputs retain Stage 2's padding and use original image pixels. Display masking is applied after inference.

For each building, AnomalyDINO builds a reference bank from that building's before crop and scores its after crop. The encoder is loaded once; reference state is cleared for each building. This assumes the before image is undamaged and the source tiles are aligned. Raw anomaly scores are not damage probabilities or severity classifications.

To rerun Stage 3 separately against an existing export:

```powershell
python -m pipeline.stage3 --boxes stage2json/<run_id>/boxes.json --output stage3Heatmaps/review
```

Replace `<run_id>` with the actual folder name. Omitting `--boxes` selects the latest successful Stage 2 export. Existing Stage 3 run folders are never overwritten: to reassess the same export with different visualization settings, use another `--output` directory, as above. Older exports without a run ID or source hashes must be regenerated through Stage 2. The separate reference-directory argument is no longer needed. `--device` and `--output` can override defaults. Anomalib 2.7.0 and timm are included in `requirements.txt`; OpenCV headless supplies `cv2` for both stages.

The configured `STAGE3_OUTPUT_DIR` defaults to `stage3Heatmaps/`. Each invocation uses the exact Stage 2 run ID:

```text
stage3Heatmaps/
    pair3_<timestamp>/
        anomalies.json
        after3_heatmap.png
        building_0_anomaly.npy
```

Each building is cropped and assessed separately in memory. Raw anomaly maps are resized and projected back onto the after scene. Overlaps use the maximum anomaly value and are blended once. Individual crop PNGs are not saved; `.npy` files preserve the original maps and JSON preserves each raw score.

`pipeline/heatmaps.py` handles visualization. By default, heatmap color is restricted to Stage 2's rotated roof polygons; missing or degenerate polygons fall back to crop rectangles. This only affects display, not inference or scores. Use `--show-context` to include the original padded rectangles, including possible rubble around the building.

All displayed buildings share one scale, derived from the 30th and 98th percentiles of the covered scene pixels. Values at or below the lower limit are transparent; higher values progress through yellow, orange, and red with opacity rising to 80%. A flat scene uses a conservative 0-to-at-least-1 fallback to avoid exaggerating tiny numerical differences. This is scene-relative contrast, not calibrated damage severity, and default colors are not directly comparable across scenes.

The image includes the run ID and a dark-backed `B<id> | Anomaly: <raw score>` label for each assessed building. Labels are placed near their building with leader lines, kept within image bounds, and positioned to reduce collisions. In very crowded scenes some overlap may remain. The label is the model's building-level score; the heatmap shows local map values, so they need not have the same numerical maximum. Annotations intentionally occupy some background pixels.

For consistent cross-scene displays, explicitly choose the same range for each run:

```powershell
python -m pipeline.stage3 --display-range 0.15 0.65 --show-context --output stage3Heatmaps/fixedScale
```

These example limits are visualization settings, not validated intact/damaged thresholds. Choose production thresholds using known-intact and known-damaged examples. Low anomaly must not be interpreted as proof that a building is intact.

`anomalies.json` records the shared run ID, matching boxed-image paths, source SHA-256 hashes, and a snapshot of each building's original `bbox_xyxy` and `rotated_corners`, alongside scores, integer crop bounds, raw-map paths, display mask type, and success/failure status. Stage 3 verifies the exact source-file bytes against Stage 2 hashes before loading the model, so replacing imagery under the same filename stops assessment instead of silently mixing runs. It also records the full-scene image path, overlap rule, selected range, scale mode, opacity settings and label positions. Crop or assessment errors are recorded while remaining buildings continue. Missing source images, mismatched dimensions/IDs, model initialization failures, out-of-memory errors, and output-write failures stop execution. Zero detections produce an empty report and annotated scene without loading AnomalyDINO.


A report has status `completed`, `partial`, `failed`, or `empty`, with success/failure counts. Generated outputs are ignored by Git. Old run directories are retained, and the exact new report path is printed and returned by Stage 3.

For a tracked integration fixture, place a small image pair and its example JSON under `examples/`, with source paths adjusted to that fixture. There is no blanket ignore rule for image or JSON extensions, so those examples can be committed. Do not use the ignored runtime output folders for fixtures you intend to share.

Virtual environments, private imagery, generated results, cached models, credentials, and editor preferences stay local. `.env.example` files can be committed with placeholder values. Docker containers and the React/backend stack described in the design report are future work and are not required to run this script.
