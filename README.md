# Satellite Damage Assessment Pipeline

The current script implements CLIP scene validation followed by before-image OWLv2 building detection, box merging, padding, and OpenCV roof orientation estimation. Matching polygons are projected onto the after image. Stage 3 damage assessment is being developed separately.

See [designReport.md](designReport.md) for the current architecture and proposed production deployment.

## Repository structure

```text
pipeline_stage1_2.py
pipeline/
    __init__.py
    config.py
    images.py
    geometry.py
    stage1.py
    stage2.py
```

`pipeline_stage1_2.py` coordinates the stages and remains the command-line entry point. Edit shared settings in `pipeline/config.py`. Image lookup lives in `images.py`, box merging and roof orientation in `geometry.py`, CLIP validation in `stage1.py`, and OWLv2 detection and export in `stage2.py`.

Add future stages as `pipeline/stage3.py` and `pipeline/stage4.py`, then import and call their entry functions from `main` in execution order. Stage 3 can consume Stage 2's existing JSON and source images. No placeholder stages are included.

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

The requirements pin the six direct dependencies, including `anomalib` for Stage 3, to versions observed in the current development environment. They are not a complete lock of transitive dependencies, and a fresh installation on every platform has not been validated. Record and review dependency changes together when Stage 3 introduces additional packages; avoid replacing this list with a freeze of an unrelated global environment.

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

Name files `before<N>` and `after<N>` with a supported extension: `.png`, `.jpg`, `.jpeg`, `.tif`, or `.tiff`. In `pipeline/config.py`, set `PAIR_ID` to match the pair you supplied. The current configuration selects pair **2**. Before/after images must have identical dimensions and represent spatially aligned tiles.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe pipeline_stage1_2.py
```

Enter a target query at the prompt, or press Enter for the default building query. On the first run, Hugging Face downloads the CLIP and OWLv2 checkpoint files as their stages are reached, requiring internet access and local storage. Package installation does not download these weights. Later runs normally reuse the local Hugging Face cache. Weights are not committed to Git.

Stage 2 runs only if both images pass Stage 1. A successful Stage 2 writes:

- `boxedOutput/before<N>_boxed.png`
- `boxedOutput/after<N>_boxed.png`
- `stage2json/boxes.json`

Outputs are ignored by Git and recreated by the pipeline. Repeated runs overwrite these paths. A failed or rejected run may leave older outputs present, so check the terminal verdict before using an existing JSON file.

## Collaborating on Stage 3

Commit source code, `requirements.txt`, this README, `.gitignore`, and the design report. Work on branches and review changes before merging. When Stage 3 adds dependencies, update the shared requirements and ask collaborators to rerun the installation command.

Stage 3 should read `stage2json/boxes.json` and the **original** before/after images. Match entries by pair and `box_id`; use `bbox_xyxy` for rectangular crops and `rotated_corners` for masks or alignment. Both sections contain identical before-derived geometry. Detection confidence is not damage confidence.

For a tracked integration fixture, place a small image pair and its example JSON under `examples/`, with source paths adjusted to that fixture. There is no blanket ignore rule for image or JSON extensions, so those examples can be committed. Do not use the ignored runtime output folders for fixtures you intend to share.

Virtual environments, private imagery, generated results, cached models, credentials, and editor preferences stay local. `.env.example` files can be committed with placeholder values. Docker containers and the React/backend stack described in the design report are future work and are not required to run this script.
