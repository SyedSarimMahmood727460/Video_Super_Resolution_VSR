# Video Super-Resolution (CUDA Kernels) on Mahti

This project runs custom CUDA kernels (no PyTorch runtime path) and is set up for Mahti SLURM jobs.

## Pipeline

- Supported input heights: `360`, `480`, `720`
- Output rules:
  - `360 -> 480 / 720 / 1080`
  - `480 -> 720`
  - `720 -> 1080`
- Per run output folder: `output/run_XXX/`
  - `out.mp4`
  - `metrics.json`
  - `report.txt`
  - `kernel_report.txt`
  - `run.log`
  - `frames_in/`, `frames_out/`

## Mahti Run Guide

### 1) Clone to scratch

```bash
cd /scratch/project_2016196/$USER
git clone git@github.com:SyedSarimMahmood727460/Video_Super_Resolution_VSR.git
cd Video_Super_Resolution_VSR
mkdir -p input output
```

### 2) Install ffmpeg once (user space)

Mahti may not provide an `ffmpeg` module. Install a local static binary once:

```bash
bash scripts/install_ffmpeg_user.sh /scratch/project_2016196/$USER/bin
```

Optional check:

```bash
/scratch/project_2016196/$USER/bin/ffmpeg -version
/scratch/project_2016196/$USER/bin/ffprobe -version
```

### 3) Put video input

```bash
cp /path/to/your_video.mp4 input/input.mp4
```

You can also keep any filename/path and pass it with `INPUT_FILE=...` during submit.

### 4) Submit GPU job

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,FFMPEG_DIR=/scratch/project_2016196/$USER/bin,INPUT_FILE=$PWD/input/input.mp4 scripts/slurm_run_mahti.sh
```

With explicit target height example:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,FFMPEG_DIR=/scratch/project_2016196/$USER/bin,INPUT_FILE=$PWD/input/input.mp4,TARGET_HEIGHT=1080 scripts/slurm_run_mahti.sh
```

### 5) Monitor job

```bash
squeue -u $USER
tail -f slurm-<jobid>.out
```

### 6) Get results

```bash
latest=$(ls -dt output/run_* | head -n1)
ls -lh "$latest"
cat "$latest/report.txt"
```

Main output video:

```bash
$latest/out.mp4
```

## Common Overrides

- `INPUT_FILE`: input video path (default `input/input.mp4`)
- `TARGET_HEIGHT`: `0` (auto), or set `480` / `720` / `1080`
- `BACKEND`: `cuda` (default) or `cpu`
- `SKIP_BUILD=1`: reuse existing `cuda_vsr/bin/cuda_vsr`
- `SKIP_VIDEO_ENCODE=1`: skip mp4 encoding (frames + metrics only)
- `GCC_MODULE` and `CUDA_MODULE`: module versions used by job script

Submit with custom modules:

```bash
sbatch --export=ALL,PROJECT_DIR=$PWD,FFMPEG_DIR=/scratch/project_2016196/$USER/bin,GCC_MODULE=gcc/10.4.0,CUDA_MODULE=cuda/12.6.1 scripts/slurm_run_mahti.sh
```

## Troubleshooting

- `AssocMaxSubmitJobLimit`:
  - You already have a running job/session (often VS Code compute session). Wait or stop that job, then submit again.
- `nvcc not found`:
  - Ensure valid module combo on Mahti (for example `gcc/10.4.0` + `cuda/12.6.1`).
- `ffmpeg not found`:
  - Re-run `scripts/install_ffmpeg_user.sh` and submit with `FFMPEG_DIR=/scratch/project_2016196/$USER/bin`.
- `input video not found`:
  - Check `INPUT_FILE` path in submit command.

## Relevant Files

- `main.sh`: entry wrapper
- `scripts/slurm_run_mahti.sh`: primary SLURM submit script
- `scripts/run_cuda_vsr_mahti.sh`: build + extract + kernel run + encode
- `scripts/install_ffmpeg_user.sh`: one-time ffmpeg install to user bin
