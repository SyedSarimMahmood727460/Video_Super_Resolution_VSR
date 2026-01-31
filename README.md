# Video Super-Resolution (Mahti)

Lightweight VSR pipeline for Mahti. This setup:
- Accepts input videos at **exactly** 360p, 480p, or 720p
- Upscales to **specific targets only**:
  - 360p → 480p / 720p / 1080p
  - 480p → 720p
  - 720p → 1080p
- Writes each run to `output/run_XXX/` with logs + reports
- Uses high‑quality encoding (CRF 16, BT.709 metadata)

## Quick Start (Mahti)

### 1) Start a VS Code session on Mahti
Recommended settings:
- Partition: `gpusmall`
- CPUs: `1`
- Memory: `2 GiB`
- Compiler: `gcc/11.2.0` (or latest)
- Python module: `pytorch/2.9` (or latest)
- Other modules: cuda

### 2) Clone the repo on scratch
```bash
cd /scratch/project_2016196/<your_user>
git clone <your_repo_url> Video-Super-Resolution-VSR-
cd Video-Super-Resolution-VSR-
```

### 3) Prepare input/output folders if not available
```bash
mkdir -p input output 
```

Place your input video here (must be 360p, 480p, or 720p):
```
input/input.mp4
```

### 4) Run
```bash
bash main.sh
```

The script will create a run folder:
```
output/run_001/
  out_720p.mp4 (or out_1080p.mp4)
  metrics.json
  report.txt
  kernel_report.txt
  run.log
```

## Targets & Overrides

By default, the target height is auto‑selected:
- 360p → 720p
- 480p → 720p
- 720p → 1080p

You can override the target height at runtime:
```bash
TARGET_HEIGHT=1080 bash main.sh
```

Allowed targets are enforced. Any other input resolution or target will error.

## Checkpoint

Place the pretrained checkpoint here:
```
checkpoints/litevsr_scale2.pt
```
If it is missing, the script exits with an error.

## Output Notes

Color is encoded with BT.709 metadata for consistency. If you see a color shift:
- Try `PIX_FMT=yuv420p` for broader player compatibility:
  ```bash
  PIX_FMT=yuv420p bash main.sh
  ```

## Troubleshooting

- **Input not found**: ensure `input/input.mp4` exists.
- **Resolution not allowed**: input must be exactly 360p / 480p / 720p.
- **Stale file handle**: re‑enter the directory and re‑run:
  ```bash
  cd /scratch/project_2016196/<your_user>/Video-Super-Resolution-VSR-
  bash main.sh
  ```

## Files ignored by Git

`input/` and `output/` folders are tracked, but their contents are ignored.
