#!/bin/bash
#SBATCH --job-name=cuda-vsr
#SBATCH --account=project_2016196
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Under SLURM, script is executed from a spool copy path; use submit dir as project root.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/main.sh" ]; then
  echo "ERROR: main.sh not found in PROJECT_DIR=$PROJECT_DIR"
  echo "Submit from repository root, or pass PROJECT_DIR explicitly:"
  echo "  sbatch --export=ALL,PROJECT_DIR=/scratch/project_2016196/$USER/Video-Super-Resolution-VSR- scripts/slurm_run_mahti.sh"
  exit 1
fi

module --force purge || true
module load "${GCC_MODULE:-gcc/10.4.0}"
module load "${CUDA_MODULE:-cuda/12.6.1}"

# Optional: set FFMPEG_MODULE if your cluster provides ffmpeg via modules.
if [ -n "${FFMPEG_MODULE:-}" ]; then
  module load "$FFMPEG_MODULE"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export INPUT_FILE="${INPUT_FILE:-$PROJECT_DIR/input/input.mp4}"
export INPUT_FRAMES_DIR="${INPUT_FRAMES_DIR:-}"
export FRAMES_TAR="${FRAMES_TAR:-$PROJECT_DIR/frames_in.tar.gz}"
export FRAMES_STAGE_DIR="${FRAMES_STAGE_DIR:-$PROJECT_DIR/frames_in}"
export INPUT_FPS="${INPUT_FPS:-}"
export SKIP_VIDEO_ENCODE="${SKIP_VIDEO_ENCODE:-0}"
export SKIP_BUILD="${SKIP_BUILD:-0}"
export FFMPEG_DIR="${FFMPEG_DIR:-/scratch/project_2016196/$USER/bin}"
export FFMPEG_BIN="${FFMPEG_BIN:-}"
export FFPROBE_BIN="${FFPROBE_BIN:-}"
export GCC_MODULE="${GCC_MODULE:-gcc/10.4.0}"
export CUDA_MODULE="${CUDA_MODULE:-cuda/12.6.1}"
export LOAD_MODULES="${LOAD_MODULES:-0}"
export BACKEND="${BACKEND:-cuda}"
export TARGET_HEIGHT="${TARGET_HEIGHT:-0}"

if [ -z "$INPUT_FRAMES_DIR" ] && [ -f "$FRAMES_TAR" ]; then
  echo "Found frames archive: $FRAMES_TAR"
  rm -rf "$FRAMES_STAGE_DIR"
  mkdir -p "$FRAMES_STAGE_DIR"
  tar -xzf "$FRAMES_TAR" -C "$FRAMES_STAGE_DIR"
  INPUT_FRAMES_DIR="$FRAMES_STAGE_DIR"
fi

if [ -d "$FFMPEG_DIR" ]; then
  export PATH="$FFMPEG_DIR:$PATH"
fi

if [ -z "$INPUT_FRAMES_DIR" ] && [ ! -f "$INPUT_FILE" ]; then
  echo "ERROR: no usable input source."
  echo "Missing input video: $INPUT_FILE"
  echo "Missing frames dir:  $INPUT_FRAMES_DIR"
  echo "Missing frames tar:  $FRAMES_TAR"
  echo "Provide one of:"
  echo "  1) INPUT_FILE=<video_path> (requires ffmpeg for frame extraction)"
  echo "  2) INPUT_FRAMES_DIR=<ppm_frames_dir> (no ffmpeg needed)"
  echo "  3) FRAMES_TAR=<frames_in.tar.gz> (auto-unpacked by this script)"
  exit 1
fi
if [ -n "$INPUT_FRAMES_DIR" ] && [ ! -d "$INPUT_FRAMES_DIR" ]; then
  echo "ERROR: INPUT_FRAMES_DIR does not exist: $INPUT_FRAMES_DIR"
  exit 1
fi
if [ -n "$INPUT_FRAMES_DIR" ] && [ "$SKIP_VIDEO_ENCODE" = "0" ] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not available; forcing SKIP_VIDEO_ENCODE=1 for frames mode."
  SKIP_VIDEO_ENCODE=1
fi

echo "Project dir:    $PROJECT_DIR"
echo "GCC module:     $GCC_MODULE"
echo "CUDA module:    $CUDA_MODULE"
if [ -n "$INPUT_FRAMES_DIR" ]; then
  echo "Input frames:   $INPUT_FRAMES_DIR"
else
  echo "Input file:     $INPUT_FILE"
fi
if [ -f "$FRAMES_TAR" ]; then
  echo "Frames tar:     $FRAMES_TAR"
fi
echo "ffmpeg dir:     $FFMPEG_DIR"
echo "ffmpeg path:    $(command -v ffmpeg || echo [none])"
echo "ffprobe path:   $(command -v ffprobe || echo [none])"
echo "Backend:        $BACKEND"
echo "Target height:  $TARGET_HEIGHT"
echo "Skip build:     $SKIP_BUILD"
echo "Skip encode:    $SKIP_VIDEO_ENCODE"
if [ -n "$INPUT_FPS" ]; then
  echo "Input FPS:      $INPUT_FPS"
fi

/bin/bash "$PROJECT_DIR/main.sh"
