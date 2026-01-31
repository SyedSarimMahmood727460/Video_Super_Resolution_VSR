#!/bin/bash
set -e

echo "========================================="
echo "=== LiteVSR Video Super-Resolution ==="
echo "========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

INPUT_DIR="$PROJECT_DIR/input"
OUTPUT_DIR="$PROJECT_DIR/output"
METRICS_DIR="$OUTPUT_DIR"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

# Create an incremented run folder: output/run_001, run_002, ...
RUN_INDEX=1
while true; do
  RUN_DIR=$(printf "%s/run_%03d" "$OUTPUT_DIR" "$RUN_INDEX")
  if [ ! -d "$RUN_DIR" ]; then
    mkdir -p "$RUN_DIR"
    break
  fi
  RUN_INDEX=$((RUN_INDEX + 1))
  if [ "$RUN_INDEX" -gt 9999 ]; then
    echo "ERROR: Too many runs in $OUTPUT_DIR"
    exit 1
  fi
done

INPUT_FILE="${INPUT_FILE:-$INPUT_DIR/input.mp4}"
METRICS_FILE="${METRICS_FILE:-$RUN_DIR/metrics.json}"
REPORT_FILE="${REPORT_FILE:-$RUN_DIR/report.txt}"
KERNEL_REPORT_FILE="${KERNEL_REPORT_FILE:-$RUN_DIR/kernel_report.txt}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/run.log}"

CKPT="${CKPT:-$PROJECT_DIR/checkpoints/litevsr_scale2.pt}"
SCALE="${SCALE:-2}"
WINDOW="${WINDOW:-5}"
TARGET_HEIGHT="${TARGET_HEIGHT:-}"
CRF="${CRF:-16}"
PRESET="${PRESET:-slow}"
TUNE="${TUNE:-film}"
PIX_FMT="${PIX_FMT:-yuv444p}"
COLORSPACE="${COLORSPACE:-bt709}"
COLOR_PRIMARIES="${COLOR_PRIMARIES:-bt709}"
COLOR_TRC="${COLOR_TRC:-bt709}"
COLOR_RANGE="${COLOR_RANGE:-tv}"
KERNEL_PROFILE_STEPS="${KERNEL_PROFILE_STEPS:-1}"

# Load environment
echo "Loading modules (gcc/11.2.0, pytorch/2.7)..."
module load gcc/11.2.0
module load pytorch/2.7
echo "Modules loaded"
echo ""

# Go to project directory
echo "Navigating to project directory..."
cd "$PROJECT_DIR"
echo "Current directory: $(pwd)"
echo ""

# Activate venv (create if missing)
echo "Activating Python virtual environment..."
if [ -d .venv ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Virtual environment not found, creating..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -e .
    echo "Virtual environment created and dependencies installed"
fi
echo ""

# Check if input file exists
echo "Checking input file..."
if [ ! -f "$INPUT_FILE" ]; then
    echo "========================================="
    echo "ERROR: Input file not found!"
    echo "  Expected: $INPUT_FILE"
    echo ""
    echo "Place your video at $INPUT_DIR/input.mp4 and try again."
    echo "========================================="
    exit 1
fi
echo "Input file found: $INPUT_FILE"
echo ""

# Check checkpoint
echo "Checking model checkpoint..."
if [ ! -f "$CKPT" ]; then
    echo "========================================="
    echo "ERROR: Checkpoint not found!"
    echo "  Expected: $CKPT"
    echo ""
    echo "Please place a pretrained checkpoint at that path."
    echo "========================================="
    exit 1
fi
echo "Checkpoint found: $CKPT"
echo ""

# Get video resolution
echo "Detecting input video resolution..."
INPUT_FILE="$INPUT_FILE" python3 << 'EOF'
import cv2
import os

path = os.environ["INPUT_FILE"]
cap = cv2.VideoCapture(path)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

with open('.video_info.txt', 'w') as f:
    f.write(f"{height}\n{width}\n{fps}")

print(f"Input Resolution: {width}x{height} @ {fps:.2f} FPS")
EOF

# Read resolution info
read INPUT_HEIGHT INPUT_WIDTH INPUT_FPS < .video_info.txt
rm -f .video_info.txt

echo "Resolution detected: ${INPUT_WIDTH}x${INPUT_HEIGHT} @ ${INPUT_FPS} FPS"
echo ""

# Auto-select target height if not provided
if [ -z "$TARGET_HEIGHT" ]; then
  case "$INPUT_HEIGHT" in
    360) TARGET_HEIGHT=720 ;;
    480) TARGET_HEIGHT=720 ;;
    720) TARGET_HEIGHT=1080 ;;
  esac
  echo "Target output height: ${TARGET_HEIGHT}p (auto)"
else
  echo "Target output height: ${TARGET_HEIGHT}p (override)"
fi
echo ""

# Validate allowed input/target combinations
case "$INPUT_HEIGHT" in
  360)
    if [ "$TARGET_HEIGHT" -ne 480 ] && [ "$TARGET_HEIGHT" -ne 720 ] && [ "$TARGET_HEIGHT" -ne 1080 ]; then
      echo "========================================="
      echo "ERROR: For 360p input, target must be 480, 720, or 1080."
      echo "========================================="
      exit 1
    fi
    ;;
  480)
    if [ "$TARGET_HEIGHT" -ne 720 ]; then
      echo "========================================="
      echo "ERROR: For 480p input, target must be 720."
      echo "========================================="
      exit 1
    fi
    ;;
  720)
    if [ "$TARGET_HEIGHT" -ne 1080 ]; then
      echo "========================================="
      echo "ERROR: For 720p input, target must be 1080."
      echo "========================================="
      exit 1
    fi
    ;;
  *)
    echo "========================================="
    echo "ERROR: Input height must be exactly 360, 480, or 720."
    echo "========================================="
    exit 1
    ;;
esac

OUTPUT_FILE="${OUTPUT_FILE:-$RUN_DIR/out_${TARGET_HEIGHT}p.mp4}"

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "GPU available"
else
    echo "WARNING: nvidia-smi not found (using CPU)"
fi
echo ""

# Run inference
echo "========================================="
echo "Starting inference..."
echo "  Input:      ${INPUT_WIDTH}x${INPUT_HEIGHT} @ ${INPUT_FPS} FPS"
echo "  Scale:      ${SCALE}x (LiteVSR)"
echo "  Target:     ${TARGET_HEIGHT}p (forced)"
echo "  Output:     $OUTPUT_FILE"
echo "  Run dir:    $RUN_DIR"
echo "========================================="
echo ""

python3 -m vsr.infer \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --ckpt "$CKPT" \
  --scale "$SCALE" --window "$WINDOW" \
  --channels_last \
  --target_height "$TARGET_HEIGHT" \
  --crf "$CRF" --preset "$PRESET" --tune "$TUNE" \
  --pix_fmt "$PIX_FMT" \
  --colorspace "$COLORSPACE" \
  --color_primaries "$COLOR_PRIMARIES" \
  --color_trc "$COLOR_TRC" \
  --color_range "$COLOR_RANGE" \
  --benchmark --warmup 5 --measure 30 \
  --save_metrics "$METRICS_FILE" \
  --report_path "$REPORT_FILE" \
  --kernel_report_path "$KERNEL_REPORT_FILE" \
  --kernel_profile --kernel_profile_steps "$KERNEL_PROFILE_STEPS" \
  | tee "$LOG_FILE"

echo ""
echo "========================================="
echo "Processing complete!"
echo "========================================="
echo "  Input:    ${INPUT_WIDTH}x${INPUT_HEIGHT}"
echo "  Output:   $OUTPUT_FILE"
echo "  Metrics:  $METRICS_FILE"
echo "  Report:   $REPORT_FILE"
echo "  Kernel:   $KERNEL_REPORT_FILE"
echo "  Log:      $LOG_FILE"
echo "  Run dir:  $RUN_DIR"
echo "========================================="
