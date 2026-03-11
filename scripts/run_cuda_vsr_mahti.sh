#!/bin/bash
set -euo pipefail

now_ms() {
  date +%s%3N
}

elapsed_sec() {
  local start_ms="$1"
  local end_ms="$2"
  awk -v s="$start_ms" -v e="$end_ms" 'BEGIN { printf "%.3f", (e - s) / 1000.0 }'
}

json_number() {
  local key="$1"
  local file="$2"
  awk -v key="\"$key\"" -F: '
    $0 ~ key {
      val = $2
      gsub(/[, ]/, "", val)
      gsub(/\r/, "", val)
      print val
      exit
    }
  ' "$file"
}

percent_of() {
  local part="$1"
  local total="$2"
  awk -v p="$part" -v t="$total" 'BEGIN { if ((t + 0) > 0) printf "%.1f", (p / t) * 100.0; else printf "0.0" }'
}

safe_div() {
  local num="$1"
  local den="$2"
  awk -v n="$num" -v d="$den" 'BEGIN { if ((d + 0) > 0) printf "%.3f", n / d; else printf "0.000" }'
}

echo "========================================="
echo "=== CUDA VSR (Custom Kernels) - Mahti ==="
echo "========================================="
echo ""

RUN_START_MS="$(now_ms)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_DIR="${INPUT_DIR:-$PROJECT_DIR/input}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
INPUT_FILE="${INPUT_FILE:-$INPUT_DIR/input.mp4}"
INPUT_FRAMES_DIR="${INPUT_FRAMES_DIR:-}"
INPUT_FPS="${INPUT_FPS:-}"
SKIP_VIDEO_ENCODE="${SKIP_VIDEO_ENCODE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
FFMPEG_DIR="${FFMPEG_DIR:-}"
FFMPEG_BIN="${FFMPEG_BIN:-}"
FFPROBE_BIN="${FFPROBE_BIN:-}"
GCC_MODULE="${GCC_MODULE:-gcc/10.4.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6.1}"
LOAD_MODULES="${LOAD_MODULES:-1}"

TARGET_HEIGHT="${TARGET_HEIGHT:-0}"
BACKEND="${BACKEND:-cuda}"
FUSION_PREV="${FUSION_PREV:-0.2}"
FUSION_CURR="${FUSION_CURR:-0.6}"
FUSION_NEXT="${FUSION_NEXT:-0.2}"
SHARPEN="${SHARPEN:-0.25}"

CRF="${CRF:-16}"
PRESET="${PRESET:-slow}"
PIX_FMT="${PIX_FMT:-yuv420p}"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

RUN_INDEX=1
while true; do
  RUN_DIR=$(printf "%s/run_%03d" "$OUTPUT_DIR" "$RUN_INDEX")
  if [ ! -d "$RUN_DIR" ]; then
    mkdir -p "$RUN_DIR"
    break
  fi
  RUN_INDEX=$((RUN_INDEX + 1))
  if [ "$RUN_INDEX" -gt 9999 ]; then
    echo "ERROR: Too many run folders in $OUTPUT_DIR"
    exit 1
  fi
done

TMP_IN="$RUN_DIR/frames_in"
TMP_OUT="$RUN_DIR/frames_out"
LOG_FILE="$RUN_DIR/run.log"
METRICS_FILE="$RUN_DIR/metrics.json"
REPORT_FILE="$RUN_DIR/report.txt"
KERNEL_REPORT_FILE="$RUN_DIR/kernel_report.txt"
OUTPUT_FILE="${OUTPUT_FILE:-$RUN_DIR/out.mp4}"

mkdir -p "$TMP_IN" "$TMP_OUT"

BUILD_SEC="0.000"
EXTRACT_SEC="0.000"
KERNEL_SEC="0.000"
ENCODE_SEC="0.000"

if [ "$LOAD_MODULES" = "1" ] && command -v module >/dev/null 2>&1; then
  module --force purge || true
  module load "$GCC_MODULE"
  module load "$CUDA_MODULE"
fi

if [ -n "$FFMPEG_DIR" ] && [ -d "$FFMPEG_DIR" ]; then
  export PATH="$FFMPEG_DIR:$PATH"
fi

if [ -z "$FFMPEG_BIN" ]; then
  FFMPEG_BIN="$(command -v ffmpeg || true)"
fi
if [ -z "$FFPROBE_BIN" ]; then
  FFPROBE_BIN="$(command -v ffprobe || true)"
fi
if [ -z "$FFMPEG_BIN" ] && [ -x "$HOME/bin/ffmpeg" ]; then
  FFMPEG_BIN="$HOME/bin/ffmpeg"
fi
if [ -z "$FFPROBE_BIN" ] && [ -x "$HOME/bin/ffprobe" ]; then
  FFPROBE_BIN="$HOME/bin/ffprobe"
fi

if [ "$SKIP_BUILD" = "1" ]; then
  if [ ! -x "$PROJECT_DIR/cuda_vsr/bin/cuda_vsr" ]; then
    echo "ERROR: SKIP_BUILD=1 but executable is missing: $PROJECT_DIR/cuda_vsr/bin/cuda_vsr"
    echo "Build once first, or run with SKIP_BUILD=0."
    exit 1
  fi
  echo "Skipping build step (SKIP_BUILD=1)."
else
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found in PATH after loading CUDA modules."
    if [ "$LOAD_MODULES" = "0" ]; then
      echo "LOAD_MODULES=0 was set, so this script expected nvcc from the parent environment."
    fi
    echo "Try discovering available CUDA modules with:"
    echo "  module spider cuda"
    echo "Then rerun with explicit modules, for example:"
    echo "  sbatch --export=ALL,GCC_MODULE=gcc/10.4.0,CUDA_MODULE=cuda/12.6.1 scripts/slurm_run_mahti.sh"
    exit 1
  fi

  echo "Building CUDA executable..."
  BUILD_START_MS="$(now_ms)"
  make -C "$PROJECT_DIR/cuda_vsr" clean all
  BUILD_END_MS="$(now_ms)"
  BUILD_SEC="$(elapsed_sec "$BUILD_START_MS" "$BUILD_END_MS")"
  echo ""
fi

KERNEL_INPUT_DIR="$TMP_IN"
if [ -n "$INPUT_FRAMES_DIR" ]; then
  if [ ! -d "$INPUT_FRAMES_DIR" ]; then
    echo "ERROR: INPUT_FRAMES_DIR does not exist: $INPUT_FRAMES_DIR"
    exit 1
  fi
  KERNEL_INPUT_DIR="$INPUT_FRAMES_DIR"
  echo "Using pre-extracted frames from: $KERNEL_INPUT_DIR"
else
  if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: input video not found: $INPUT_FILE"
    exit 1
  fi
  if [ -z "$FFMPEG_BIN" ]; then
    echo "ERROR: ffmpeg is required to extract frames from video."
    echo "Set FFMPEG_DIR/FFMPEG_BIN or provide INPUT_FRAMES_DIR=<path/to/ppm_frames>."
    exit 1
  fi
  echo "Extracting frames from input video..."
  EXTRACT_START_MS="$(now_ms)"
  "$FFMPEG_BIN" -hide_banner -loglevel error -y \
    -i "$INPUT_FILE" \
    -vsync 0 \
    -pix_fmt rgb24 \
    "$TMP_IN/frame_%06d.ppm"
  EXTRACT_END_MS="$(now_ms)"
  EXTRACT_SEC="$(elapsed_sec "$EXTRACT_START_MS" "$EXTRACT_END_MS")"
fi

echo "Running CUDA VSR kernels..."
KERNEL_START_MS="$(now_ms)"
"$PROJECT_DIR/cuda_vsr/bin/cuda_vsr" \
  --input_dir "$KERNEL_INPUT_DIR" \
  --output_dir "$TMP_OUT" \
  --backend "$BACKEND" \
  --target_height "$TARGET_HEIGHT" \
  --w_prev "$FUSION_PREV" \
  --w_curr "$FUSION_CURR" \
  --w_next "$FUSION_NEXT" \
  --sharpen "$SHARPEN" \
  --metrics "$METRICS_FILE" \
  | tee "$LOG_FILE"
KERNEL_END_MS="$(now_ms)"
KERNEL_SEC="$(elapsed_sec "$KERNEL_START_MS" "$KERNEL_END_MS")"

FPS="$INPUT_FPS"
if [ -z "$FPS" ]; then
  if [ -f "$INPUT_FILE" ] && [ -n "$FFPROBE_BIN" ]; then
    FPS_RAW=$("$FFPROBE_BIN" -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nokey=1:noprint_wrappers=1 "$INPUT_FILE" | head -n1)
    FPS=$(echo "$FPS_RAW" | awk -F/ '{ if (NF==2 && $2!=0) printf "%.6f", $1/$2; else if (NF==1 && $1!="") printf "%.6f", $1; else printf "30.000000"; }')
  else
    FPS="30.000000"
  fi
fi

if [ "$SKIP_VIDEO_ENCODE" = "1" ]; then
  echo "Skipping video encoding step (SKIP_VIDEO_ENCODE=1)."
else
  if [ -z "$FFMPEG_BIN" ]; then
    echo "ERROR: ffmpeg is required to encode output video."
    echo "Either set FFMPEG_DIR/FFMPEG_BIN, or rerun with SKIP_VIDEO_ENCODE=1."
    exit 1
  fi
  echo "Encoding output video..."
  ENCODE_START_MS="$(now_ms)"
  "$FFMPEG_BIN" -hide_banner -loglevel error -y \
    -framerate "$FPS" \
    -i "$TMP_OUT/frame_%06d.ppm" \
    -c:v libx264 \
    -crf "$CRF" \
    -preset "$PRESET" \
    -pix_fmt "$PIX_FMT" \
    "$OUTPUT_FILE"
  ENCODE_END_MS="$(now_ms)"
  ENCODE_SEC="$(elapsed_sec "$ENCODE_START_MS" "$ENCODE_END_MS")"
fi

RUN_END_MS="$(now_ms)"
WALL_SEC="$(elapsed_sec "$RUN_START_MS" "$RUN_END_MS")"
PIPELINE_SEC="$(awk -v a="$BUILD_SEC" -v b="$EXTRACT_SEC" -v c="$KERNEL_SEC" -v d="$ENCODE_SEC" 'BEGIN { printf "%.3f", a + b + c + d }')"

FRAMES="0"
IN_W="0"
IN_H="0"
OUT_W="0"
OUT_H="0"
LOAD_MS="0"
H2D_MS="0"
U8_TO_F32_MS="0"
TEMPORAL_MS="0"
RESIZE_MS="0"
SHARPEN_MS="0"
F32_TO_U8_MS="0"
D2H_MS="0"
WRITE_MS="0"
TOTAL_MS="0"
FPS_EST="0"

if [ -f "$METRICS_FILE" ]; then
  FRAMES="$(json_number "frames" "$METRICS_FILE" || true)"; FRAMES="${FRAMES:-0}"
  IN_W="$(json_number "input_width" "$METRICS_FILE" || true)"; IN_W="${IN_W:-0}"
  IN_H="$(json_number "input_height" "$METRICS_FILE" || true)"; IN_H="${IN_H:-0}"
  OUT_W="$(json_number "output_width" "$METRICS_FILE" || true)"; OUT_W="${OUT_W:-0}"
  OUT_H="$(json_number "output_height" "$METRICS_FILE" || true)"; OUT_H="${OUT_H:-0}"
  LOAD_MS="$(json_number "load" "$METRICS_FILE" || true)"; LOAD_MS="${LOAD_MS:-0}"
  H2D_MS="$(json_number "h2d" "$METRICS_FILE" || true)"; H2D_MS="${H2D_MS:-0}"
  U8_TO_F32_MS="$(json_number "u8_to_f32" "$METRICS_FILE" || true)"; U8_TO_F32_MS="${U8_TO_F32_MS:-0}"
  TEMPORAL_MS="$(json_number "temporal" "$METRICS_FILE" || true)"; TEMPORAL_MS="${TEMPORAL_MS:-0}"
  RESIZE_MS="$(json_number "resize" "$METRICS_FILE" || true)"; RESIZE_MS="${RESIZE_MS:-0}"
  SHARPEN_MS="$(json_number "sharpen" "$METRICS_FILE" || true)"; SHARPEN_MS="${SHARPEN_MS:-0}"
  F32_TO_U8_MS="$(json_number "f32_to_u8" "$METRICS_FILE" || true)"; F32_TO_U8_MS="${F32_TO_U8_MS:-0}"
  D2H_MS="$(json_number "d2h" "$METRICS_FILE" || true)"; D2H_MS="${D2H_MS:-0}"
  WRITE_MS="$(json_number "write" "$METRICS_FILE" || true)"; WRITE_MS="${WRITE_MS:-0}"
  TOTAL_MS="$(json_number "total" "$METRICS_FILE" || true)"; TOTAL_MS="${TOTAL_MS:-0}"
  FPS_EST="$(json_number "fps_est" "$METRICS_FILE" || true)"; FPS_EST="${FPS_EST:-0}"
fi

if [ "$FRAMES" = "0" ] && [ -f "$LOG_FILE" ]; then
  FRAMES="$(awk -F': *' '/^Processed frames:/ { print $2; exit }' "$LOG_FILE" || true)"
  FRAMES="${FRAMES:-0}"
fi

if [ "$IN_W" = "0" ] && [ -f "$LOG_FILE" ]; then
  INPUT_SIZE_LOG="$(awk -F': *' '/^Input size:/ { print $2; exit }' "$LOG_FILE" || true)"
  if [ -n "$INPUT_SIZE_LOG" ]; then
    IN_W="${INPUT_SIZE_LOG%x*}"
    IN_H="${INPUT_SIZE_LOG#*x}"
  fi
fi
if [ "$OUT_W" = "0" ] && [ -f "$LOG_FILE" ]; then
  OUTPUT_SIZE_LOG="$(awk -F': *' '/^Output size:/ { print $2; exit }' "$LOG_FILE" || true)"
  if [ -n "$OUTPUT_SIZE_LOG" ]; then
    OUT_W="${OUTPUT_SIZE_LOG%x*}"
    OUT_H="${OUTPUT_SIZE_LOG#*x}"
  fi
fi

SCALE_FACTOR="$(safe_div "$OUT_H" "$IN_H")"

LOAD_PCT="$(percent_of "$LOAD_MS" "$TOTAL_MS")"
H2D_PCT="$(percent_of "$H2D_MS" "$TOTAL_MS")"
U8_TO_F32_PCT="$(percent_of "$U8_TO_F32_MS" "$TOTAL_MS")"
TEMPORAL_PCT="$(percent_of "$TEMPORAL_MS" "$TOTAL_MS")"
RESIZE_PCT="$(percent_of "$RESIZE_MS" "$TOTAL_MS")"
SHARPEN_PCT="$(percent_of "$SHARPEN_MS" "$TOTAL_MS")"
F32_TO_U8_PCT="$(percent_of "$F32_TO_U8_MS" "$TOTAL_MS")"
D2H_PCT="$(percent_of "$D2H_MS" "$TOTAL_MS")"
WRITE_PCT="$(percent_of "$WRITE_MS" "$TOTAL_MS")"

IO_MS="$(awk -v a="$LOAD_MS" -v b="$WRITE_MS" 'BEGIN { printf "%.4f", a + b }')"
TRANSFER_MS="$(awk -v a="$H2D_MS" -v b="$D2H_MS" 'BEGIN { printf "%.4f", a + b }')"
KERNELS_MS="$(awk -v a="$U8_TO_F32_MS" -v b="$TEMPORAL_MS" -v c="$RESIZE_MS" -v d="$SHARPEN_MS" -v e="$F32_TO_U8_MS" 'BEGIN { printf "%.4f", a + b + c + d + e }')"
IO_PCT="$(percent_of "$IO_MS" "$TOTAL_MS")"
TRANSFER_PCT="$(percent_of "$TRANSFER_MS" "$TOTAL_MS")"
KERNELS_PCT="$(percent_of "$KERNELS_MS" "$TOTAL_MS")"

read -r BOTTLENECK_STAGE BOTTLENECK_MS <<< "$(awk \
  -v load="$LOAD_MS" \
  -v h2d="$H2D_MS" \
  -v u8="$U8_TO_F32_MS" \
  -v temporal="$TEMPORAL_MS" \
  -v resize="$RESIZE_MS" \
  -v sharpen="$SHARPEN_MS" \
  -v f32="$F32_TO_U8_MS" \
  -v d2h="$D2H_MS" \
  -v write="$WRITE_MS" \
  'BEGIN {
    max = load; name = "load";
    if (h2d > max) { max = h2d; name = "h2d"; }
    if (u8 > max) { max = u8; name = "u8_to_f32"; }
    if (temporal > max) { max = temporal; name = "temporal"; }
    if (resize > max) { max = resize; name = "resize"; }
    if (sharpen > max) { max = sharpen; name = "sharpen"; }
    if (f32 > max) { max = f32; name = "f32_to_u8"; }
    if (d2h > max) { max = d2h; name = "d2h"; }
    if (write > max) { max = write; name = "write"; }
    printf "%s %.4f", name, max;
  }')"

END_TO_END_FPS="$(safe_div "$FRAMES" "$PIPELINE_SEC")"
MEAN_TOTAL_MS="$(awk -v t="$TOTAL_MS" 'BEGIN { printf "%.4f", t }')"
GPU_INFO="[unknown]"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1 || true)"
  GPU_INFO="${GPU_INFO:-[unknown]}"
fi
NVCC_INFO="[none]"
if command -v nvcc >/dev/null 2>&1; then
  NVCC_INFO="$(nvcc --version | tail -n1 | sed 's/^ *//' || true)"
  NVCC_INFO="${NVCC_INFO:-[none]}"
fi
OUTPUT_VIDEO_SIZE="[skipped]"
if [ "$SKIP_VIDEO_ENCODE" = "0" ] && [ -f "$OUTPUT_FILE" ]; then
  OUTPUT_VIDEO_SIZE="$(du -h "$OUTPUT_FILE" | awk '{print $1}')"
fi

{
  echo "CUDA VSR Performance Report"
  echo "==========================="
  echo "Run: ${RUN_DIR##*/}"
  echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "Project dir: $PROJECT_DIR"
  echo ""
  if [ -n "$INPUT_FRAMES_DIR" ]; then
    echo "Input frames dir: $INPUT_FRAMES_DIR"
  else
    echo "Input video: $INPUT_FILE"
  fi
  echo "Input size: ${IN_W}x${IN_H}"
  echo "Output size: ${OUT_W}x${OUT_H}"
  echo "Scale factor: ${SCALE_FACTOR}x"
  echo "Frames: $FRAMES"
  echo "Input/Output FPS setting: $FPS"
  echo "Estimated kernel FPS: $FPS_EST"
  echo "Mean total/frame: ${MEAN_TOTAL_MS} ms"
  echo ""
  echo "Execution config"
  echo "  Backend: $BACKEND"
  echo "  Target height arg: $TARGET_HEIGHT (0 = auto)"
  echo "  Resolved target height: $OUT_H"
  echo "  Fusion weights: prev=$FUSION_PREV curr=$FUSION_CURR next=$FUSION_NEXT"
  echo "  Sharpen: $SHARPEN"
  echo "  Encoder: crf=$CRF, preset=$PRESET, pix_fmt=$PIX_FMT"
  echo "  Skip build: $SKIP_BUILD"
  echo "  Skip encode: $SKIP_VIDEO_ENCODE"
  echo ""
  echo "Environment"
  echo "  GCC module: $GCC_MODULE"
  echo "  CUDA module: $CUDA_MODULE"
  echo "  nvcc: $NVCC_INFO"
  echo "  ffmpeg: ${FFMPEG_BIN:-[none]}"
  echo "  ffprobe: ${FFPROBE_BIN:-[none]}"
  echo "  GPU: $GPU_INFO"
  echo ""
  echo "Pipeline timing (seconds, end-to-end stages)"
  echo "  build: $BUILD_SEC"
  echo "  extract: $EXTRACT_SEC"
  echo "  kernels: $KERNEL_SEC"
  echo "  encode: $ENCODE_SEC"
  echo "  total (stages sum): $PIPELINE_SEC"
  echo "  total (wall): $WALL_SEC"
  echo "  end-to-end FPS (frames / stages sum): $END_TO_END_FPS"
  echo ""
  echo "Stage timing (ms per frame, from metrics.json)"
  echo "  load: $LOAD_MS"
  echo "  h2d: $H2D_MS"
  echo "  u8_to_f32: $U8_TO_F32_MS"
  echo "  temporal: $TEMPORAL_MS"
  echo "  resize: $RESIZE_MS"
  echo "  sharpen: $SHARPEN_MS"
  echo "  f32_to_u8: $F32_TO_U8_MS"
  echo "  d2h: $D2H_MS"
  echo "  write: $WRITE_MS"
  echo "  total: $TOTAL_MS"
  echo ""
  echo "Bottleneck analysis (% of per-frame total)"
  echo "  load: ${LOAD_PCT}%"
  echo "  h2d: ${H2D_PCT}%"
  echo "  u8_to_f32: ${U8_TO_F32_PCT}%"
  echo "  temporal: ${TEMPORAL_PCT}%"
  echo "  resize: ${RESIZE_PCT}%"
  echo "  sharpen: ${SHARPEN_PCT}%"
  echo "  f32_to_u8: ${F32_TO_U8_PCT}%"
  echo "  d2h: ${D2H_PCT}%"
  echo "  write: ${WRITE_PCT}%"
  echo "  bottleneck: $BOTTLENECK_STAGE (${BOTTLENECK_MS} ms/frame)"
  echo ""
  echo "Grouped bottlenecks"
  echo "  io (load + write): $IO_MS ms (${IO_PCT}%)"
  echo "  transfer (h2d + d2h): $TRANSFER_MS ms (${TRANSFER_PCT}%)"
  echo "  kernels (u8_to_f32 + temporal + resize + sharpen + f32_to_u8): $KERNELS_MS ms (${KERNELS_PCT}%)"
  echo ""
  echo "Artifacts"
  if [ "$SKIP_VIDEO_ENCODE" = "1" ]; then
    echo "  Output video: [skipped]"
  else
    echo "  Output video: $OUTPUT_FILE ($OUTPUT_VIDEO_SIZE)"
  fi
  echo "  Metrics JSON: $METRICS_FILE"
  echo "  Run log: $LOG_FILE"
  echo "  Output frames: $TMP_OUT"
  echo "  Kernel report: $KERNEL_REPORT_FILE"
} > "$REPORT_FILE"

{
  echo "CUDA VSR Kernel Stage Report"
  echo "============================"
  echo "Run: ${RUN_DIR##*/}"
  echo "Frames: $FRAMES"
  echo "Input size: ${IN_W}x${IN_H}"
  echo "Output size: ${OUT_W}x${OUT_H}"
  echo "Backend: $BACKEND"
  echo "Estimated kernel FPS: $FPS_EST"
  echo ""
  echo "Per-frame stage cost (sorted, ms and % of total):"
  {
    echo "load $LOAD_MS"
    echo "h2d $H2D_MS"
    echo "u8_to_f32 $U8_TO_F32_MS"
    echo "temporal $TEMPORAL_MS"
    echo "resize $RESIZE_MS"
    echo "sharpen $SHARPEN_MS"
    echo "f32_to_u8 $F32_TO_U8_MS"
    echo "d2h $D2H_MS"
    echo "write $WRITE_MS"
  } | sort -k2,2gr | awk -v total="$TOTAL_MS" '
    {
      pct = ((total + 0) > 0) ? (($2 / total) * 100.0) : 0.0;
      printf "  %-12s %10.4f ms  (%5.1f%%)\n", $1, $2, pct;
    }'
  echo ""
  echo "Top bottleneck stage: $BOTTLENECK_STAGE (${BOTTLENECK_MS} ms/frame)"
  echo "Kernel group total: $KERNELS_MS ms/frame (${KERNELS_PCT}%)"
  echo "Transfer total: $TRANSFER_MS ms/frame (${TRANSFER_PCT}%)"
  echo "IO total: $IO_MS ms/frame (${IO_PCT}%)"
  echo ""
  echo "For low-level kernel occupancy and memory details, run:"
  echo "  sbatch scripts/slurm_ncu.sh"
  echo "  sbatch scripts/slurm_nsys.sh"
} > "$KERNEL_REPORT_FILE"

echo ""
echo "Done."
if [ "$SKIP_VIDEO_ENCODE" = "1" ]; then
  echo "Output video:  [skipped]"
else
  echo "Output video:  $OUTPUT_FILE"
fi
echo "Output frames: $TMP_OUT"
echo "Metrics JSON:  $METRICS_FILE"
echo "Report:        $REPORT_FILE"
echo "Kernel report: $KERNEL_REPORT_FILE"
echo "Log:           $LOG_FILE"
