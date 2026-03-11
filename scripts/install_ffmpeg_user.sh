#!/bin/bash
set -euo pipefail

INSTALL_DIR="${1:-/scratch/project_2016196/$USER/bin}"
FFMPEG_STATIC_URL="${FFMPEG_STATIC_URL:-https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz}"

mkdir -p "$INSTALL_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

ARCHIVE="$TMP_DIR/ffmpeg-static.tar.xz"

echo "Installing ffmpeg to: $INSTALL_DIR"
echo "Source URL: $FFMPEG_STATIC_URL"

if command -v curl >/dev/null 2>&1; then
  curl -fL "$FFMPEG_STATIC_URL" -o "$ARCHIVE"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$ARCHIVE" "$FFMPEG_STATIC_URL"
else
  echo "ERROR: neither curl nor wget is available."
  exit 1
fi

tar -xf "$ARCHIVE" -C "$TMP_DIR"

SRC_DIR="$(find "$TMP_DIR" -maxdepth 1 -type d -name 'ffmpeg*-static' | head -n1)"
if [ -z "$SRC_DIR" ]; then
  echo "ERROR: could not locate extracted ffmpeg static directory."
  exit 1
fi

install -m 755 "$SRC_DIR/ffmpeg" "$INSTALL_DIR/ffmpeg"
install -m 755 "$SRC_DIR/ffprobe" "$INSTALL_DIR/ffprobe"

echo "Installed:"
echo "  $INSTALL_DIR/ffmpeg"
echo "  $INSTALL_DIR/ffprobe"
echo ""
echo "Verify:"
echo "  $INSTALL_DIR/ffmpeg -version"
echo "  $INSTALL_DIR/ffprobe -version"
echo ""
echo "Run on Mahti with:"
echo "  sbatch --export=ALL,PROJECT_DIR=\$PWD,FFMPEG_DIR=$INSTALL_DIR scripts/slurm_run_mahti.sh"
