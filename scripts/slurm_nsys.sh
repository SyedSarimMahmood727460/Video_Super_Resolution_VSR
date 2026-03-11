#!/bin/bash
#SBATCH --account=project_2016196
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --mem=64G

module purge
module load gcc/11.2.0
module load cuda
module load ffmpeg

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"

make -C cuda_vsr clean all

RUN_DIR=/scratch/project_2016196/$USER/cuda_vsr_nsys
IN_FRAMES=$RUN_DIR/frames_in
OUT_FRAMES=$RUN_DIR/frames_out
mkdir -p "$IN_FRAMES" "$OUT_FRAMES"

ffmpeg -hide_banner -loglevel error -y \
  -i /scratch/project_2016196/$USER/input.mp4 \
  -vsync 0 -pix_fmt rgb24 \
  "$IN_FRAMES/frame_%06d.ppm"

srun nsys profile --stats=true --force-overwrite=true -o nsys_vsr \
  ./cuda_vsr/bin/cuda_vsr \
    --input_dir "$IN_FRAMES" \
    --output_dir "$OUT_FRAMES" \
    --backend cuda \
    --target_height 0 \
    --metrics "$RUN_DIR/metrics.json"
