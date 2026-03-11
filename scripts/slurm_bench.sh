#!/bin/bash
#SBATCH --account=project_2016196
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G

module purge
module load gcc/11.2.0
module load cuda
module load ffmpeg

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export INPUT_FILE=/scratch/project_2016196/$USER/input.mp4
export OUTPUT_FILE=/scratch/project_2016196/$USER/out_cuda_vsr.mp4
export TARGET_HEIGHT=0

srun bash ./main.sh
