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
module load pytorch

cd /scratch/project_2016196/$USER/Video-Super-Resolution-VSR-
source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python -m vsr.infer \
  --input /scratch/project_2016196/$USER/input.mp4 \
  --output /scratch/project_2016196/$USER/out.mp4 \
  --ckpt checkpoints/litevsr_scale2.pt \
  --scale 2 --window 5 \
  --precision amp --channels_last \
  --benchmark --warmup 50 --measure 300 \
  --save_metrics metrics.json
