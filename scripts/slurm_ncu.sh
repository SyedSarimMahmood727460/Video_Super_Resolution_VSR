#!/bin/bash
#SBATCH --account=project_2016196
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --mem=64G

module purge
module load pytorch

cd /scratch/project_2016196/$USER/Video-Super-Resolution-VSR-
source .venv/bin/activate

srun ncu --set full --force-overwrite -o ncu_vsr \
  python -m vsr.infer \
    --input /scratch/project_2016196/$USER/input.mp4 \
    --output /scratch/project_2016196/$USER/out.mp4 \
    --ckpt checkpoints/litevsr_scale2.pt \
    --scale 2 --window 5 \
    --precision amp --channels_last \
    --benchmark --warmup 5 --measure 10
