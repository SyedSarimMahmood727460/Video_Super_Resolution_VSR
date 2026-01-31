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
module load pytorch

cd /scratch/project_2016196/$USER/Video-Super-Resolution-VSR-
source .venv/bin/activate

srun nsys profile --stats=true --force-overwrite=true -o nsys_vsr \
  python -m vsr.infer \
    --input /scratch/project_2016196/$USER/input.mp4 \
    --output /scratch/project_2016196/$USER/out.mp4 \
    --ckpt checkpoints/litevsr_scale2.pt \
    --scale 2 --window 5 \
    --precision amp --channels_last \
    --benchmark --warmup 10 --measure 50 \
    --save_metrics metrics.json
