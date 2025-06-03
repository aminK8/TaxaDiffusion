#!/bin/bash
#SBATCH --job-name=taxa_diffusion
#SBATCH --time=6-22:40:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=18


export MASTER_PORT=12340
export WORLD_SIZE2=14


echo "NODELIST="${SLURM_NODELIST}
echo "SLURM_NTASKS="${SLURM_NTASKS}
echo "SLURM_PROCID="${SLURM_PROCID}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# module spider cuda
# module load cuda/12.3.0
module load miniconda3
conda info --envs
conda activate taxa_diffusion

echo "Starting accelerate..."
srun python3 train.py --config configs/taxa_diffusion.yaml --launcher slurm
