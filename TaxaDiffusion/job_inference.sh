#!/bin/bash
#SBATCH --job-name=taxa_diffusion

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=164gb


export MASTER_PORT=12340
export WORLD_SIZE2=14

echo "NODELIST="${SLURM_NODELIST}
echo "SLURM_NTASKS="${SLURM_NTASKS}
echo "SLURM_PROCID="${SLURM_PROCID}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module load cuda
module spider cuda
### init virtual environment if needed

conda activate taxa_diffusion

echo "Starting accelerate..."
srun python3 inference.py --config configs/taxa_diffusion.yaml --launcher slurm
