#!/bin/bash
#SBATCH -A debug
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p gilbreth-debug
#SBATCH --constraint 'A10|A100|A30'
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name="mt3-debug"
#SBATCH --error=./logs/debug/%x-%J-%u.err
#SBATCH --output=./logs/debug/%x-%J-%u.out

module --force purge
module load rcac
module load cuda/12.1.1 
module load anaconda 
module list

# run cav-mae pretraining, use smaller lr and batch size, fits smaller GPUs (4*12GB GPUs)
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_NSOCKS_PERTHREA=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

set -x
cd /home/chou150/code/mt3-pytorch-2
source /depot/yunglu/data/ben/venvs/MT3/bin/activate
# export TORCH_HOME=/home/chou150/code/cav-mae/pretrained_model


srun python inference.py