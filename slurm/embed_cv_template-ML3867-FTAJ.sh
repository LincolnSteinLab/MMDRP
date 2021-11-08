#!/bin/bash
# shellcheck disable=SC2206
# Slurm does not support using variables in the #SBATCH section, so we need to set the job name in the submit command.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0:30:00


module purge
module load MistEnv/2020a
module load cuda/10.2.89 gcc/8.4.0 anaconda3/2019.10 cudnn/7.6.5.32
source activate drp1
cd ~/.conda/envs/drp1/

export CUBLAS_WORKSPACE_CONFIG=:4096:2


python3 DRP/src/ccle_embed_hypopt_pytorch.py --machine mist --train 1 --max_num_epochs 100 --name_tag optimal_autoencoders --loss_type rmse --data_type ${DATA_TYPE}