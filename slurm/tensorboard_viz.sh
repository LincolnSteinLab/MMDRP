#!/bin/bash
# shellcheck disable=SC2206
# Slurm does not support using variables in the #SBATCH section, so we need to set the job name in the submit command.
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00

module purge
module load MistEnv/2020a
module load cuda/10.2.89 gcc/8.4.0 anaconda3/2019.10 cudnn/7.6.5.32
source activate drp1
cd ~/.conda/envs/drp1/

export CUBLAS_WORKSPACE_CONFIG=:4096:2

python3 -u DRP/src/tensorboard_visualization.py --machine mist --train_file CTRP_AAC_SMILES.txt --optimize 0 --max_num_epochs 100 --train --train_only --n_folds 5 --data_types gnndrug exp --name_tag HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GnnDrugs_gnndrug_exp_test --cv_subset_type both --stratify 1 --bottleneck 0 --full 0 --pretrain 0 --merge_method lmf --loss_type weighted_rmse --omic_standardize