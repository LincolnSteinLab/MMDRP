#!/bin/bash
# shellcheck disable=SC2206
# Slurm does not support using variables in the #SBATCH section, so we need to set the job name in the submit command.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

#SBATCH --time=24:00:00


module purge
module load MistEnv/2020a
module load cuda/10.2.89 gcc/8.4.0 anaconda3/2019.10 cudnn/7.6.5.32
source activate drp1
cd ~/.conda/envs/drp1/

export CUBLAS_WORKSPACE_CONFIG=:4096:2
# # __doc_head_address_start__

# # Getting the node names
# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# nodes_array=($nodes)

# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# # if we detect a space character in the head node IP, we'll
# # convert it to an ipv4 address. This step is optional.
# if [[ "$head_node_ip" == *" "* ]]; then
# IFS=' ' read -ra ADDR <<<"$head_node_ip"
# if [[ ${#ADDR[0]} -gt 16 ]]; then
#   head_node_ip=${ADDR[1]}
# else
#   head_node_ip=${ADDR[0]}
# fi
# echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
# fi
# # __doc_head_address_end__

# port=6379
# ip_head=$head_node_ip:$port
# export ip_head
# echo "IP Head: $ip_head"

# echo "Starting HEAD at $head_node"
# srun --nodes=1 --ntasks=1 -w "$head_node" \
#     ray start --head --node-ip-address="$head_node_ip" --port=$port --include-dashboard False\
#     --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &

# # optional, though may be useful in certain versions of Ray < 1.0.
# sleep 10

# # number of nodes other than the head node
# worker_num=$((SLURM_JOB_NUM_NODES - 1))

# for ((i = 1; i <= worker_num; i++)); do
#     node_i=${nodes_array[$i]}
#     echo "Starting WORKER $i at $node_i"
#     srun --nodes=1 --ntasks=1 -w "$node_i" \
#         ray start --address "$ip_head" \
#         --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
#     sleep 5
# done

# Finally run the script

# python3 -u DRP/src/drp_full_model.py --machine mist --train_file ${TRAIN_FILE} --optimize 1 --max_num_epochs 50 --init_cpus 32 --init_gpus 1 --gpus_per_trial ${GPU_PER_TRIAL} --num_samples ${NUM_SAMPLES} --n_folds ${N_FOLDS}  --data_types ${DATA_TYPES} --name_tag ${NAME_TAG} --cv_subset_type ${SUBSET_TYPE} --stratify ${STRATIFY} --bottleneck ${BOTTLENECK} --full ${FULL} --encoder_train ${ENCODER_TRAIN} --pretrain ${PRETRAIN} --merge_method ${MERGE_METHOD} --loss_type ${LOSS_TYPE} --one_hot_drugs ${ONE_HOT_DRUGS}
# python3 -u DRP/src/drp_full_model.py --machine mist --train_file CTRP_AAC_SMILES.txt --optimize 1 --max_num_epochs 100 --init_cpus 128 --init_gpus 4 --gpus_per_trial 0.25 --num_samples 100 --n_folds 5 --data_types gnndrug exp --name_tag HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GnnDrugs_gnndrug_exp_test --cv_subset_type both --stratify 1 --bottleneck 0 --full 0 --pretrain 0 --merge_method lmf --loss_type weighted_rmse
python3 -u DRP/src/drp_full_model.py --machine mist --train_file ${TRAIN_FILE} --optimize 1 --max_num_epochs 100 --init_cpus 40 --init_gpus 1 --gpus_per_trial ${GPU_PER_TRIAL} --num_samples ${NUM_SAMPLES} --n_folds ${N_FOLDS} --data_types ${DATA_TYPES} --name_tag ${NAME_TAG} --cv_subset_type ${SUBSET_TYPE} --stratify ${STRATIFY} --bottleneck ${BOTTLENECK} --full ${FULL} --pretrain ${PRETRAIN} --merge_method ${MERGE_METHOD} --loss_type ${LOSS_TYPE}  --omic_standardize --resume
# python3 -u DRP/src/drp_full_model.py --machine mist --train_file CTRP_AAC_SMILES.txt --optimize 1 --max_num_epochs 50 --init_cpus 32 --init_gpus 1 --gpus_per_trial 1 --num_samples 1 --n_folds 5 --data_types gnndrug exp --name_tag HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GnnDrugs_gnndrug_exp_test --cv_subset_type both --stratify 1 --bottleneck 0 --full 0 --pretrain 0 --merge_method lmf --loss_type weighted_rmse

# python3 -u DRP/src/drp_full_model.py --machine mist --train_file CTRP_AAC_MORGAN_1024.hdf --optimize 1 --max_num_epochs 30 --init_cpus 32 --init_gpus 1 --gpus_per_trial 0.2 --num_samples 20 --n_folds 5  --data_types drug rppa --name_tag HyperOpt_DRP_CTRP_1024_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_MorganDrugs_drug_rppa --cv_subset_type both --stratify 1 --bottleneck 0 --full 0 --encoder_train 1 --pretrain 0 --merge_method lmf --global_code_size 512 --loss_type weighted_rmse --one_hot_drugs 0
# python3 -u DRP/src/drp_full_model.py --machine mist --train_file CTRP_AAC_SMILES.txt --optimize 1 --max_num_epochs 30 --init_cpus 32 --init_gpus 1 --gpus_per_trial 1 --num_samples 1 --n_folds 5 --data_types gnndrug exp --name_tag HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GnnDrugs_gnndrug_exp --cv_subset_type both --stratify 1 --bottleneck 0 --full 0 --encoder_train 1 --pretrain 0 --merge_method lmf --loss_type weighted_rmse