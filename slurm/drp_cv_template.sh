#!/bin/bash
# shellcheck disable=SC2206
# Slurm does not support using variables in the #SBATCH section, so we need to set the job name in the submit command.
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=8
#SBATCH --time=12:00:00

module load cuda/10.2.89 gcc/8.4.0 anaconda3/2019.10 cudnn/7.6.5.32
source activate drp1
cd ~/.conda/envs/drp1/

# python3 DRP/src/drp_full_model.py --machine mist --train_file "CTRP_AAC_MORGAN_512.hdf" --optimize 0 --train 1 --max_num_epochs 100 --n_folds 10  --data_types drug exp --name_tag "CTRP_Full_Split_Cells" --cv_subset_type 'cell_line' --stratify 1 --full 1 --encoder_train 1
python3 DRP/src/drp_full_model.py --machine mist --train_file ${TRAIN_FILE} --optimize 0 --train 1 --max_num_epochs 100 --n_folds ${N_FOLDS}  --data_types ${DATA_TYPES} --name_tag ${NAME_TAG} --cv_subset_type ${SUBSET_TYPE} --stratify ${STRATIFY} --full ${FULL} --encoder_train ${ENCODER_TRAIN}