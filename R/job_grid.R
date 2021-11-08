# ${TRAIN_FILE}${GPU_PER_TRIAL}${NUM_SAMPLES}${N_FOLDS}${DATA_TYPES}${NAME_TAG}${SUBSET_TYPE}${STRATIFY}${BOTTLENECK}${FULL}${ENCODER_TRAIN}
require(utils)
require(data.table)
all_combos <- c(
  # "gnndrug mut exp",
  # "gnndrug cnv exp",
  # "gnndrug exp prot",
  # "gnndrug exp rppa",
  # "gnndrug exp hist",
  # "gnndrug exp metab",
  # "gnndrug exp mirna",
  # "gnndrug prot rppa",
  # "gnndrug cnv prot",
  # "gnndrug mut cnv",
  # "gnndrug mirna metab",
  # "gnndrug metab hist",
  # "gnndrug metab rppa",
  # "gnndrug mut exp cnv",
  # "gnndrug mut cnv exp",
  # "gnndrug mut cnv exp prot",
  # "gnndrug cnv exp prot",
  # "gnndrug exp rppa prot",
  # "gnndrug exp rppa hist prot",
  # "gnndrug mirna metab hist rppa",
  # "gnndrug mut cnv exp prot mirna metab hist rppa",
  # "gnndrug cnv exp prot mirna metab hist rppa"
  "drug mut exp",
  "drug cnv exp",
  "drug exp prot",
  "drug exp rppa",
  "drug exp hist",
  "drug exp metab",
  "drug exp mirna",
  "drug prot rppa",
  "drug cnv prot",
  "drug mut cnv",
  "drug mirna metab",
  "drug metab hist",
  "drug metab rppa",
  "drug mut exp cnv",
  "drug mut cnv exp",
  "drug mut cnv exp prot",
  "drug cnv exp prot",
  "drug exp rppa prot",
  "drug exp rppa hist prot",
  "drug mirna metab hist rppa",
  "drug mut cnv exp prot mirna metab hist rppa",
  "drug cnv exp prot mirna metab hist rppa"
  
  # "gnndrug exp",
  # "gnndrug mut",
  # "gnndrug cnv",
  # "gnndrug prot",
  # "gnndrug mirna",
  # "gnndrug metab",
  # "gnndrug hist",
  # "gnndrug rppa"
  
  # "drug exp",
  # "drug mut",
  # "drug cnv",
  # "drug prot",
  # "drug mirna",
  # "drug metab",
  # "drug hist",
  # "drug rppa"
  
  )
# ${TRAIN_FILE} ${N_FOLDS} ${DATA_TYPES} ${NAME_TAG} ${SUBSET_TYPE} ${STRATIFY} ${FULL} ${ENCODER_TRAIN}

all_grids <- vector(mode = "list")
for (combo in all_combos) {
  ENCODER_TRAIN <- "1"
  NUM_SAMPLES <- "40"
  if (grepl("cnv", combo)) {
    GPU_PER_TRIAL <- "1"
  } else if (combo == "gnndrug prot" | combo == "drug prot") {
    GPU_PER_TRIAL <- "0.5"
    # NUM_SAMPLES <- "32"
  } else if (combo  == "gnndrug exp" | combo == "gnndrug exp prot" | combo  == "drug exp" | combo == "drug exp prot") {
    GPU_PER_TRIAL <- "0.5"
    # NUM_SAMPLES <- "8"
  } else if (combo == "gnndrug mirna" | combo == "gnndrug metab" | combo == "gnndrug hist" | combo == "gnndrug rppa" |
             combo == "drug mirna" | combo == "drug metab" | combo == "drug hist" | combo == "drug rppa") {
    GPU_PER_TRIAL <- "0.2"
    # NUM_SAMPLES <- "40"
  } else {
    GPU_PER_TRIAL <- "1"
    # NUM_SAMPLES <- "8"
  }
  # 
  LOSS_TYPE = "rmse"
  loss_type_name = "RMSELoss"
  # LOSS_TYPE = "weighted_rmse"
  # loss_type_name = "WeightedRMSELoss"
  
  for (ONE_HOT_DRUGS in c("0", "1")) {
    if (ONE_HOT_DRUGS == "1") {
      one_hot_drugs_name = "OneHotDrugs"
    } else {
      if (grepl("gnndrug", combo) == TRUE) {
        one_hot_drugs_name = "GNNDrugs"
      } else {
        one_hot_drugs_name = "MorganDrugs"
      }
    }
    for (MERGE_METHOD in c("sum", "concat", "lmf")) {
      if (MERGE_METHOD == "sum") {
        merge_method_name = "MergeBySum"
      } else if (MERGE_METHOD == "concat") {
        merge_method_name = "MergeByConcat"
      } else if (MERGE_METHOD == "lmf") {
        merge_method_name = "MergeByLMF"
      }
      for (BOTTLENECK in c("0", "1")) {
        for (TRAIN_FILE in c("CTRP_AAC_MORGAN_1024.hdf")) {
        # for (TRAIN_FILE in c("CTRP_AAC_SMILES.txt", "GDSC1_AAC_SMILES.txt", "GDSC2_AAC_SMILES.txt")) {
        # for (TRAIN_FILE in c("CTRP_AAC_MORGAN_1024.hdf", "GDSC1_AAC_MORGAN_1024.hdf", "GDSC2_AAC_MORGAN_1024.hdf")) {
        # for (TRAIN_FILE in c("CTRP_AAC_SMILES.txt")) {
          train_set_name <- "CTRP"
          for (SUBSET_TYPE in c("cell_line", "drug", "both")) {
            if (SUBSET_TYPE == "both") {
              N_FOLDS <- "5"
            } else {
              N_FOLDS <- "5"
            }
            for (FULL in c("0", "1")) {
              if (FULL == "1") {
                cur_pretrain <- "0"
                # train_set_name <- gsub("\\_.+", "", TRAIN_FILE)
                full <- "FullModel"
                encoder <- "EncoderTrain"
                split <- toupper(SUBSET_TYPE)
                data_types <- gsub(" ", "_", combo)
                if (BOTTLENECK == "0") {
                  bottleneck <- "NoBottleNeck"
                } else {
                  bottleneck <- "WithBottleNeck"
                }
                if (cur_pretrain == "0") {
                  pretrain <- "NoTCGAPretrain"
                } else {
                  pretrain <- "WithTCGAPretrain"
                }
                
                NAME_TAG <- paste("HyperOpt_DRP", train_set_name, full, encoder, "Split", split, bottleneck, pretrain, merge_method_name, loss_type_name, one_hot_drugs_name, data_types, sep = "_")
                
                cur_grid <- data.table(
                  TRAIN_FILE = TRAIN_FILE,
                  GPU_PER_TRIAL = GPU_PER_TRIAL,
                  NUM_SAMPLES = NUM_SAMPLES,
                  N_FOLDS = N_FOLDS,
                  DATA_TYPES = combo,
                  NAME_TAG = NAME_TAG,
                  SUBSET_TYPE = SUBSET_TYPE,
                  STRATIFY = "1",
                  BOTTLENECK = BOTTLENECK,
                  FULL = FULL,
                  ENCODER_TRAIN = ENCODER_TRAIN,
                  PRETRAIN = cur_pretrain,
                  MERGE_METHOD = MERGE_METHOD,
                  LOSS_TYPE = LOSS_TYPE,
                  ONE_HOT_DRUGS = ONE_HOT_DRUGS
                )
                all_grids <- append(all_grids, list(cur_grid))
                
              } else {
                
                if (grepl("cnv", combo) | grepl("exp", combo)) {
                  for (PRETRAIN in c("0", "1")) {
                    cur_encoder_train <- "1"
                    encoder <- "EncoderTrain"
                    # train_set_name <- gsub("\\_.+", "", TRAIN_FILE)
                    full <- "ResponseOnly"
                    split <- toupper(SUBSET_TYPE)
                    data_types <- gsub(" ", "_", combo)
                    if (BOTTLENECK == "0") {
                      bottleneck <- "NoBottleNeck"
                    } else {
                      bottleneck <- "WithBottleNeck"
                    }
                    if (PRETRAIN == "0") {
                      pretrain <- "NoTCGAPretrain"
                    } else {
                      pretrain <- "WithTCGAPretrain"
                    }
                    
                    NAME_TAG <- paste("HyperOpt_DRP", train_set_name, full, encoder, "Split", split, bottleneck, pretrain, merge_method_name, loss_type_name, one_hot_drugs_name, data_types, sep = "_")
                    
                    cur_grid <- data.table(
                      TRAIN_FILE = TRAIN_FILE,
                      GPU_PER_TRIAL = GPU_PER_TRIAL,
                      NUM_SAMPLES = NUM_SAMPLES,
                      N_FOLDS = N_FOLDS,
                      DATA_TYPES = combo,
                      NAME_TAG = NAME_TAG,
                      SUBSET_TYPE = SUBSET_TYPE,
                      STRATIFY = "1",
                      BOTTLENECK = BOTTLENECK,
                      FULL = FULL,
                      ENCODER_TRAIN = cur_encoder_train,
                      PRETRAIN = PRETRAIN,
                      MERGE_METHOD = MERGE_METHOD,
                      LOSS_TYPE = LOSS_TYPE,
                      ONE_HOT_DRUGS = ONE_HOT_DRUGS
                    )
                    all_grids <- append(all_grids, list(cur_grid))
                    
                  }
                } else {
                  cur_encoder_train <- "1"
                  encoder <- "EncoderTrain"
                  # train_set_name <- gsub("\\_.+", "", TRAIN_FILE)
                  full <- "ResponseOnly"
                  split <- toupper(SUBSET_TYPE)
                  data_types <- gsub(" ", "_", combo)
                  if (BOTTLENECK == "0") {
                    bottleneck <- "NoBottleNeck"
                  } else {
                    bottleneck <- "WithBottleNeck"
                  }
                  cur_pretrain <- "0"
                  pretrain <- "NoTCGAPretrain"
                  
                  NAME_TAG <- paste("HyperOpt_DRP", train_set_name, full, encoder, "Split", split, bottleneck, pretrain, merge_method_name, loss_type_name, one_hot_drugs_name, data_types, sep = "_")
                  
                  cur_grid <- data.table(
                    TRAIN_FILE = TRAIN_FILE,
                    GPU_PER_TRIAL = GPU_PER_TRIAL,
                    NUM_SAMPLES = NUM_SAMPLES,
                    N_FOLDS = N_FOLDS,
                    DATA_TYPES = combo,
                    NAME_TAG = NAME_TAG,
                    SUBSET_TYPE = SUBSET_TYPE,
                    STRATIFY = "1",
                    BOTTLENECK = BOTTLENECK,
                    FULL = FULL,
                    ENCODER_TRAIN = cur_encoder_train,
                    PRETRAIN = cur_pretrain,
                    MERGE_METHOD = MERGE_METHOD,
                    LOSS_TYPE = LOSS_TYPE,
                    ONE_HOT_DRUGS = ONE_HOT_DRUGS
                  )
                  all_grids <- append(all_grids, list(cur_grid))
                }
              }
            }
          }
        }
      }
    }
  }
}
all_param_combos <- rbindlist(all_grids)

# ==== (GDSC) Bi-modal with GNN + LMF ====
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & SUBSET_TYPE == "both" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/drp_opt_gdsc_grid.csv", col.names = F)

# ==== Bi-modal Baseline (Morgan + Concat + RMSE) ====
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & SUBSET_TYPE == "both" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/grids/drp_opt_baseline_grid.csv", col.names = F)
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & (SUBSET_TYPE == "cell_line" | SUBSET_TYPE == "drug") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/grids/drp_opt_baseline_cell_line_drug_grid.csv", col.names = F)

# ==== Bi-modal Baseline + LDS (Morgan + Concat + WeightedRMSE) ====
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & SUBSET_TYPE == "both" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/grids/drp_opt_baseline_grid.csv", col.names = F)
# combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & (SUBSET_TYPE == "cell_line" | SUBSET_TYPE == "drug") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/grids/drp_opt_baseline_with_lds_cell_line_drug_grid.csv", col.names = F)

# ==== Bi-modal Baseline + LMF (Morgan + LMF + RMSE) ====
# combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & SUBSET_TYPE == "both" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
# combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
# fwrite(combos, "DRP/slurm/grids/drp_opt_baseline_grid.csv", col.names = F)
# combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & (SUBSET_TYPE == "cell_line" | SUBSET_TYPE == "drug") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/grids/drp_opt_baseline_with_lmf_cell_line_drug_grid.csv", col.names = F)

# ==== Bi-modal Baseline + GNN (GNN + concat + RMSE) ====
# combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & SUBSET_TYPE == "both" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
# combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
# fwrite(combos, "DRP/slurm/grids/drp_opt_baseline_grid.csv", col.names = F)
# combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & (SUBSET_TYPE == "cell_line" | SUBSET_TYPE == "drug") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/grids/drp_opt_baseline_with_gnn_cell_line_drug_grid.csv", col.names = F)


# ==== Bi-modal with GNN + LMF but no LDS ====
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & SUBSET_TYPE == "both" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/drp_opt_noLDS_grid.csv", col.names = F)
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & (SUBSET_TYPE == "cell_line" | SUBSET_TYPE == "drug") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/drp_opt_noLDS_cell_line_drug_grid.csv", col.names = F)

# ==== Bi-modal with Morgan Drugs (LMF + LDS) ====
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & SUBSET_TYPE == "both" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/drp_opt_morgan_grid.csv", col.names = F)
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & (SUBSET_TYPE == "cell_line" | SUBSET_TYPE == "drug") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/drp_opt_morgan_cell_line_drug_grid.csv", col.names = F)

# ==== Bi-modal GNN + Concatenation (non-LMF) ====
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & SUBSET_TYPE == "both" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/drp_opt_concat_grid.csv", col.names = F)
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & (SUBSET_TYPE == "cell_line" | SUBSET_TYPE == "drug") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/drp_opt_concat_cell_line_drug_grid.csv", col.names = F)
# combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & SUBSET_TYPE == "drug" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
# combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
# fwrite(combos, "DRP/slurm/drp_opt_concat_drug_grid.csv", col.names = F)

# ==== Bi-modal GNN + Sum (non-LMF) ====
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "sum") & SUBSET_TYPE == "both" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/drp_opt_sum_grid.csv", col.names = F)
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "sum") & (SUBSET_TYPE == "cell_line" | SUBSET_TYPE == "drug") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/grids/drp_opt_sum_cell_line_drug_grid.csv", col.names = F)

# ==== Bi-modal GNN + LMF + LDS (trifecta) ====
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & SUBSET_TYPE == "both" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/grids/drp_opt_bi_lmf_lds_grid.csv", col.names = F)
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & (SUBSET_TYPE == "cell_line" | SUBSET_TYPE == "drug") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
# combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/grids/drp_opt_bi_lmf_lds_cell_line_drug_grid.csv", col.names = F)

# ==== Bi-modal MORGAN + LMF + LDS But only cell or drug splitting ====
# combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & SUBSET_TYPE == "cell_line" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
# combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
# fwrite(combos, "DRP/slurm/drp_opt_bi_lmf_lds_cell_line_grid.csv", col.names = F)
# combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & SUBSET_TYPE == "drug" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
# combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
# fwrite(combos, "DRP/slurm/drp_opt_bi_lmf_lds_drug_grid.csv", col.names = F)


# ==== Multi-modal GNN + LMF + LDS (trifecta) ====
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/grids/drp_opt_multimodal_trifecta_grid.csv", col.names = F)

# ==== Multi-modal Baseline ====
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "concat") & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/grids/drp_opt_multimodal_baseline_grid.csv", col.names = F)


table(all_param_combos$DATA_TYPES)
all_param_combos[PRETRAIN == 0]
all_param_combos[PRETRAIN == 1]
unique(all_param_combos[, .SD, .SDcols = !c("PRETRAIN")])
all_param_combos[PRETRAIN == 0 & FULL == 1]
all_param_combos[PRETRAIN == 0 & FULL == 1]
all_param_combos[FULL == 1]
table(all_param_combos[FULL == 1]$DATA_TYPES)
table(all_param_combos[FULL == 1 & PRETRAIN == 0]$DATA_TYPES)
all_param_combos[FULL == 1 & PRETRAIN == 0 & MERGE_METHOD == "concat" & SUBSET_TYPE == "drug"]
all_param_combos[FULL == 0 & PRETRAIN == 0 & MERGE_METHOD == "concat" & SUBSET_TYPE == "drug"]

# 2 data types including exp
combos <- all_param_combos[PRETRAIN == 0 & MERGE_METHOD == "sum" & SUBSET_TYPE == "drug" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
fwrite(combos, "DRP/slurm/drp_opt_grid.csv", col.names = F)
# unimodal + bimodal with exp + lmf + gnndrug
combos <- all_param_combos[PRETRAIN == 0 & (MERGE_METHOD == "lmf") & SUBSET_TYPE == "both" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 0]
combos <- combos[, !c("ENCODER_TRAIN", "ONE_HOT_DRUGS")]
fwrite(combos, "DRP/slurm/drp_opt_grid.csv", col.names = F)
# fwrite(combos, "DRP/slurm/drp_opt_grid_sub.csv", col.names = F)

combos_1 <- all_param_combos[PRETRAIN == 0 & MERGE_METHOD == "sum" & SUBSET_TYPE == "drug" & BOTTLENECK == 0 & FULL == 1]
combos_2 <- all_param_combos[PRETRAIN == 0 & MERGE_METHOD == "sum" & SUBSET_TYPE == "drug" & BOTTLENECK == 0 & FULL == 0 & DATA_TYPES == "drug cnv"]
combos_3 <- all_param_combos[PRETRAIN == 0 & MERGE_METHOD == "sum" & SUBSET_TYPE == "drug" & BOTTLENECK == 0 & FULL == 0 & ONE_HOT_DRUGS == 1]
fwrite(unique(rbindlist(list(combos_1, combos_2, combos_3))), "DRP/slurm/drp_opt_grid.csv", col.names = F)


fwrite(all_param_combos[PRETRAIN == 0 & MERGE_METHOD == "sum" & SUBSET_TYPE == "drug" & BOTTLENECK == 0], "DRP/slurm/drp_opt_grid.csv", col.names = F)
fwrite(all_param_combos[PRETRAIN == 0 & MERGE_METHOD == "sum" & SUBSET_TYPE == "both" & BOTTLENECK == 0 & ONE_HOT_DRUGS == 0], "DRP/slurm/drp_opt_grid.csv", col.names = F)
fwrite(all_param_combos[PRETRAIN == 0 & FULL == 0 & MERGE_METHOD == "concat" & SUBSET_TYPE == "drug" & ONE_HOT_DRUGS == 0], "DRP/slurm/drp_opt_grid.csv", col.names = F)

fwrite(all_param_combos[PRETRAIN == 0 & FULL == 1], "DRP/slurm/drp_opt_extra_grid.csv", col.names = F)
fwrite(all_param_combos[PRETRAIN == 0 & FULL == 1][1], "DRP/slurm/drp_opt_test_grid.csv", col.names = F)
fwrite(all_param_combos[PRETRAIN == 0 & FULL == 1], "DRP/slurm/drp_opt_drug_grid.csv", col.names = F)

colnames(all_param_combos)

# "TRAIN_FILE"    "GPU_PER_TRIAL" "NUM_SAMPLES"   "N_FOLDS"       "DATA_TYPES"    "NAME_TAG"      "SUBSET_TYPE"   "STRATIFY"      "BOTTLENECK"   
# "FULL"          "ENCODER_TRAIN" "PRETRAIN" "MERGE_METHOD"  "LOSS_TYPE"  "ONE_HOT_DRUGS"
# ==== CV Grid ====
require(data.table)
cur_opt_grid <- fread("DRP/slurm/drp_opt_grid.csv")
# ${TRAIN_FILE} ${N_FOLDS}   ${DATA_TYPES}  ${NAME_TAG}  ${SUBSET_TYPE}  ${STRATIFY}  ${FULL} ${ENCODER_TRAIN}
cur_cv_grid <- unique(cur_opt_grid[, c(1, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15)])
fwrite(cur_cv_grid, "DRP/slurm/drp_cv_grid.csv", col.names = F)

cur_cv_grid[V6 %like% ".*ResponseOnly.*"]
fwrite(cur_cv_grid[V6 %like% ".*ResponseOnly.*" & V5 != "drug cnv" & V15 == 0], "DRP/slurm/drp_cv_grid.csv", col.names = F)
fwrite(cur_cv_grid[!(V6  %like% ".*ResponseOnly.*" & V5 == "drug cnv")], "DRP/slurm/drp_cv_grid.csv", col.names = F)
fwrite(cur_cv_grid[!(V6  %like% ".*ResponseOnly.*" & V5 == "drug cnv") & !(V6 %like% "OneHotDrugs")], "DRP/slurm/drp_infer_grid.csv", col.names = F)

cur_cv_grid <- fread("DRP/slurm/drp_cv_grid.csv")
cur_cv_grid$V1 <- gsub(pattern = "CTRP", replacement = "GDSC2", cur_cv_grid$V1)
cur_cv_grid$V4 <- gsub(pattern = "CTRP", replacement = "GDSC2", cur_cv_grid$V4)



fwrite(cur_cv_grid, "DRP/slurm/drp_cv_grid.csv", col.names = F)
