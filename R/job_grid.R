# ${TRAIN_FILE}${GPU_PER_TRIAL}${NUM_SAMPLES}${N_FOLDS}${DATA_TYPES}${NAME_TAG}${SUBSET_TYPE}${STRATIFY}${BOTTLENECK}${FULL}${ENCODER_TRAIN}
require(utils)
require(data.table)
all_combos <- c(
  # "drug mut cnv exp",
  # "drug mut cnv exp prot",
  # "drug cnv exp prot",
  # "drug cnv prot",
  "drug exp",
  # "drug exp prot",
  "drug prot",
  "drug mut",
  "drug cnv")
# ${TRAIN_FILE} ${N_FOLDS} ${DATA_TYPES} ${NAME_TAG} ${SUBSET_TYPE} ${STRATIFY} ${FULL} ${ENCODER_TRAIN}

all_grids <- vector(mode = "list")
for (combo in all_combos) {
  ENCODER_TRAIN <- "1"
  if (grepl("cnv", combo)) {
    GPU_PER_TRIAL <- "1"
    NUM_SAMPLES <- "16"
  } else if (combo == "drug prot") {
    GPU_PER_TRIAL <- "0.2"
    NUM_SAMPLES <- "80"
  } else if (combo  == "drug exp" | combo == "drug exp prot") {
    GPU_PER_TRIAL <- "0.5"
    NUM_SAMPLES <- "32"
  }
  
  for (BOTTLENECK in c("0", "1")) {
    for (TRAIN_FILE in c("CTRP_AAC_MORGAN_512.hdf")) {
      for (SUBSET_TYPE in c("cell_line", "drug", "both")) {
        if (SUBSET_TYPE == "both") {
          N_FOLDS <- "5"
        } else {
          N_FOLDS <- "5"
        }
        for (FULL in c("0", "1")) {
          if (FULL == "1") {
            cur_pretrain <- "0"
            train_set_name <- gsub("\\_.+", "", TRAIN_FILE)
            full <- "FullModel"
            encoder <- "EncoderTrain"
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
            
            NAME_TAG <- paste("HyperOpt_DRP", train_set_name, full, encoder, "Split", split, bottleneck, pretrain, data_types, sep = "_")
            
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
              PRETRAIN = cur_pretrain
            )
            all_grids <- append(all_grids, list(cur_grid))
            
          } else {
            
            if (grepl("cnv", combo) | grepl("exp", combo)) {
              for (PRETRAIN in c("0", "1")) {
                cur_encoder_train <- "1"
                encoder <- "EncoderTrain"
                train_set_name <- gsub("\\_.+", "", TRAIN_FILE)
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
                
                NAME_TAG <- paste("HyperOpt_DRP", train_set_name, full, encoder, "Split", split, bottleneck, pretrain, data_types, sep = "_")
                
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
                  PRETRAIN = PRETRAIN
                )
                all_grids <- append(all_grids, list(cur_grid))
                
              }
            } else {
              cur_encoder_train <- "1"
              encoder <- "EncoderTrain"
              train_set_name <- gsub("\\_.+", "", TRAIN_FILE)
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
              
              NAME_TAG <- paste("HyperOpt_DRP", train_set_name, full, encoder, "Split", split, bottleneck, pretrain, data_types, sep = "_")
              
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
                PRETRAIN = cur_pretrain
              )
              all_grids <- append(all_grids, list(cur_grid))
            }
          }
        }
      }
    }
  }
}

all_param_combos <- rbindlist(all_grids)
table(all_param_combos$DATA_TYPES)
all_param_combos[PRETRAIN == 0]
all_param_combos[PRETRAIN == 1]
unique(all_param_combos[, .SD, .SDcols = !c("PRETRAIN")])
all_param_combos[PRETRAIN == 0 & FULL == 1]
all_param_combos[FULL == 1]
table(all_param_combos[FULL == 1]$DATA_TYPES)

fwrite(all_param_combos[PRETRAIN == 0 & FULL == 1], "DRP/slurm/drp_opt_grid.csv", col.names = F)
fwrite(all_param_combos[PRETRAIN == 0 & FULL == 1][1], "DRP/slurm/drp_opt_test_grid.csv", col.names = F)

colnames(all_param_combos)