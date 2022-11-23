# GNN_vs_Morgan_by_Murcko_Scaffold.R
require(data.table)
require(ggplot2)

# ==== Cross-validation per fold comparison ====
# Select per fold validation files
cur_cv_files <- list.files("Data/CV_Results/", recursive = T,
                           pattern = ".*final_validation.*", full.names = T)
length(cur_cv_files)
# Read all data
all_results <- vector(mode = "list", length = length(cur_cv_files))
for (i in 1:length(cur_cv_files)) {
  cur_res <- fread(cur_cv_files[i])
  data_types <- gsub(".+ResponseOnly_\\w*drug_(.+)_HyperOpt.+", "\\1", cur_cv_files[i])
  data_types <- toupper(data_types)
  merge_method <- gsub(".+MergeBy(\\w+)_.*RMSE.+", "\\1", cur_cv_files[i])
  loss_method <- gsub(".+_(.*)RMSE.+", "\\1RMSE", cur_cv_files[i])
  drug_type <- gsub(".+ResponseOnly_(\\w*)drug.+_HyperOpt.+", "\\1drug", cur_cv_files[i])
  drug_type <- toupper(drug_type)
  split_method <- gsub(".+Split_(\\w+)_NoBottleNeck.+", "\\1", cur_cv_files[i])
  cur_fold <- gsub(".+CV_Index_(\\d)_.+", "\\1", cur_cv_files[i])
  # data_types <- strsplit(data_types, "_")[[1]]
  # cur_res$epoch <- as.integer(epoch)
  cur_res$data_types <- data_types
  cur_res$merge_method <- merge_method
  cur_res$loss_type <- loss_method
  cur_res$drug_type <- drug_type
  cur_res$split_method <- split_method
  cur_res$fold <- cur_fold
  
  all_results[[i]] <- cur_res
}
all_results <- rbindlist(all_results)
all_results[, RMSELoss := abs(target - predicted), by = .I]

all_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold")]
all_results$V1 <- NULL

# ==== Generalization to new drug scaffolds (Murcko) ====
setDTthreads(8)
ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")
length(unique(ctrp$scaffold))  # 447 unique scaffolds
length(unique(ctrp$cpd_name))  # 495 unique drugs

# Check for drug leakage among folds for the DRUG SPLITTING case
# all_drug_split_results <- all_results[split_method == "CELL_LINE"]
all_drug_split_results <- all_results[split_method == "BOTH"]
all_drug_split_results[, fold_0_1_overlaps := sum(.SD[fold == 0]$cpd_name %in% .SD[fold == 1]$cpd_name), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method")]

my_gnn_sub <- all_drug_split_results[data_types == "EXP" & merge_method == "Concat" & loss_type == "RMSE" & drug_type == "GNNDRUG"]
sum(unique(my_gnn_sub[fold == 0]$cpd_name) %in% unique(my_gnn_sub[fold == 1]$cpd_name))
sum(unique(my_gnn_sub[fold == 1]$cpd_name) %in% unique(my_gnn_sub[fold == 2]$cpd_name))
sum(unique(my_gnn_sub[fold == 3]$cpd_name) %in% unique(my_gnn_sub[fold == 4]$cpd_name))
sum(unique(my_gnn_sub[fold == 0]$cell_name) %in% unique(my_gnn_sub[fold == 1]$cell_name))
sum(unique(my_gnn_sub[fold == 1]$cell_name) %in% unique(my_gnn_sub[fold == 2]$cell_name))
sum(unique(my_gnn_sub[fold == 2]$cell_name) %in% unique(my_gnn_sub[fold == 3]$cell_name))
unique(my_gnn_sub[fold == 2 & cell_name %in% unique(my_gnn_sub[fold == 3]$cell_name)]$cell_name)
unique(my_gnn_sub[fold == 0 & cell_name %in% unique(my_gnn_sub[fold == 1]$cell_name)]$cell_name)

sum(unique(my_gnn_sub[fold == 1]$cpd_name) %in% unique(my_gnn_sub[fold == 2]$cpd_name))
