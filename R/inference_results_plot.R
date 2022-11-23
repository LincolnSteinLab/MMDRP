# inference_results_plot.R

require(data.table)
require(ggplot2)
options(scipen = 3)
rsq <- function (x, y) cor(x, y) ^ 2

# TEMP: Train on CTRP, Test on GDSC using respective omic data (exp) ====
ctrp_gnn_exp <- fread("Data/CV_Results//HyperOpt_DRP_ResponseOnly_gnndrug_exp_HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GnnDrugs_gnndrug_exp/CTRP_AAC_SMILES_inference_results.csv")
gdsc1_gnn_exp <- fread("Data/CV_Results//HyperOpt_DRP_ResponseOnly_gnndrug_exp_HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GnnDrugs_gnndrug_exp/GDSC1_AAC_SMILES_inference_results.csv")
gdsc2_gnn_exp <- fread("Data/CV_Results//HyperOpt_DRP_ResponseOnly_gnndrug_exp_HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GnnDrugs_gnndrug_exp/GDSC2_AAC_SMILES_inference_results.csv")

rsq(ctrp_gnn_exp$target, ctrp_gnn_exp$predicted)  # 0.833
rsq(gdsc1_gnn_exp$target, gdsc1_gnn_exp$predicted)  # 0.07
rsq(gdsc2_gnn_exp$target, gdsc2_gnn_exp$predicted)  # 0.119  
# Conclusion, DepMap + CTRP is not good at predicting GDSC. Fine-tuning might help

# TEMP: Train on GDSC2, Test on CTRP using respective omic data (exp) ====
ctrp_gnn_exp <- fread("Data/CV_Results/HyperOpt_DRP_ResponseOnly_gnndrug_exp_HyperOpt_DRP_GDSC2_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_exp/CTRP_AAC_SMILES_inference_results.csv")
gdsc1_gnn_exp <- fread("Data/CV_Results/HyperOpt_DRP_ResponseOnly_gnndrug_exp_HyperOpt_DRP_GDSC2_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_exp/GDSC1_AAC_SMILES_inference_results.csv")
gdsdc2_gnn_exp <- fread("Data/CV_Results/HyperOpt_DRP_ResponseOnly_gnndrug_exp_HyperOpt_DRP_GDSC2_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_exp/GDSC2_AAC_SMILES_inference_results.csv")

rsq(ctrp_gnn_exp$target, ctrp_gnn_exp$predicted)  # 0.04
rsq(gdsc1_gnn_exp$target, gdsc1_gnn_exp$predicted)  # 0.12
rsq(gdsc2_gnn_exp$target, gdsc2_gnn_exp$predicted)  # 0.119


# ==== Bimodal Case ====
require(data.table)
require(ggplot2)
options(scipen = 3)
# all_csv_results <- list.files("Data/CV_Results/", "CV_results.csv", recursive = T, full.names = T)
all_csv_results <- list.files("Data/CV_Results/", "CTRP_.+_inference_results.csv", recursive = T, full.names = T)
bimodal_results <- grep(pattern = ".+drug_.{3,5}_HyperOpt.+", x = all_csv_results, value = T)


all_results <- vector(mode = "list", length = length(bimodal_results))
for (i in 1:length(bimodal_results)) {
  cur_res <- fread(bimodal_results[i])
  data_types <- gsub(".+ResponseOnly_\\w*drug_(.+)_HyperOpt.+", "\\1", bimodal_results[i])
  data_types <- toupper(data_types)
  merge_method <- gsub(".+MergeBy(\\w+)_.*RMSE.+", "\\1", bimodal_results[i])
  loss_method <- gsub(".+_(.*)RMSE.+", "\\1RMSE", bimodal_results[i])
  drug_type <- gsub(".+ResponseOnly_(\\w*)drug.+_HyperOpt.+", "\\1drug", bimodal_results[i])
  drug_type <- toupper(drug_type)
  split_method <- gsub(".+Split_(\\w+)_NoBottleNeck.+", "\\1", bimodal_results[i])
  # data_types <- strsplit(data_types, "_")[[1]]
  # cur_res$epoch <- as.integer(epoch)
  cur_res$data_types <- data_types
  cur_res$merge_method <- merge_method
  cur_res$loss_type <- loss_method
  cur_res$drug_type <- drug_type
  cur_res$split_method <- split_method
  
  all_results[[i]] <- cur_res
}
all_results <- rbindlist(all_results)

all_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method")]

# all_results <- all_results[!(V1 %in% c("max_final_epoch", "time_this_iter_s", "num_samples", "avg_cv_untrained_loss"))]

long_results <- melt(unique(all_results[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "loss_by_config")]),
                     id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method"))
# long_results[V1 == "avg_cv_train_loss"]$V1 <- "Mean CV Training Loss"
# long_results[V1 == "avg_cv_valid_loss"]$V1 <- "Mean CV Validation Loss"
# long_results <- long_results[split_method == "DRUG"]
# long_results <- long_results[merge_method == "Concat"]
# long_results <- long_results[merge_method == "Sum"]
# long_results <- long_results[loss_type == "RMSE"]
# long_results <- long_results[merge_method == "LMF" & loss_type == "WeightedRMSE"]
# long_results <- long_results[split_method == "CELL_LINE"]
# long_results <- long_results[drug_type == "DRUG"]
# All loss comparison ====
ggplot(long_results) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = split_method), stat = "identity", position='dodge') +
  facet_wrap(~merge_method+loss_type+drug_type+split_method, nrow = 2) + 
  scale_fill_discrete(name = "Split Type:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the bi-modal case"),
          subtitle = "Training RMSE Loss using strict splitting during hyper-parameter optimization")

dir.create("Plots/Training_Inference_Results")
ggsave(filename = "Plots/Training_Inference_Results/Bimodal_Full_RMSELoss_Comparison.pdf")


# Upper AAC loss comparison ====
temp_results <- all_results
temp_results <- temp_results[target > 0.7]
temp_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method")]
temp_results[, rsq_by_config := rsq(target, predicted), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method")]
temp_long_results <- melt(unique(temp_results[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "loss_by_config")]),
                     id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method"))

ggplot(temp_long_results) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = split_method), stat = "identity", position='dodge') +
  facet_wrap(~merge_method+loss_type+drug_type+split_method, nrow = 2) + 
  scale_fill_discrete(name = "Split Method:") +
  # scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
  #                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the bi-modal case"),
          subtitle = "Training RMSE Loss with AAC Targets >= 0.7, using strict splitting during hyper-parameter optimization")

dir.create("Plots/Training_Inference_Results")
ggsave(filename = "Plots/Training_Inference_Results/Bimodal_UpperAAC_RMSELoss_Comparison.pdf")


# temp_long_results <- melt(unique(temp_results[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "rsq_by_config")]),
#                           id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method"))
# 
# ggplot(temp_long_results) +
#   geom_bar(mapping = aes(x = data_types, y = value, fill = split_method), stat = "identity", position='dodge') +
#   facet_wrap(~merge_method+loss_type+drug_type+split_method, nrow = 2) + 
#   scale_fill_discrete(name = "Split Method:") +
#   # scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
#   #                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
#   ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the bi-modal case"),
#           subtitle = "Training RMSE Loss with AAC Targets >= 0.7, using strict splitting during hyper-parameter optimization")
# 
# dir.create("Plots/Training_Inference_Results")
# ggsave(filename = "Plots/Training_Inference_Results/Bimodal_UpperAAC_RMSELoss_Comparison.pdf")


# Lower AAC loss comparison ====
temp_results <- all_results
temp_results <- temp_results[target < 0.3]
temp_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method")]
temp_results[, rsq_by_config := rsq(target, predicted), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method")]
temp_long_results <- melt(unique(temp_results[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "loss_by_config")]),
                          id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method"))

ggplot(temp_long_results) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = split_method), stat = "identity", position='dodge') +
  facet_wrap(~merge_method+loss_type+drug_type+split_method, nrow = 2) + 
  scale_fill_discrete(name = "Split Method:") +
  # scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
  #                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the bi-modal case"),
          subtitle = "Training RMSE Loss with AAC Targets <= 0.3, using strict splitting during hyper-parameter optimization")

dir.create("Plots/Training_Inference_Results")
ggsave(filename = "Plots/Training_Inference_Results/Bimodal_LowerAAC_RMSELoss_Comparison.pdf")

# =========
all_gnn_inference_results <- list.files("Data/CV_Results/", "CTRP_AAC_SMILES_inference_results.csv", recursive = T, full.names = T)
all_morgan_inference_results <- list.files("Data/CV_Results/", "CTRP_AAC_MORGAN_1024_inference_results.csv", recursive = T, full.names = T)
gnn_bimodal_results <- grep(pattern = ".+gnndrug_.{3,5}_HyperOpt.+", x = all_gnn_inference_results, value = T)
morgan_bimodal_results <- grep(pattern = ".+_drug_.{3,5}_HyperOpt.+", x = all_morgan_inference_results, value = T)

all_gnn_results <- vector(mode = "list", length = length(gnn_bimodal_results))
for (i in 1:length(gnn_bimodal_results)) {
  cur_res <- fread(gnn_bimodal_results[i])
  data_types <- gsub(".+ResponseOnly_(.+)_HyperOpt.+", "\\1", gnn_bimodal_results[i])
  data_types <- toupper(data_types)
  # data_types <- strsplit(data_types, "_")[[1]]
  # cur_res$epoch <- as.integer(epoch)
  cur_res$data_types <- data_types
  all_gnn_results[[i]] <- cur_res
}

all_gnn_results <- rbindlist(all_gnn_results)

# rsq(all_gnn_results[data_types == "GNNDRUG_PROT"]$target, all_gnn_results[data_types == "GNNDRUG_PROT"]$predicted)
# rsq(all_gnn_results[data_types == "GNNDRUG_MUT"]$target, all_gnn_results[data_types == "GNNDRUG_MUT"]$predicted)
# mean(all_gnn_results[data_types == "GNNDRUG_PROT"]$RMSELoss)
# mean(all_gnn_results[data_types == "GNNDRUG_MUT"]$RMSELoss)

all_morgan_results <- vector(mode = "list", length = length(morgan_bimodal_results))
for (i in 1:length(morgan_bimodal_results)) {
  cur_res <- fread(morgan_bimodal_results[i])
  data_types <- gsub(".+ResponseOnly_(.+)_HyperOpt.+", "\\1", morgan_bimodal_results[i])
  data_types <- toupper(data_types)
  # data_types <- strsplit(data_types, "_")[[1]]
  # cur_res$epoch <- as.integer(epoch)
  cur_res$data_types <- data_types
  all_morgan_results[[i]] <- cur_res
}

all_morgan_results <- rbindlist(all_morgan_results)

# ggplot(data = all_gnn_results[data_types == "GNNDRUG_EXP"], aes(x = predicted, y = target)) +
#   geom_point() +
#   coord_fixed(ratio = 1) +
#   geom_abline(intercept = 0, colour = "red")
#   # facet_grid(~data_types,)
# 
# 
# rsq(all_morgan_results[data_types == "DRUG_PROT"]$target, all_morgan_results[data_types == "DRUG_PROT"]$predicted)
# mean(all_morgan_results[data_types == "DRUG_PROT"]$RMSELoss)
# 
# ggplot(data = all_morgan_results[data_types == "DRUG_PROT"], aes(x = predicted, y = target)) +
#   geom_point() +
#   coord_fixed(ratio = 1) +
#   geom_abline(intercept = 0, colour = "red")
#   # facet_grid(~data_types,)

all_data_types <- c("MUT", "EXP", "PROT", "MIRNA", "HIST", "METAB", "RPPA")


for (data_type in all_data_types) {
  cur_data <- rbindlist(list(all_morgan_results[data_types == paste0("DRUG_", data_type)], all_gnn_results[data_types == paste0("GNNDRUG_", data_type)]))
  
  morgan_rsq <- rsq(all_morgan_results[data_types == paste0("DRUG_", data_type)]$target, all_morgan_results[data_types == paste0("DRUG_", data_type)]$predicted)
  gnn_rsq <- rsq(all_gnn_results[data_types == paste0("GNNDRUG_", data_type)]$target, all_gnn_results[data_types == paste0("GNNDRUG_", data_type)]$predicted)
  
  p <- ggplot(data = cur_data, aes(x = predicted, y = target)) +
    geom_point() +
    coord_fixed(ratio = 1) +
    geom_abline(intercept = 0, colour = "red") +
    facet_grid(~data_types) +
    ggtitle(label = "Performance Comparison on CTRPv2", subtitle = paste0("Model Trained on CTRPv2, R^2 Morgan: ", round(morgan_rsq, 2), ", R^2 GNN Drug: ", round(gnn_rsq, 2)))
  ggsave(filename = paste0("Plots/R2_Line/CTRP_Morgan_vs_GNNDrug_", data_type, ".jpg"), plot = p)
}

# data_type <- "EXP"
# data_type <- "METAB"
cur_data <- all_gnn_results[data_types == paste0("GNNDRUG_", data_type)]
gnn_rsq <- rsq(all_gnn_results[data_types == paste0("GNNDRUG_", data_type)]$target, all_gnn_results[data_types == paste0("GNNDRUG_", data_type)]$predicted)

p <- ggplot(data = cur_data, aes(x = predicted, y = target)) +
  geom_point() +
  coord_fixed(ratio = 1) +
  geom_abline(intercept = 0, colour = "red") +
  facet_grid(~data_types) +
  ggtitle(label = "Performance Comparison on CTRPv2", subtitle = paste0("Model Trained on CTRPv2, R^2 GNN Drug: ", round(gnn_rsq, 2)))
ggsave(filename = paste0("Plots/R2_Line/GNNDrug_", data_type, ".jpg"), plot = p)

# data_type <- "EXP"
# data_type <- "METAB"
cur_data <- all_gnn_results[data_types == paste0("GNNDRUG_", data_type)]
gnn_rsq <- rsq(all_gnn_results[data_types == paste0("GNNDRUG_", data_type)]$target, all_gnn_results[data_types == paste0("GNNDRUG_", data_type)]$predicted)

p <- ggplot(data = cur_data, aes(x = predicted, y = target)) +
  geom_point() +
  coord_fixed(ratio = 1) +
  geom_abline(intercept = 0, colour = "red") +
  facet_grid(~data_types) +
  ggtitle(label = "Performance Comparison on CTRPv2", subtitle = paste0("Model Trained on CTRPv2, R^2 GNN Drug: ", round(gnn_rsq, 2)))
ggsave(filename = paste0("Plots/R2_Line/GNNDrug_", data_type, ".jpg"), plot = p)
