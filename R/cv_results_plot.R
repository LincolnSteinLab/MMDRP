# cv_results_plot.R

# ==== Bimodal Case ====
require(data.table)
require(ggplot2)
options(scipen = 3)
# all_csv_results <- list.files("Data/CV_Results/", "CV_results.csv", recursive = T, full.names = T)
all_csv_results <- list.files("Data/CV_Results/", "CTRP_AAC_SMILES_inference_results.csv", recursive = T, full.names = T)
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
all_results <- all_results[!(V1 %in% c("max_final_epoch", "time_this_iter_s", "num_samples", "avg_cv_untrained_loss"))]
long_results <- melt(all_results, id.vars = c("V1", "data_types", "merge_method", "loss_type", "drug_type", "split_method"))
long_results[V1 == "avg_cv_train_loss"]$V1 <- "Mean CV Training Loss"
long_results[V1 == "avg_cv_valid_loss"]$V1 <- "Mean CV Validation Loss"
# long_results <- long_results[split_method == "DRUG"]
# long_results <- long_results[merge_method == "Concat"]
# long_results <- long_results[merge_method == "Sum"]
long_results <- long_results[loss_type == "RMSE"]
# long_results <- long_results[merge_method == "LMF" & loss_type == "WeightedRMSE"]
# long_results <- long_results[split_method == "CELL_LINE"]
# long_results <- long_results[drug_type == "DRUG"]
ggplot(long_results) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = V1), stat = "identity", position='dodge') +
  facet_wrap(~merge_method+loss_type+drug_type+split_method, nrow = 2) + 
  scale_fill_discrete(name = "Loss Type:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the bi-modal case"),
          subtitle = "Cross-validation using strict drug and cell line splitting")
  
dir.create("Plots/CV_Results")
ggsave(filename = "Plots/CV_Results/Bimodal_LMF_vs_GNN_vs_LDS_CV_Stacked.pdf")


# ==== Multi-modal Case ====
all_csv_results <- list.files("Data/CV_Results/", "CV_results.csv", recursive = T, full.names = T)
# all_csv_results <- list.files("Data/CV_Results/", "CTRP_AAC_SMILES_inference_results.csv", recursive = T, full.names = T)
trimodal_results <- grep(pattern = ".+drug_.{6,}_HyperOpt.+", x = all_csv_results, value = T)


all_results <- vector(mode = "list", length = length(trimodal_results))
for (i in 1:length(trimodal_results)) {
  cur_res <- fread(trimodal_results[i])
  data_types <- gsub(".+ResponseOnly_\\w*drug_(.+)_HyperOpt.+", "\\1", trimodal_results[i])
  data_types <- toupper(data_types)
  merge_method <- gsub(".+MergeBy(\\w+)_.*RMSE.+", "\\1", trimodal_results[i])
  loss_method <- gsub(".+_(.*)RMSE.+", "\\1RMSE", trimodal_results[i])
  drug_type <- gsub(".+ResponseOnly_(\\w*)drug.+_HyperOpt.+", "\\1drug", trimodal_results[i])
  drug_type <- toupper(drug_type)
  # data_types <- strsplit(data_types, "_")[[1]]
  # cur_res$epoch <- as.integer(epoch)
  cur_res$data_types <- data_types
  cur_res$merge_method <- merge_method
  cur_res$loss_type <- loss_method
  cur_res$drug_type <- drug_type
  
  all_results[[i]] <- cur_res
}
all_results <- rbindlist(all_results)
all_results <- all_results[!(V1 %in% c("max_final_epoch", "time_this_iter_s", "num_samples", "avg_cv_untrained_loss"))]
long_results <- melt(all_results, id.vars = c("V1", "data_types", "merge_method", "loss_type", "drug_type"))
long_results[V1 == "avg_cv_train_loss"]$V1 <- "Mean CV Training Loss"
long_results[V1 == "avg_cv_valid_loss"]$V1 <- "Mean CV Validation Loss"
long_results <- long_results[-c(30, 32, 38), ]

ggplot(long_results) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = V1), stat = "identity", position='dodge') +
  coord_flip() + 
  facet_wrap(~merge_method+loss_type+drug_type, nrow = 1) + 
  scale_fill_discrete(name = "Loss Type:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the multi-modal case"),
          subtitle = "Cross-validation using strict drug and cell line splitting")

dir.create("Plots/CV_Results")
ggsave(filename = "Plots/CV_Results/Multimodal_LMF_vs_GNN_vs_LDS_CV_Horizontal.pdf")

# ==== Bi-modal vs Multi-modal comparison ====
all_csv_results <- list.files("Data/CV_Results/", "CV_results.csv", recursive = T, full.names = T)
bimodal_results <- grep(pattern = ".+drug_.{3,5}_HyperOpt.+", x = all_csv_results, value = T)
multimodal_results <- grep(pattern = ".+drug_.{6,}_HyperOpt.+", x = all_csv_results, value = T)

get_cv_results <- function(cur_results) {
  all_results <- vector(mode = "list", length = length(cur_results))
  for (i in 1:length(cur_results)) {
    cur_res <- fread(cur_results[i])
    data_types <- gsub(".+ResponseOnly_\\w*drug_(.+)_HyperOpt.+", "\\1", cur_results[i])
    data_types <- toupper(data_types)
    merge_method <- gsub(".+MergeBy(\\w+)_.*RMSE.+", "\\1", cur_results[i])
    loss_method <- gsub(".+_(.*)RMSE.+", "\\1RMSE", cur_results[i])
    drug_type <- gsub(".+ResponseOnly_(\\w*)drug.+_HyperOpt.+", "\\1drug", cur_results[i])
    drug_type <- toupper(drug_type)
    # data_types <- strsplit(data_types, "_")[[1]]
    # cur_res$epoch <- as.integer(epoch)
    cur_res$data_types <- data_types
    cur_res$merge_method <- merge_method
    cur_res$loss_type <- loss_method
    cur_res$drug_type <- drug_type
    
    all_results[[i]] <- cur_res
  }
  all_results <- rbindlist(all_results)
}
bi_results <- get_cv_results(bimodal_results)
multi_results <- get_cv_results(multimodal_results)
all_results <- rbindlist(list(bi_results, multi_results))
all_results <- all_results[!(V1 %in% c("max_final_epoch", "time_this_iter_s", "num_samples", "avg_cv_untrained_loss"))]
long_results <- melt(all_results, id.vars = c("V1", "data_types", "merge_method", "loss_type", "drug_type"))
long_results[V1 == "avg_cv_train_loss"]$V1 <- "Mean CV Training Loss"
long_results[V1 == "avg_cv_valid_loss"]$V1 <- "Mean CV Validation Loss"
# long_results <- long_results[-c(30, 32, 38), ]

long_results <- long_results[value < 1]
long_results <- long_results[merge_method == "LMF" & loss_type == "WeightedRMSE"]
ggplot(long_results) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = V1), stat = "identity", position='dodge') +
  geom_hline(yintercept = 0.05) +
  # coord_flip() +
  facet_wrap(~merge_method+loss_type+drug_type, nrow = 3, scales = 'free_x') + 
  scale_fill_discrete(name = "Loss Type:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
  # theme(axis.text.x = element_blank(),
        legend.position = 'none') +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the multi-modal case"),
          subtitle = "Cross-validation using strict drug and cell line splitting")

dir.create("Plots/CV_Results")
ggsave(filename = "Plots/CV_Results/Bimodal_vs_Multimodal_LMF_vs_GNN_vs_LDS_CV_Horizontal.pdf")
# ==== Inference Results ====
# cv_results_plot.R
require(data.table)
require(ggplot2)
options(scipen = 3)
# all_csv_results <- list.files("Data/CV_Results/", "CV_results.csv", recursive = T, full.names = T)
all_csv_results <- list.files("Data/CV_Results/", "CTRP_AAC_.*_inference_results.csv", recursive = T, full.names = T)
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
  # data_types <- strsplit(data_types, "_")[[1]]
  # cur_res$epoch <- as.integer(epoch)
  cur_res$data_types <- data_types
  cur_res$merge_method <- merge_method
  cur_res$loss_type <- loss_method
  cur_res$drug_type <- drug_type
  
  all_results[[i]] <- cur_res
}
all_results <- rbindlist(all_results)


all_results <- all_results[target > 0.6]
# Percentage of samples where predictions are within 0.2 RMSE
all_results[, within_range := RMSELoss < 0.2, by = .I]
all_results[, sum_within_range := sum(within_range), by = c("data_types", "merge_method", "loss_type", "drug_type")]
all_results[, nrow_sd := nrow(.SD), by = c("data_types", "merge_method", "loss_type", "drug_type")]
all_results[, perc_within_range := sum_within_range / nrow_sd]

within_range_results <- unique(all_results[, c("perc_within_range", "data_types", "merge_method", "loss_type", "drug_type")])

# all_results <- all_results[!(V1 %in% c("max_final_epoch", "time_this_iter_s", "num_samples", "avg_cv_untrained_loss"))]
# long_results <- melt(all_results, id.vars = c("V1", "data_types", "merge_method", "loss_type", "drug_type"))
# long_results[V1 == "avg_cv_train_loss"]$V1 <- "Mean CV Training Loss"
# long_results[V1 == "avg_cv_valid_loss"]$V1 <- "Mean CV Validation Loss"

ggplot(within_range_results) +
  geom_bar(mapping = aes(x = data_types, y = perc_within_range), stat = "identity", position='dodge') +
  facet_wrap(~merge_method+loss_type+drug_type, nrow = 1) + 
  # scale_fill_discrete(name = "Loss Type:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method bi-modal case"),
          subtitle = "Training RMSE, percentage of samples with RMSE <= 0.2")

dir.create("Plots/CV_Results")
ggsave(filename = "Plots/CV_Results/Bimodal_Perc_Within_Range.pdf")
