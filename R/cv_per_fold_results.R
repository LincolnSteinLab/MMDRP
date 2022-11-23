# cv_per_fold_results.R

require(data.table)
require(ggplot2)
options(scipen = 3)

drug_info <- fread("Data/DRP_Training_Data/CTRP_DRUG_INFO.csv")
targeted_drugs <- drug_info[gene_symbol_of_protein_target != "" & cpd_status == "clinical"]$rn

cur_cv_files <- list.files("Data/CV_Results/", recursive = T,
                            pattern = ".*final_validation.*", full.names = T)

# all_csv_results <- list.files("Data/CV_Results/", "CV_results.csv", recursive = T, full.names = T)
# all_csv_results <- list.files("Data/CV_Results/", "CTRP_AAC_SMILES_inference_results.csv", recursive = T, full.names = T)
# cur_cv_files <- grep(pattern = ".+drug_.{3,5}_HyperOpt.+", x = all_csv_results, value = T)


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

# mean(all_results$RMSELoss)
all_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold")]
all_results$V1 <- NULL
long_results <- melt(unique(all_results[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config")]),
                     id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold"))


long_results[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "split_method")]
# split_both_results <- long_results[split_method == "BOTH"]
# split_drug_results <- long_results[split_method == "DRUG"]
baseline_with_lds_results <- long_results[(merge_method == "Concat" & drug_type == "DRUG")]

targeted_drug_results <- all_results[cpd_name %in% targeted_drugs]
targeted_drug_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold")]
long_targeted_drug_results <- melt(unique(targeted_drug_results[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config")]),
                     id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold"))
long_targeted_drug_results[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "split_method")]

baseline_with_lds_targeted <- long_targeted_drug_results[(merge_method == "Concat" & drug_type == "DRUG")]

ggplot(baseline_with_lds_targeted) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = fold), stat = "identity", position='dodge') +
  facet_wrap(~merge_method+loss_type+drug_type+split_method, nrow = 2) + 
  scale_fill_discrete(name = "CV Fold:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the bi-modal case"),
          subtitle = "Validation RMSE loss using strict splitting") +
  geom_errorbar(aes(x=data_types,
                    y=cv_mean,
                    ymax=cv_mean, 
                    ymin=cv_mean, col='red'), linetype=2, show.legend = FALSE) +
  geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean), vjust = -0.5)

# scale_y_continuous(breaks = c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5)) + ylim(c(0, 0.7))

  
dir.create("Plots/CV_Results/")
# ggsave(filename = "Plots/CV_Results/Bimodal_CV_split_BOTH_per_fold_Full_Comparison.pdf")
# ggsave(filename = "Plots/CV_Results/Bimodal_CV_split_DRUG_per_fold_Full_Comparison.pdf")
ggsave(filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_with_LDS_Full_Comparison.pdf")

# = Upper AAC Comparison ====
temp_results <- all_results
temp_results$loss_by_config <- NULL
temp_results <- temp_results[target > 0.7]

temp_results <- temp_results[(merge_method == "Concat" & drug_type == "DRUG")]

temp_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold")]

long_temp_results <- melt(unique(temp_results[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config")]),
                     id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold"))


long_temp_results[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "split_method")]
# split_both_results <- long_temp_results[split_method == "BOTH"]
# split_drug_results <- long_temp_results[split_method == "DRUG"]

ggplot(long_temp_results) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = fold), stat = "identity", position='dodge') +
  facet_wrap(~merge_method+loss_type+drug_type+split_method, nrow = 2) + 
  scale_fill_discrete(name = "CV Fold:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the bi-modal case"),
          subtitle = "Subset of AAC >= 0.7 Validation RMSE loss using strict splitting") +
  geom_errorbar(aes(x=data_types,
                    y=cv_mean,
                    ymax=cv_mean, 
                    ymin=cv_mean, col='red'), linetype=2, show.legend = FALSE) +
  geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean), vjust = -0.5) 

# ggsave(filename = "Plots/CV_Results/Bimodal_CV_per_fold_split_BOTH_Upper_0.7_Comparison.pdf")
# ggsave(filename = "Plots/CV_Results/Bimodal_CV_per_fold_split_DRUG_Upper_0.7_Comparison.pdf")
ggsave(filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_with_LDS_Upper_0.7_Comparison.pdf")

# = Upper AAC (0.9) Comparison ====
temp_results <- all_results
temp_results$loss_by_config <- NULL
temp_results <- temp_results[target > 0.9]

temp_results <- temp_results[(merge_method == "Concat" & drug_type == "DRUG")]

temp_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold")]

long_temp_results <- melt(unique(temp_results[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config")]),
                          id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold"))


long_temp_results[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "split_method")]
# split_both_results <- long_temp_results[split_method == "BOTH"]
# split_drug_results <- long_temp_results[split_method == "DRUG"]

ggplot(long_temp_results) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = fold), stat = "identity", position='dodge') +
  facet_wrap(~merge_method+loss_type+drug_type+split_method, nrow = 2) + 
  scale_fill_discrete(name = "CV Fold:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the bi-modal case"),
          subtitle = "Subset of AAC >= 0.9 Validation RMSE loss using strict splitting") +
  geom_errorbar(aes(x=data_types,
                    y=cv_mean,
                    ymax=cv_mean, 
                    ymin=cv_mean, col='red'), linetype=2, show.legend = FALSE) +
  geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean), vjust = -0.5) 
# ggsave(filename = "Plots/CV_Results/Bimodal_CV_per_fold_split_BOTH_Upper_0.9_Comparison.pdf")
# ggsave(filename = "Plots/CV_Results/Bimodal_CV_per_fold_split_DRUG_Upper_0.9_Comparison.pdf")
ggsave(filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_with_LDS_Upper_0.9_Comparison.pdf")
