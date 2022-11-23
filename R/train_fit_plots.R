# train_fit_plots.R
require(data.table)
require(ggplot2)

gnn_lmf_lds <- fread("Data/CV_Results/HyperOpt_DRP_ResponseOnly_gnndrug_exp_HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GnnDrugs_gnndrug_exp/CTRP_AAC_SMILES_inference_results.csv")
gnn_lmf_lds <- fread("Data/EpochResults/CrossValidation/HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_exp/CV_Index_1_Epoch_100_inference_results.csv")

mean(gnn_lmf_lds$RMSELoss)
sd(gnn_lmf_lds$RMSELoss)
rsq <- function (x, y) cor(x, y) ^ 2
rsq(gnn_lmf_lds$target, gnn_lmf_lds$predicted)

plot_loss_by_lineage <- function(path,
                                 plot_path, cell_line_data, title, subtitle, plot_filename, display_plot = FALSE) {
  
  cv_results <- fread(paste0(path, "CV_results.csv"))
  cv_valid_loss <- cv_results[V1 == "avg_cv_valid_loss"][,2]
  cv_valid_loss <- format(round(cv_valid_loss, 4), nsmall = 4)
  ctrp_data <- fread(paste0(path, "CTRP_AAC_MORGAN_1024_inference_results.csv"))
  ctrp_data <- merge(ctrp_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
  # gdsc1_data <- fread(paste0(path, "GDSC1_AAC_MORGAN_1024_inference_results.csv"))
  # gdsc1_data <- merge(gdsc1_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
  # gdsc2_data <- fread(paste0(path, "GDSC2_AAC_MORGAN_1024_inference_results.csv"))
  # gdsc2_data <- merge(gdsc2_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
  
  # ctrp_data[, abs_loss := sqrt(MSE_loss)]
  ctrp_data[, lineage_loss_avg := mean(RMSE_loss), by = "lineage"]
  ctrp_data[, lineage_loss_sd := sd(RMSE_loss), by = "lineage"]
  ctrp_data[, sample_by_lineage_count := .N, by = "lineage"]
  ctrp_avg_abs_by_lineage <- unique(ctrp_data[, c("lineage", "lineage_loss_avg", "lineage_loss_sd")])
  ctrp_avg_abs_by_lineage$Dataset <- "CTRPv2"
  
  # gdsc1_data[, lineage_loss_avg := mean(RMSE_loss), by = "lineage"]
  # gdsc1_data[, lineage_loss_sd := sd(RMSE_loss), by = "lineage"]
  # gdsc1_data[, sample_by_lineage_count := .N, by = "lineage"]
  # gdsc1_avg_abs_by_lineage <- unique(gdsc1_data[, c("lineage", "lineage_loss_avg", "lineage_loss_sd")])
  # gdsc1_avg_abs_by_lineage$Dataset <- "GDSC1"
  # 
  # gdsc2_data[, lineage_loss_avg := mean(RMSE_loss), by = "lineage"]
  # gdsc2_data[, lineage_loss_sd := sd(RMSE_loss), by = "lineage"]
  # gdsc2_data[, sample_by_lineage_count := .N, by = "lineage"]
  # gdsc2_avg_abs_by_lineage <- unique(gdsc2_data[, c("lineage", "lineage_loss_avg", "lineage_loss_sd")])
  # gdsc2_avg_abs_by_lineage$Dataset <- "GDSC2"
  
  # all_avg_abs_by_lineage <- rbindlist(list(ctrp_avg_abs_by_lineage, gdsc1_avg_abs_by_lineage, gdsc2_avg_abs_by_lineage))
  all_avg_abs_by_lineage <- ctrp_avg_abs_by_lineage
  all_avg_abs_by_lineage <- merge(all_avg_abs_by_lineage, unique(ctrp_data[, c("lineage", "sample_by_lineage_count")]))
  all_avg_abs_by_lineage$lineage <- paste0(all_avg_abs_by_lineage$lineage, ", n = ", all_avg_abs_by_lineage$sample_by_lineage_count)
  
  g <- ggplot(data = all_avg_abs_by_lineage, mapping = aes(x = reorder(lineage, -lineage_loss_avg), y = lineage_loss_avg, fill = Dataset)) +
    geom_bar(stat = "identity", position = position_dodge()) +
    # geom_errorbar(aes(ymin = lineage_loss_avg - lineage_loss_sd, ymax = lineage_loss_avg + lineage_loss_sd), width = 0.2, position = position_dodge(0.9)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
    geom_hline(yintercept = mean(ctrp_data$lineage_loss_avg), linetype="dashed", color = "red") +
    # geom_text(aes(10, mean(ctrp_data$abs_loss),label = mean(ctrp_data$abs_loss), vjust = -1)) +
    # geom_hline(yintercept = mean(gdsc1_data$lineage_loss_avg), linetype="dashed", color = "green") +
    # geom_hline(yintercept = mean(gdsc2_data$lineage_loss_avg), linetype="dashed", color = "blue") +
    xlab("Cell Line Lineage + # testing datapoints") + ylab("RMSE Loss") + 
    # scale_y_discrete(limits = c("0.001", "0.002")) +
    scale_y_continuous(breaks = sort(c(seq(0, 0.25, length.out=10),
                                       c(mean(ctrp_data$lineage_loss_avg)
                                         # mean(gdsc1_data$lineage_loss_avg),
                                         # mean(gdsc2_data$lineage_loss_avg)
                                       )
    ))) +
    # ggtitle(label = "Full DRP Mean Absolute Loss by Cell Line Lineage", subtitle = "Data: Drug + Proteomics | Trained on CTRPv2 | Tested on All 3")
    ggtitle(label = title, subtitle = paste0(subtitle, "\nAverage Cross-Validation RMSE Loss:", as.character(cv_valid_loss)))
  if (display_plot == TRUE) {
    print(g)
  }
  # ggsave(filename = paste0(plot_path, "drug_prot_train_CTRPv2_test_All_avg_Abs_by_lineage.pdf"), device = "pdf")
  ggsave(plot = g, filename = paste0(plot_path, plot_filename), device = "pdf")
  
}

all_lmf_dirs <- c(grep(".*MergeByLMF.*", list.dirs("Data/CV_Results/"), value = T))
  # grep(".*MergeBySum_WeightedRMSELoss.*", list.dirs("Data/CV_Results/"), value = T))

all_sum_dirs <- gsub("MergeByLMF", "MergeBySum", all_lmf_dirs)
all_dirs <- c(all_lmf_dirs, all_sum_dirs)
# all_dirs <- c(all_lmf_dirs)

all_ctrp_results <- vector(mode = "list", length = length(all_dirs))
for (i in 1:length(all_dirs)) {
  cur_res <- fread(paste0(all_dirs[i], "/CTRP_AAC_MORGAN_1024_inference_results.csv"))
  tag <- gsub("Data.*_MorganDrugs_", "", all_dirs[i])
  merge_method <- gsub(".*(MergeBy.{3}).*", "\\1", all_dirs[i])
  cur_res$tag <- paste(merge_method, tag, sep = "_")
  all_ctrp_results[[i]] <- cur_res
}

all_res <- rbindlist(all_ctrp_results)
all_res

ctrp <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")
t1 <- unique(all_res[, c("cpd_name", "cell_name", "target")])
t2 <- unique(ctrp[, c("cpd_name", "ccl_name", "area_above_curve")])
colnames(t2) <- c("cpd_name", "cell_name", "target")
setorder(t1)
setorder(t2)
t1$target <- round(t1$target, 3)
t2$target <- round(t2$target, 3)
is_df1_subset_of_df2 <- function(t1, t2) {
  intersection <- data.table::fintersect(t1, t2)
  data.table::fsetequal(t1, intersection)
}
is_df1_subset_of_df2(t1, t2)



max(all_res$predicted)
min(all_res$predicted)
all_res[, mean(target), by = cell_name][cell_name == "ZR7530"]


all_res <- fread("Data/CV_Results//HyperOpt_DRP_ResponseOnly_drug_cnv_exp_prot_HyperOpt_DRP_CTRP_1024_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeBySum_WeightedRMSELoss_MorganDrugs_drug_cnv_exp_prot/CTRP_AAC_MORGAN_1024_inference_results.csv")
all_res$tag <- "MergeBySum_drug_cnv_exp_prot"
mean(all_res$RMSELoss)

ggplot(gnn_lmf_lds) +
  # geom_freqpoly(aes(x = predicted, group = tag, color = tag), binwidth = 0.01) +
  geom_freqpoly(aes(x = predicted), binwidth = 0.01) +
  # geom_density(alpha = 0.2, aes(x = target, fill = "Targets")) + 
  geom_freqpoly(aes(x = target, color = "Targets"), binwidth = 0.01) + 
  stat_summary_bin(aes(x = target, y = RMSELoss*100000), fun = "mean", geom = "bar", size = 1, binwidth = 0.01, alpha = 0.2) +
  # facet_wrap(~tag) +
  # ggtitle(label = title,
  #         subtitle = subtitle) +
  xlab("Area Above Curve") + ylab("Density")



plot_grid_mono <- function(model_type, data_type, split, bottleneck, drug_type) {
  path <- paste0("Data/CV_Results/HyperOpt_DRP_", model_type, "_drug", data_type,
                 "_HyperOpt_DRP_CTRP_1024_", model_type, "_EncoderTrain_Split_", split, "_", bottleneck, "_NoTCGAPretrain_MergeBySum_RMSELoss_", drug_type, "_drug", data_type, "/")
  # HyperOpt_DRP_ResponseOnly_drug_rppa_HyperOpt_DRP_CTRP_1024_ResponseOnly_EncoderTrain_Split_DRUG_NoBottleNeck_NoTCGAPretrain_MergeBySum_RMSELoss_OneHotDrugs_drug_rppa
  if (split == "CELL_LINE") {
    plot_path <- "Plots/DRP/Split_by_Cell/"
    plot_split_name <- "SplitByCell"
    title_split_name <- "Cell Line"
  } else if (split == "DRUG") {
    plot_path <- "Plots/DRP/Split_by_Drug/"
    plot_split_name <- "SplitByDrug"
    title_split_name <- "Drug"
    
  } else {
    plot_path <- "Plots/DRP/Split_by_Both/"
    plot_split_name <- "SplitByBoth"
    title_split_name <- "Cell Line & Drug"
    
  }
  if (bottleneck == "WithBottleNeck") {
    subtitle_bottleneck_name <- "With Bottleneck"
    
  } else  {
    subtitle_bottleneck_name <- "No Bottleneck"
  }
  
  dir.create(plot_path)
  plot_filename <- paste0(model_type, "_drug", data_type, "_train_CTRPv2_test_All_RMSE_", plot_split_name, "_", bottleneck, "_", drug_type, ".pdf")
  title <- paste0("DRP RMSE (Validation by Strict ", title_split_name, " Splitting)")
  subtitle <- paste0("Model Type: ", model_type, " | Data: Drug + ", gsub("_", "", toupper(data_type)), " | Drug Type: ", drug_type, " | Trained on CTRPv2 | Tested on All 3 | Hyper-Param Search: ", subtitle_bottleneck_name)
  plot_loss_by_lineage(path = path, plot_path = plot_path, cell_line_data = cell_line_data, title = title, subtitle = subtitle, plot_filename = plot_filename)
  
}

model_types <- c("FullModel", "ResponseOnly")
data_types <- c("mut", "exp", "prot", "mirna", "metab", "rppa", "hist")
data_types <- paste0("_", data_types)
data_types <- c("", data_types)
# splits <- c("CELL_LINE", "DRUG", "BOTH")
splits <- c("DRUG")
# bottlenecking <- c("WithBottleNeck", "NoBottleNeck")
bottlenecking <- c("NoBottleNeck")
drug_types <- c("OneHotDrugs")
grid <- expand.grid(model_types, data_types, splits, bottlenecking, drug_types)

for (i in 1:nrow(grid)) {
  plot_grid_mono(model_type = grid[i, 1], data_type = grid[i, 2], split = grid[i, 3], bottleneck = grid[i, 4], drug_type = grid[i, 5])
}
