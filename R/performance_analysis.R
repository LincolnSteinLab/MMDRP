# performance_analysis.R

require(data.table)
require(ggplot2)


plot_loss_by_bottleneck_and_split <- function(with_bottleneck_path, without_bottleneck_path, split_by_cell_path,
                                 split_by_drug_path, split_by_both_path,
                                 plot_path, cell_line_data, inference_results_paths, subtitle, plot_name) {
  
  
  ctrp_data <- fread(paste0(path, "CTRP_AAC_MORGAN_512_inference_results.csv"))
  ctrp_data <- merge(ctrp_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
  gdsc1_data <- fread(paste0(path, "GDSC1_AAC_MORGAN_512_inference_results.csv"))
  gdsc1_data <- merge(gdsc1_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
  gdsc2_data <- fread(paste0(path, "GDSC2_AAC_MORGAN_512_inference_results.csv"))
  gdsc2_data <- merge(gdsc2_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
  
  # ctrp_data[, abs_loss := sqrt(MSE_loss)]
  ctrp_data[, lineage_loss_avg := mean(MAE_loss), by = "lineage"]
  ctrp_data[, lineage_loss_sd := sd(MAE_loss), by = "lineage"]
  ctrp_data[, sample_by_lineage_count := .N, by = "lineage"]
  ctrp_avg_abs_by_lineage <- unique(ctrp_data[, c("lineage", "lineage_loss_avg", "lineage_loss_sd")])
  ctrp_avg_abs_by_lineage$Dataset <- "CTRPv2"
  
  # gdsc1_data[, abs_loss := sqrt(MSE_loss)]
  gdsc1_data[, lineage_loss_avg := mean(MAE_loss), by = "lineage"]
  gdsc1_data[, lineage_loss_sd := sd(MAE_loss), by = "lineage"]
  gdsc1_data[, sample_by_lineage_count := .N, by = "lineage"]
  gdsc1_avg_abs_by_lineage <- unique(gdsc1_data[, c("lineage", "lineage_loss_avg", "lineage_loss_sd")])
  gdsc1_avg_abs_by_lineage$Dataset <- "GDSC1"
  
  # gdsc2_data[, abs_loss := sqrt(MSE_loss)]
  gdsc2_data[, lineage_loss_avg := mean(MAE_loss), by = "lineage"]
  gdsc2_data[, lineage_loss_sd := sd(MAE_loss), by = "lineage"]
  gdsc2_data[, sample_by_lineage_count := .N, by = "lineage"]
  gdsc2_avg_abs_by_lineage <- unique(gdsc2_data[, c("lineage", "lineage_loss_avg", "lineage_loss_sd")])
  gdsc2_avg_abs_by_lineage$Dataset <- "GDSC2"
  
  all_avg_abs_by_lineage <- rbindlist(list(ctrp_avg_abs_by_lineage, gdsc1_avg_abs_by_lineage, gdsc2_avg_abs_by_lineage))
  all_avg_abs_by_lineage <- merge(all_avg_abs_by_lineage, unique(ctrp_data[, c("lineage", "sample_by_lineage_count")]))
  all_avg_abs_by_lineage$lineage <- paste0(all_avg_abs_by_lineage$lineage, ", n = ", all_avg_abs_by_lineage$sample_by_lineage_count)
  
  ggplot(data = all_avg_abs_by_lineage, mapping = aes(x = reorder(lineage, -lineage_loss_avg), y = lineage_loss_avg, fill = Dataset)) +
    geom_bar(stat = "identity", position = position_dodge()) +
    # geom_errorbar(aes(ymin = lineage_loss_avg - lineage_loss_sd, ymax = lineage_loss_avg + lineage_loss_sd), width = 0.2, position = position_dodge(0.9)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
    geom_hline(yintercept = mean(ctrp_data$abs_loss), linetype="dashed", color = "red") +
    # geom_text(aes(10, mean(ctrp_data$abs_loss),label = mean(ctrp_data$abs_loss), vjust = -1)) +
    geom_hline(yintercept = mean(gdsc1_data$abs_loss), linetype="dashed", color = "green") +
    geom_hline(yintercept = mean(gdsc2_data$abs_loss), linetype="dashed", color = "blue") +
    xlab("Cell Line Lineage + # training datapoints") + ylab("Mean Absolute Loss") + 
    # scale_y_discrete(limits = c("0.001", "0.002")) +
    scale_y_continuous(breaks = sort(c(seq(0, 0.25, length.out=10),
                                       c(mean(ctrp_data$abs_loss),
                                         mean(gdsc1_data$abs_loss),
                                         mean(gdsc2_data$abs_loss))
    ))) +
    # ggtitle(label = "Full DRP Mean Absolute Loss by Cell Line Lineage", subtitle = "Data: Drug + Proteomics | Trained on CTRPv2 | Tested on All 3")
    ggtitle(label = title, subtitle = subtitle)
  # ggsave(filename = paste0(plot_path, "drug_prot_train_CTRPv2_test_All_avg_Abs_by_lineage.pdf"), device = "pdf")
  ggsave(filename = paste0(plot_path, plot_name), device = "pdf")
  
}


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
# plot_path <- "Plots/DRP/Lineage_Results/"
cell_line_data <- fread("Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv")


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

# ==== Drug + Mut ====

# Split by Cell, No Bottleneck
path = "Data/CV_Results/HyperOpt_DRP_FullModel_drug_mut_HyperOpt_DRP_CTRP_FullModel_EncoderTrain_Split_CELL_LINE_NoBottleNeck_WithTCGAPretrain_drug_mut/"
plot_path <- "Plots/DRP/Split_by_Cell/"
dir.create(plot_path)
plot_filename <- "drug_prot_train_CTRPv2_test_All_MAE_SplitByCell_NoBottleneck.pdf"
subtitle <- "Data: Drug + Mutational | Trained on CTRPv2 | Tested on All 3 | Hyper-Param Search: No Bottleneck"
title <- "Full DRP Mean Absolute Loss (Validation by Strict Cell Line Splitting)"
plot_loss_by_lineage(path = path, plot_path = plot_path, cell_line_data = cell_line_data, title = title, subtitle = subtitle, plot_filename = plot_filename)

# Split by Cell, With Bottleneck
path = "Data/CV_Results/HyperOpt_DRP_FullModel_drug_mut_HyperOpt_DRP_CTRP_FullModel_EncoderTrain_Split_CELL_LINE_WithBottleNeck_WithTCGAPretrain_drug_mut/"
plot_path <- "Plots/DRP/Split_by_Cell/"
dir.create(plot_path)
plot_filename <- "drug_prot_train_CTRPv2_test_All_MAE_SplitByCell_WithBottleneck.pdf"
subtitle <- "Data: Drug + Mutational | Trained on CTRPv2 | Tested on All 3 | Hyper-Param Search: With Bottleneck"
title <- "Full DRP Mean Absolute Loss (Validation by Strict Cell Line Splitting)"
plot_loss_by_lineage(path = path, plot_path = plot_path, cell_line_data = cell_line_data, title = title, subtitle = subtitle, plot_filename = plot_filename)
# ============
# Split by Drug, No Bottleneck
path = "Data/CV_Results/HyperOpt_DRP_FullModel_drug_mut_HyperOpt_DRP_CTRP_FullModel_EncoderTrain_Split_DRUG_NoBottleNeck_WithTCGAPretrain_drug_mut/"
plot_path <- "Plots/DRP/Split_by_Drug/"
dir.create(plot_path)
plot_filename <- "drug_prot_train_CTRPv2_test_All_MAE_SplitByDrug_NoBottleneck.pdf"
subtitle <- "Data: Drug + Mutational | Trained on CTRPv2 | Tested on All 3 | Hyper-Param Search: No Bottleneck"
title <- "Full DRP Mean Absolute Loss (Validation by Strict Drug Splitting)"
plot_loss_by_lineage(path = path, plot_path = plot_path, cell_line_data = cell_line_data, title = title, subtitle = subtitle, plot_filename = plot_filename)

# Split by Drug, With Bottleneck
path = "Data/CV_Results/HyperOpt_DRP_FullModel_drug_mut_HyperOpt_DRP_CTRP_FullModel_EncoderTrain_Split_DRUG_WithBottleNeck_WithTCGAPretrain_drug_mut/"
plot_path <- "Plots/DRP/Split_by_Drug/"
dir.create(plot_path)
plot_filename <- "drug_prot_train_CTRPv2_test_All_MAE_SplitByDrug_WithBottleneck.pdf"
subtitle <- "Data: Drug + Mutational | Trained on CTRPv2 | Tested on All 3 | Hyper-Param Search: With Bottleneck"
title <- "Full DRP Mean Absolute Loss (Validation by Strict Drug Splitting)"
plot_loss_by_lineage(path = path, plot_path = plot_path, cell_line_data = cell_line_data, title = title, subtitle = subtitle, plot_filename = plot_filename)
# ============
# Split by Drug, No Bottleneck
path = "Data/CV_Results/HyperOpt_DRP_FullModel_drug_mut_HyperOpt_DRP_CTRP_FullModel_EncoderTrain_Split_BOTH_NoBottleNeck_WithTCGAPretrain_drug_mut/"
plot_path <- "Plots/DRP/Split_by_Both/"
dir.create(plot_path)
plot_filename <- "drug_prot_train_CTRPv2_test_All_MAE_SplitByBoth_NoBottleneck.pdf"
subtitle <- "Data: Drug + Mutational | Trained on CTRPv2 | Tested on All 3 | Hyper-Param Search: No Bottleneck"
title <- "Full DRP Mean Absolute Loss (Validation by Strict Cell Line & Drug Splitting)"
plot_loss_by_lineage(path = path, plot_path = plot_path, cell_line_data = cell_line_data, title = title, subtitle = subtitle, plot_filename = plot_filename)

# Split by Drug, With Bottleneck
path = "Data/CV_Results/HyperOpt_DRP_FullModel_drug_mut_HyperOpt_DRP_CTRP_FullModel_EncoderTrain_Split_BOTH_WithBottleNeck_WithTCGAPretrain_drug_mut/"
plot_path <- "Plots/DRP/Split_by_Both/"
dir.create(plot_path)
plot_filename <- "drug_prot_train_CTRPv2_test_All_MAE_SplitByBoth_NoBottleneck.pdf"
subtitle <- "Data: Drug + Mutational | Trained on CTRPv2 | Tested on All 3 | Hyper-Param Search: With Bottleneck"
title <- "Full DRP Mean Absolute Loss (Validation by Strict Cell Line & Drug Splitting)"
plot_loss_by_lineage(path = path, plot_path = plot_path, cell_line_data = cell_line_data, title = title, subtitle = subtitle, plot_filename = plot_filename)


# 
plot_path <- "Plots/DRP/Split_by_Drug/"
plot_path <- "Plots/DRP/Split_by_Both/"
plot_name <- ""
# Plot average MSE by lineage Full (drug + prot) ================================
path = "Data/CV_Results/HyperOpt_DRP_FullModel_drug_prot_CTRP_Full/"

### GDSC1 ====
cur_data <- fread(paste0(path, "GDSC1_AAC_MORGAN_512_inference_results.csv"))
cur_data <- merge(cur_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

cur_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
avg_mse_by_lineage <- unique(cur_data[, c("lineage", "lineage_loss_avg")])
ggplot(data =  avg_mse_by_lineage)+
  geom_bar(mapping = aes(x = reorder(lineage,-lineage_loss_avg), y = lineage_loss_avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text()) + 
  geom_hline(yintercept = mean(cur_data$MSE_loss), linetype="dashed", color = "red") +
  xlab("Cell Line Lineage") + ylab("Average MSE Loss") +
  ggtitle(label = "Full DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Proteomics | Trained on CTRPv2 | Tested on GDSC1")
ggsave(filename = paste0(plot_path, "drug_prot_full_train_CTRPv2_test_GDSC1_avg_MSE_by_lineage.pdf"), device = "pdf")


### GDSC2 ====
cur_data <- fread(paste0(path, "GDSC2_AAC_MORGAN_512_inference_results.csv"))
cur_data <- merge(cur_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

cur_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
avg_mse_by_lineage <- unique(cur_data[, c("lineage", "lineage_loss_avg")])
ggplot(data =  avg_mse_by_lineage)+
  geom_bar(mapping = aes(x = reorder(lineage,-lineage_loss_avg), y = lineage_loss_avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text()) + 
  geom_hline(yintercept = mean(cur_data$MSE_loss), linetype="dashed", color = "red") +
  xlab("Cell Line Lineage") + ylab("Average MSE Loss") +
  ggtitle(label = "Full DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Proteomics | Trained on CTRPv2 | Tested on GDSC2")
ggsave(filename = paste0(plot_path, "drug_prot_full_train_CTRPv2_test_GDSC2_avg_MSE_by_lineage.pdf"), device = "pdf")


### CTRPv2 ====
cur_data <- fread(paste0(path, "CTRP_AAC_MORGAN_512_inference_results.csv"))
cur_data <- merge(cur_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

cur_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
avg_mse_by_lineage <- unique(cur_data[, c("lineage", "lineage_loss_avg")])
ggplot(data = avg_mse_by_lineage) +
  geom_bar(mapping = aes(x = reorder(lineage,-lineage_loss_avg), y = lineage_loss_avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text()) + 
  geom_hline(yintercept = mean(cur_data$MSE_loss), linetype="dashed", color = "red") +
  xlab("Cell Line Lineage") + ylab("Average MSE Loss") +
  ggtitle(label = "Full DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Proteomics | Trained on CTRPv2 | Tested on CTRPv2")
ggsave(filename = paste0(plot_path, "drug_prot_train_CTRPv2_test_CTRP_avg_MSE_by_lineage.pdf"), device = "pdf")


### All side by side (lineage bar plot) ====
ctrp_data <- fread(paste0(path, "CTRP_AAC_MORGAN_512_inference_results.csv"))
ctrp_data <- merge(ctrp_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
gdsc1_data <- fread(paste0(path, "GDSC1_AAC_MORGAN_512_inference_results.csv"))
gdsc1_data <- merge(gdsc1_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
gdsc2_data <- fread(paste0(path, "GDSC2_AAC_MORGAN_512_inference_results.csv"))
gdsc2_data <- merge(gdsc2_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

ctrp_data[, abs_loss := sqrt(MSE_loss)]
ctrp_data[, lineage_loss_avg := mean(abs_loss), by = "lineage"]
ctrp_data[, lineage_loss_sd := sd(abs_loss), by = "lineage"]
ctrp_data[, sample_by_lineage_count := .N, by = "lineage"]
ctrp_avg_abs_by_lineage <- unique(ctrp_data[, c("lineage", "lineage_loss_avg", "lineage_loss_sd")])
ctrp_avg_abs_by_lineage$Dataset <- "CTRPv2"

gdsc1_data[, abs_loss := sqrt(MSE_loss)]
gdsc1_data[, lineage_loss_avg := mean(abs_loss), by = "lineage"]
gdsc1_data[, lineage_loss_sd := sd(abs_loss), by = "lineage"]
gdsc1_data[, sample_by_lineage_count := .N, by = "lineage"]
gdsc1_avg_abs_by_lineage <- unique(gdsc1_data[, c("lineage", "lineage_loss_avg", "lineage_loss_sd")])
gdsc1_avg_abs_by_lineage$Dataset <- "GDSC1"

gdsc2_data[, abs_loss := sqrt(MSE_loss)]
gdsc2_data[, lineage_loss_avg := mean(abs_loss), by = "lineage"]
gdsc2_data[, lineage_loss_sd := sd(abs_loss), by = "lineage"]
gdsc2_data[, sample_by_lineage_count := .N, by = "lineage"]
gdsc2_avg_abs_by_lineage <- unique(gdsc2_data[, c("lineage", "lineage_loss_avg", "lineage_loss_sd")])
gdsc2_avg_abs_by_lineage$Dataset <- "GDSC2"

all_avg_abs_by_lineage <- rbindlist(list(ctrp_avg_abs_by_lineage, gdsc1_avg_abs_by_lineage, gdsc2_avg_abs_by_lineage))
all_avg_abs_by_lineage <- merge(all_avg_abs_by_lineage, unique(ctrp_data[, c("lineage", "sample_by_lineage_count")]))
all_avg_abs_by_lineage$lineage <- paste0(all_avg_abs_by_lineage$lineage, ", n = ", all_avg_abs_by_lineage$sample_by_lineage_count)

ggplot(data = all_avg_abs_by_lineage, mapping = aes(x = reorder(lineage, -lineage_loss_avg), y = lineage_loss_avg, fill = Dataset)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  # geom_errorbar(aes(ymin = lineage_loss_avg - lineage_loss_sd, ymax = lineage_loss_avg + lineage_loss_sd), width = 0.2, position = position_dodge(0.9)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  geom_hline(yintercept = mean(ctrp_data$abs_loss), linetype="dashed", color = "red") +
  # geom_text(aes(10, mean(ctrp_data$abs_loss),label = mean(ctrp_data$abs_loss), vjust = -1)) +
  geom_hline(yintercept = mean(gdsc1_data$abs_loss), linetype="dashed", color = "green") +
  geom_hline(yintercept = mean(gdsc2_data$abs_loss), linetype="dashed", color = "blue") +
  xlab("Cell Line Lineage + # training datapoints") + ylab("Average Absolute Loss") + 
  # scale_y_discrete(limits = c("0.001", "0.002")) +
  scale_y_continuous(breaks = sort(c(seq(0, 0.12, length.out=5),
                                     c(mean(ctrp_data$abs_loss),
                                       mean(gdsc1_data$abs_loss),
                                       mean(gdsc2_data$abs_loss))
  ))) +
  ggtitle(label = "Full DRP Mean Absolute Loss by Cell Line Lineage", subtitle = "Data: Drug + Proteomics | Trained on CTRPv2 | Tested on All 3")
ggsave(filename = paste0(plot_path, "drug_prot_train_CTRPv2_test_All_avg_Abs_by_lineage.pdf"), device = "pdf")


### All side by side (cell line dot plot) ====
path = "Data/CV_Results/HyperOpt_DRP_FullModel_drug_prot_CTRP_Full/"

ctrp_data <- fread(paste0(path, "CTRP_AAC_MORGAN_512_inference_results.csv"))
ctrp_data <- merge(ctrp_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
gdsc1_data <- fread(paste0(path, "GDSC1_AAC_MORGAN_512_inference_results.csv"))
gdsc1_data <- merge(gdsc1_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
gdsc2_data <- fread(paste0(path, "GDSC2_AAC_MORGAN_512_inference_results.csv"))
gdsc2_data <- merge(gdsc2_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

ctrp_data[, lineage_loss_sd := sd(MSE_loss), by = "lineage"]
ctrp_data[, cell_line_loss_avg := mean(MSE_loss), by = "cell_name"]
ctrp_avg_mse_by_cell_line <- unique(ctrp_data[, c("cell_name", "lineage", "cell_line_loss_avg", "lineage_loss_sd")])
ctrp_avg_mse_by_cell_line$Dataset <- "CTRPv2"

gdsc1_data[, lineage_loss_sd := sd(MSE_loss), by = "lineage"]
gdsc1_data[, cell_line_loss_avg := mean(MSE_loss), by = "cell_name"]
gdsc1_avg_mse_by_cell_line <- unique(gdsc1_data[, c("cell_name", "lineage", "cell_line_loss_avg", "lineage_loss_sd")])
gdsc1_avg_mse_by_cell_line$Dataset <- "GDSC1"

gdsc2_data[, lineage_loss_sd := sd(MSE_loss), by = "lineage"]
gdsc2_data[, cell_line_loss_avg := mean(MSE_loss), by = "cell_name"]
gdsc2_avg_mse_by_cell_line <- unique(gdsc2_data[, c("cell_name", "lineage", "cell_line_loss_avg", "lineage_loss_sd")])
gdsc2_avg_mse_by_cell_line$Dataset <- "GDSC2"

all_avg_mse_by_cell_line <- rbindlist(list(ctrp_avg_mse_by_cell_line, gdsc1_avg_mse_by_cell_line, gdsc2_avg_mse_by_cell_line))
ggplot(data = all_avg_mse_by_cell_line, mapping = aes(x = cell_name, y = cell_line_loss_avg, group = Dataset)) +
  facet_wrap(vars(lineage), scales = "free") +
  # geom_bar(stat = "identity", position = position_dodge()) +
  # geom_dotplot(binaxis = 'y', stackdir = 'center') +
  geom_boxplot() +
  geom_errorbar(aes(ymin = cell_line_loss_avg - lineage_loss_sd, ymax = cell_line_loss_avg + lineage_loss_sd), width = 0.2, position = position_dodge(0.9)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  geom_hline(yintercept = mean(ctrp_data$MSE_loss), linetype="dashed", color = "red") +
  # geom_text(aes(10, mean(ctrp_data$MSE_loss),label = mean(ctrp_data$MSE_loss), vjust = -1)) +
  geom_hline(yintercept = mean(gdsc1_data$MSE_loss), linetype="dashed", color = "green") +
  geom_hline(yintercept = mean(gdsc2_data$MSE_loss), linetype="dashed", color = "blue") +
  xlab("Cell Line Lineage + # training datapoints") + ylab("Average MSE Loss") + 
  # scale_y_discrete(limits = c("0.001", "0.002")) +
  scale_y_continuous(breaks = sort(c(seq(0, 0.12, length.out=5),
                                     c(mean(ctrp_data$MSE_loss),
                                       mean(gdsc1_data$MSE_loss),
                                       mean(gdsc2_data$MSE_loss))
  ))) +
  ggtitle(label = "Full DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Proteomics | Trained on CTRPv2 | Tested on All 3")
ggsave(filename = paste0(plot_path, "drug_prot_train_CTRPv2_test_All_avg_MSE_by_cell_line.pdf"), device = "pdf")

# Plot average MSE by lineage Full Response Only + EncoderTrain + PreTrain (drug + exp) ================================
path = "Data/CV_Results/HyperOpt_DRP_ResponseOnly_drug_exp_CTRP_EncoderTrain_PreTrain/"

### GDSC1
cur_data <- fread(paste0(path, "GDSC1_AAC_MORGAN_512_inference_results.csv"))
cur_data <- merge(cur_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

cur_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
avg_mse_by_lineage <- unique(cur_data[, c("lineage", "lineage_loss_avg")])
ggplot(data =  avg_mse_by_lineage)+
  geom_bar(mapping = aes(x = reorder(lineage,-lineage_loss_avg), y = lineage_loss_avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text()) + 
  geom_hline(yintercept = mean(cur_data$MSE_loss), linetype="dashed", color = "red") +
  xlab("Cell Line Lineage") + ylab("Average MSE Loss") +
  ggtitle(label = "ResponseOnly DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Proteomics | Trained on CTRPv2 | Tested on GDSC1")
ggsave(filename = paste0(plot_path, "drug_prot_full_train_CTRPv2_test_GDSC1_avg_MSE_by_lineage.pdf"), device = "pdf")


### GDSC2
cur_data <- fread(paste0(path, "GDSC2_AAC_MORGAN_512_inference_results.csv"))
cur_data <- merge(cur_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

cur_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
avg_mse_by_lineage <- unique(cur_data[, c("lineage", "lineage_loss_avg")])
ggplot(data =  avg_mse_by_lineage)+
  geom_bar(mapping = aes(x = reorder(lineage,-lineage_loss_avg), y = lineage_loss_avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text()) + 
  geom_hline(yintercept = mean(cur_data$MSE_loss), linetype="dashed", color = "red") +
  xlab("Cell Line Lineage") + ylab("Average MSE Loss") +
  ggtitle(label = "Full DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Proteomics | Trained on CTRPv2 | Tested on GDSC2")
ggsave(filename = paste0(plot_path, "drug_prot_full_train_CTRPv2_test_GDSC2_avg_MSE_by_lineage.pdf"), device = "pdf")


### CTRPv2
cur_data <- fread(paste0(path, "CTRP_AAC_MORGAN_512_inference_results.csv"))
cur_data <- merge(cur_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

cur_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
avg_mse_by_lineage <- unique(cur_data[, c("lineage", "lineage_loss_avg")])
ggplot(data =  avg_mse_by_lineage)+
  geom_bar(mapping = aes(x = reorder(lineage,-lineage_loss_avg), y = lineage_loss_avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text()) + 
  geom_hline(yintercept = mean(cur_data$MSE_loss), linetype="dashed", color = "red") +
  xlab("Cell Line Lineage") + ylab("Average MSE Loss") +
  ggtitle(label = "Full DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Proteomics | Trained on CTRPv2 | Tested on CTRPv2")
ggsave(filename = paste0(plot_path, "drug_prot_train_CTRPv2_test_CTRP_avg_MSE_by_lineage.pdf"), device = "pdf")


# Plot average MSE by lineage (drug + exp) ================================
path = "Data/CV_Results/HyperOpt_DRP_FullModel_drug_exp_CTRP_Full/"
# GDSC1
cur_data <- fread(paste0(path, "GDSC1_AAC_MORGAN_512_inference_results.csv"))
cur_data <- merge(cur_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

cur_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
avg_mse_by_lineage <- unique(cur_data[, c("lineage", "lineage_loss_avg")])
ggplot(data =  avg_mse_by_lineage)+
  geom_bar(mapping = aes(x = reorder(lineage,-lineage_loss_avg), y = lineage_loss_avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text()) + 
  geom_hline(yintercept = mean(cur_data$MSE_loss), linetype="dashed", color = "red") +
  xlab("Cell Line Lineage") + ylab("Average MSE Loss") +
  ggtitle(label = "Full DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Gene Expression | Trained on CTRPv2 | Tested on GDSC1")
ggsave(filename = paste0(plot_path, "drug_exp_train_CTRPv2_test_GDSC1_avg_MSE_by_lineage.pdf"), device = "pdf")

# Plot average MSE by lineage Full (drug + exp + prot) ================================
path = "Data/CV_Results/HyperOpt_DRP_FullModel_drug_exp_prot_CTRP_Full/"
### GDSC1 ====
cur_data <- fread(paste0(path, "GDSC1_AAC_MORGAN_512_inference_results.csv"))
cur_data <- merge(cur_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

cur_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
avg_mse_by_lineage <- unique(cur_data[, c("lineage", "lineage_loss_avg")])
ggplot(data =  avg_mse_by_lineage)+
  geom_bar(mapping = aes(x = reorder(lineage,-lineage_loss_avg), y = lineage_loss_avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text()) + 
  geom_hline(yintercept = mean(cur_data$MSE_loss), linetype="dashed", color = "red") +
  xlab("Cell Line Lineage") + ylab("Average MSE Loss") +
  ggtitle(label = "Full DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Gene Expression + Proteomics | Trained on CTRPv2 | Tested on GDSC1")
ggsave(filename = paste0(plot_path, "drug_exp_prot_full_train_CTRPv2_test_GDSC1_avg_MSE_by_lineage.pdf"), device = "pdf")


# GDSC2
cur_data <- fread(paste0(path, "GDSC2_AAC_MORGAN_512_inference_results.csv"))
cur_data <- merge(cur_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

cur_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
avg_mse_by_lineage <- unique(cur_data[, c("lineage", "lineage_loss_avg")])
ggplot(data =  avg_mse_by_lineage)+
  geom_bar(mapping = aes(x = reorder(lineage,-lineage_loss_avg), y = lineage_loss_avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text()) + 
  geom_hline(yintercept = mean(cur_data$MSE_loss), linetype="dashed", color = "red") +
  xlab("Cell Line Lineage") + ylab("Average MSE Loss") +
  ggtitle(label = "Full DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Gene Expression + Proteomics | Trained on CTRPv2 | Tested on GDSC2")
ggsave(filename = paste0(plot_path, "drug_exp_prot_full_train_CTRPv2_test_GDSC2_avg_MSE_by_lineage.pdf"), device = "pdf")

# CTRPv2
cur_data <- fread(paste0(path, "CTRP_AAC_MORGAN_512_inference_results.csv"))
cur_data <- merge(cur_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

cur_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
avg_mse_by_lineage <- unique(cur_data[, c("lineage", "lineage_loss_avg")])
ggplot(data =  avg_mse_by_lineage)+
  geom_bar(mapping = aes(x = reorder(lineage,-lineage_loss_avg), y = lineage_loss_avg), stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text()) + 
  geom_hline(yintercept = mean(cur_data$MSE_loss), linetype="dashed", color = "red") +
  xlab("Cell Line Lineage") + ylab("Average MSE Loss") +
  ggtitle(label = "Full DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Gene Expression + Proteomics | Trained on CTRPv2 | Tested on CTRPv2")
ggsave(filename = paste0(plot_path, "drug_exp_prot_full_train_CTRPv2_test_CTRP_avg_MSE_by_lineage.pdf"), device = "pdf")


### All side by side ====
ctrp_data <- fread(paste0(path, "CTRP_AAC_MORGAN_512_inference_results.csv"))
ctrp_data <- merge(ctrp_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
gdsc1_data <- fread(paste0(path, "GDSC1_AAC_MORGAN_512_inference_results.csv"))
gdsc1_data <- merge(gdsc1_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
gdsc2_data <- fread(paste0(path, "GDSC2_AAC_MORGAN_512_inference_results.csv"))
gdsc2_data <- merge(gdsc2_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

ctrp_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
ctrp_avg_mse_by_lineage <- unique(ctrp_data[, c("lineage", "lineage_loss_avg")])
ctrp_data[, sample_by_lineage_count := .N, by = "lineage"]
ctrp_avg_mse_by_lineage$Dataset <- "CTRPv2"

gdsc1_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
gdsc1_avg_mse_by_lineage <- unique(gdsc1_data[, c("lineage", "lineage_loss_avg")])
gdsc1_avg_mse_by_lineage$Dataset <- "GDSC1"

gdsc2_data[, lineage_loss_avg := mean(MSE_loss), by = "lineage"]
gdsc2_avg_mse_by_lineage <- unique(gdsc2_data[, c("lineage", "lineage_loss_avg")])
gdsc2_avg_mse_by_lineage$Dataset <- "GDSC2"

all_avg_mse_by_lineage <- rbindlist(list(ctrp_avg_mse_by_lineage, gdsc1_avg_mse_by_lineage, gdsc2_avg_mse_by_lineage))
all_avg_mse_by_lineage <- merge(all_avg_mse_by_lineage, unique(ctrp_data[, c("lineage", "sample_by_lineage_count")]))
all_avg_mse_by_lineage$lineage <- paste0(all_avg_mse_by_lineage$lineage, ", n = ", all_avg_mse_by_lineage$sample_by_lineage_count)

ggplot(data = all_avg_mse_by_lineage) +
  geom_bar(mapping = aes(x = reorder(lineage, -lineage_loss_avg), y = lineage_loss_avg, fill = Dataset), stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  geom_hline(yintercept = mean(ctrp_data$MSE_loss), linetype="dashed", color = "red") +
  # geom_text(aes(10, mean(ctrp_data$MSE_loss),label = mean(ctrp_data$MSE_loss), vjust = -1)) +
  geom_hline(yintercept = mean(gdsc1_data$MSE_loss), linetype="dashed", color = "green") +
  geom_hline(yintercept = mean(gdsc2_data$MSE_loss), linetype="dashed", color = "blue") +
  xlab("Cell Line Lineage  + # training datapoints") + ylab("Average MSE Loss") + 
  # scale_y_discrete(limits = c("0.001", "0.002")) +
  scale_y_continuous(breaks = sort(c(seq(0, 0.12, length.out=5),
                                     c(mean(ctrp_data$MSE_loss),
                                       mean(gdsc1_data$MSE_loss),
                                       mean(gdsc2_data$MSE_loss))
  ))) +
  ggtitle(label = "Full DRP Mean MSE Loss by Cell Line Lineage", subtitle = "Data: Drug + Expression + Proteomics | Trained on CTRPv2 | Tested on All 3")
ggsave(filename = paste0(plot_path, "drug_exp_prot_train_CTRPv2_test_All_avg_MSE_by_lineage.pdf"), device = "pdf")

