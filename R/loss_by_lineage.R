# loss_by_lineage.R
require(data.table)
require(ggplot2)


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
