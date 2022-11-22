# DR_Data_Exploration.R
require(ggplot2)
require(data.table)


# ==== CTRPv2 vs GDSC1/2 ====
ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")
gdsc2 <- fread("Data/DRP_Training_Data/GDSC2_AAC_SMILES.txt")
gdsc1 <- fread("Data/DRP_Training_Data/GDSC1_AAC_SMILES.txt")

length(unique(ctrp$ccl_name))

mean(ctrp$area_above_curve)
max(ctrp$area_above_curve)
min(ctrp$area_above_curve)
quantile(ctrp$area_above_curve)
hist(ctrp$area_above_curve)
# Mean by drug
ctrp[ , mean_by_drug := mean(area_above_curve), by = "cpd_name"]
ctrp[ , mean_by_cell := mean(area_above_curve), by = "ccl_name"]
ctrp[, Dataset := "CTRPv2"]
gdsc2[ , mean_by_drug := mean(area_above_curve), by = "cpd_name"]
gdsc2[ , mean_by_cell := mean(area_above_curve), by = "ccl_name"]
gdsc2[, Dataset := "GDSC2"]
gdsc1[ , mean_by_drug := mean(area_above_curve), by = "cpd_name"]
gdsc1[ , mean_by_cell := mean(area_above_curve), by = "ccl_name"]
gdsc1[, Dataset := "GDSC1"]
# hist(unique(ctrp$mean_by_drug))
all <- rbindlist(list(ctrp, gdsc1, gdsc2), fill = T)

ggplot(all, aes(x = area_above_curve, colour = Dataset)) +
  # geom_density(bins=100) +
  geom_freqpoly(bins=100) +
  geom_vline(aes(xintercept = round(mean(ctrp$area_above_curve), 3)), color="black", linetype="dashed", size=1) +
  geom_vline(aes(xintercept = round(median(ctrp$area_above_curve), 3)), color="black", linetype="dashed", size=1) +
  scale_x_continuous(breaks=c(0,0.25, 0.5, 0.75, 1),
                     minor_breaks = c(round(median(ctrp$area_above_curve), 3),
                     round(mean(ctrp$area_above_curve), 3))) +
  annotate(x=round(mean(ctrp$area_above_curve), 3) - 0.03, y=30000,label="CTRPv2 Mean",vjust=1.5,geom="text", angle = 90) +
  annotate(x=round(median(ctrp$area_above_curve), 3) - 0.03, y=30000,label="CTRPv2 Median",vjust=1.5,geom="text", angle = 90) +
  # ggtitle(label = "Area Above Curve Frequency Polygon for CTRP and GDSC") +
  xlab("Area Above Curve") + ylab("Count") +
  theme(text = element_text(size = 18, face = "bold"),
        legend.position = "top",
        axis.text.x = element_text(angle = 45, hjust = 1))

dir.create("Plots/Dataset_Exploration")
# ggsave(filename = "Plots/Dataset_Exploration/CTRPv2_AAC_Distribution.pdf")
ggsave(filename = "Plots/Dataset_Exploration/CTRPv2_GDSC_AAC_Distribution.pdf")

# ==== GDSC2 ====
gdsc2 <- fread("Data/DRP_Training_Data/GDSC2_AAC_SMILES.txt")
ggplot(gdsc2, aes(x = area_above_curve)) +
  geom_density() +
  geom_vline(aes(xintercept = mean(area_above_curve)), color="blue", linetype="dashed", size=1) +
  geom_vline(aes(xintercept = median(area_above_curve)), color="blue", linetype="dashed", size=1) +
  scale_x_continuous(breaks=c(0, round(median(gdsc2$area_above_curve), 3), round(mean(gdsc2$area_above_curve), 3), 0.25, 0.5, 0.75, 1)) +
  annotate(x=mean(gdsc2$area_above_curve), y=6,label="Mean",vjust=1.5,geom="text", angle = 90) + 
  annotate(x=median(gdsc2$area_above_curve), y=6,label="Median",vjust=1.5,geom="text", angle = 90) + 
  ggtitle(label = "Area Above Curve Distribution for GDSC2", subtitle = paste0("Dataset Length: ", nrow(gdsc2))) +
  xlab("Area Above Curve") + ylab("Density")
ggsave(filename = "Plots/Dataset_Exploration/GDSC2_AAC_Distribution.pdf")

# ==== CTRPv2 AAC Distrubtion vs LDS weight Distribution ====
# install.packages("cowplot")
require(cowplot)
lds <- fread("ctrp_targets_lds_weights.csv")
# lds$Weighted <- lds$targets * lds$loss_weights
lds$V1 <- NULL
# lds$loss_weights <- NULL
long_lds <- melt(lds)

p_targets <- ggplot(lds, aes(x=targets)) +
  geom_freqpoly(bins=100) +
  xlab("True AAC") + ylab("Count")
  # theme(axis.title.x = element_text()"True AAC", axis.title.y = "Count")
  
p_lds_weights <- ggplot(lds) +
  geom_line(aes(x = targets, y = loss_weights)) +
  xlab("True AAC") + ylab("LDS Weight")
  # theme(axis.title.x = "True AAC", axis.title.y = "Weighted Loss Factor")

cowplot::plot_grid(p_targets, p_lds_weights, ncol = 1)

ggsave("Plots/Dataset_Exploration/CTRPv2_AAC_vs_LDS_Weights.pdf")

# Update (March 2022) ====
# Add the 0.7 cutoff to this plot

p_targets <- ggplot(lds, aes(x=targets)) +
  geom_freqpoly(bins=100) +
  xlab("True AAC") + ylab("Count") +
  geom_vline(xintercept = 0.5, linetype = "dotted") +
  scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.7, 0.75, 1))
# theme(axis.title.x = element_text()"True AAC", axis.title.y = "Count")

p_lds_weights <- ggplot(lds) +
  geom_line(aes(x = targets, y = loss_weights)) +
  xlab("True AAC") + ylab("LDS Weight") +
  geom_vline(xintercept = 0.7, linetype = "dotted") +
  scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.7, 0.75, 1)) +


# theme(axis.title.x = "True AAC", axis.title.y = "Weighted Loss Factor")

cowplot::plot_grid(p_targets, p_lds_weights, ncol = 1)

ggsave("Plots/Dataset_Exploration/CTRPv2_AAC_vs_LDS_Weights_With_0.7_Cutoff.pdf")

ctrp[, log_aac := log(area_above_curve + 1), by = c("cpd_name", 'ccl_name')]
ctrp_long <- melt(ctrp, measure.vars = c("area_above_curve", "log_aac"))

ctrp[, sqrt_aac := sqrt(area_above_curve), by = c("cpd_name", 'ccl_name')]
ctrp_long <- melt(ctrp, measure.vars = c("area_above_curve", "sqrt_aac"))
ctrp_long <- melt(ctrp, measure.vars = c("area_above_curve", "sqrt_aac", "log_aac"))

require(e1071)
skewness(ctrp$area_above_curve)  # 1.779
skewness(ctrp$log_aac)  # 1.381
skewness(ctrp$sqrt_aac)  # 0.4231

ggplot(ctrp_long, aes(x = value, group = variable, fill = variable)) +
  geom_density(alpha=.2) +
  geom_histogram(alpha=0.4) + 
  facet_wrap(~variable) + 
  # geom_vline(aes(xintercept = mean(value)), color="blue", linetype="dashed", size=1) +
  # geom_vline(aes(xintercept = median(value)), color="blue", linetype="dashed", size=1) +
  # scale_x_continuous(breaks=c(0, round(median(ctrp$value), 3), round(mean(ctrp$area_above_curve), 3), 0.25, 0.5, 0.75, 1)) +
  # annotate(x=mean(ctrp$area_above_curve), y=6,label="Mean",vjust=1.5,geom="text", angle = 90) + 
  # annotate(x=median(ctrp$area_above_curve), y=6,label="Median",vjust=1.5,geom="text", angle = 90) + 
  ggtitle(label = "Area Above Curve Distribution for CTRPv2", subtitle = paste0("Dataset Length: ", nrow(ctrp))) +
  xlab("Area Above Curve") + ylab("Density")

dir.create("Plots/Dataset_Exploration")
ggsave(filename = "Plots/Dataset_Exploration/CTRPv2_AAC_Sqrt_Log_Distribution.pdf")


# ==== LDS vs no-LDS Bi-modal, LMF, GNN (gnndrug + prot) ====
require(data.table)
all_csv_results <- list.files("Data/CV_Results/", "CTRP_AAC_SMILES_inference_results.csv", recursive = T, full.names = T)
bimodal_results <- grep(pattern = ".+_gnndrug_.{3,5}_HyperOpt.+CTRP.+MergeByLMF.+", x = all_csv_results, value = T)
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

upper_results <- all_results[target >= 0.8 & predicted >= 0.8]
long_results <- melt(upper_results[, c("target", "predicted", "loss_type", "data_types")],
                     id.vars = c("loss_type", "data_types"))
# upper_results[, sum_point_two := sum(RMSELoss <= 0.22), by = "loss_type"]
upper_results[, count := nrow(.SD), by = c("loss_type", "data_types")]
melt_upper <- unique(upper_results[, c("loss_type", "data_types", "count")])


ggplot(melt_upper, aes(x = data_types, y = count, fill = loss_type)) +
  # geom_density(alpha=.2) +
  # geom_freqpoly() +
  geom_bar(stat = "identity", position = "dodge") +
  # theme(legend.title = element_text("Loss Type")) +
  scale_fill_discrete(name = "Loss Type") +
  # geom_histogram(alpha=0.4) + 
  # facet_wrap(~loss_type) + 
  ggtitle(label = "CTRPv2: Number of recovered samples for AAC Target & Predictions >= 0.8 ",
          subtitle = "GNN for drugs and LMF for fusion") +
  xlab("Omic Data Type") + ylab("Counts")
ggsave(filename = "Plots/LDS/CTRPv2_AAC_bimodal_thresh_0.8_upper_recovery.pdf")


ggplot(long_results, aes(x = value, group = variable, colour = loss_type)) +
  # geom_density(alpha=.2) +
  geom_freqpoly() +
  # geom_histogram(alpha=0.4) + 
  facet_wrap(~variable+loss_type+data_types) + 
  ggtitle(label = "AAC Density in CTRPv2 for Target >= 0.7 and Prediction >= 0.6", subtitle = "gnndrug_exp + LMF | ~ %15 better recovery of AAC targets >= 0.7") +
  xlab("Area Above Curve") + ylab("Density")
dir.create("Plots/LDS")
ggsave(filename = "Plots/LDS/CTRPv2_AAC_target_0.7_predict_0.6_gnndrug_prot_upper_recovery.pdf")


# ==== CTRP True vs Predicted AAC Distribution ====
dir.create("Plots/Dataset_Exploration/PredictedvsTarget_AAC_Distribution")
dir.create("Plots/Dataset_Exploration/PredictedvsTarget_AAC_Distribution/Split_by_Cell")
dir.create("Plots/Dataset_Exploration/PredictedvsTarget_AAC_Distribution/Split_by_Drug")
dir.create("Plots/Dataset_Exploration/PredictedvsTarget_AAC_Distribution/Split_by_Both")

options("scipen"=100, "digits"=4)
rsq <- function (x, y) cor(x, y) ^ 2
rsq(ctrp_morgan$target[1:1000], ctrp_morgan$predicted[1:1000])
rsq(ctrp_morgan$target, ctrp_morgan$predicted)
rsq(ctrp_onehot$target[1:1000], ctrp_onehot$predicted[1:1000])
mean(ctrp_morgan$RMSE_loss[1:1000])
mean(ctrp_morgan$RMSE_loss)
mean(ctrp_onehot$RMSE_loss[1:1000])

mean(ctrp_onehot[target > 0.2]$RMSE_loss)  # 0.1639
mean(ctrp_onehot[target < 0.2]$RMSE_loss)  # 0.009007

mean(ctrp_nontransform$RMSE_loss)
mean(ctrp_nontransform[target > 0.2]$RMSE_loss)
mean(ctrp_nontransform[target < 0.2]$RMSE_loss)
max(ctrp_nontransform$predicted)
min(ctrp_nontransform$predicted)

# morgan_path <- "HyperOpt_DRP_ResponseOnly_drug_exp_HyperOpt_DRP_CTRP_1024_ResponseOnly_EncoderTrain_Split_DRUG_NoBottleNeck_NoTCGAPretrain_MergeBySum_RMSELoss_MorganDrugs_SqrtTransform_drug_exp/"
sqrt_morgan_path <- "Data/CV_Results/HyperOpt_DRP_ResponseOnly_drug_exp_HyperOpt_DRP_CTRP_1024_ResponseOnly_EncoderTrain_Split_DRUG_NoBottleNeck_NoTCGAPretrain_MergeBySum_RMSELoss_MorganDrugs_SqrtTransform_drug_exp/"
morgan_only_path <- "Data/CV_Results/HyperOpt_DRP_ResponseOnly_drug_HyperOpt_DRP_CTRP_1024_ResponseOnly_EncoderTrain_Split_DRUG_NoBottleNeck_NoTCGAPretrain_MergeBySum_RMSELoss_MorganDrugs_drug/"
log_morgan_path <- "Data/CV_Results/HyperOpt_DRP_ResponseOnly_drug_exp_HyperOpt_DRP_CTRP_1024_ResponseOnly_EncoderTrain_Split_DRUG_NoBottleNeck_NoTCGAPretrain_MergeBySum_RMSELoss_MorganDrugs_LogTransform_drug_exp/"
nontransform_morgan_path <- "Data/CV_Results/HyperOpt_DRP_ResponseOnly_drug_exp_HyperOpt_DRP_CTRP_1024_ResponseOnly_EncoderTrain_Split_DRUG_NoBottleNeck_NoTCGAPretrain_MergeBySum_RMSELoss_MorganDrugs_drug_exp/"
nontransform_onehot_path <- "Data/CV_Results/HyperOpt_DRP_ResponseOnly_drug_exp_HyperOpt_DRP_CTRP_1024_ResponseOnly_EncoderTrain_Split_DRUG_NoBottleNeck_NoTCGAPretrain_MergeBySum_RMSELoss_OneHotDrugs_drug_exp/"
plot_aac_dist <- function(morgan_path, onehot_path, plot_path, cell_line_data, title, subtitle, plot_filename) {
  ctrp_sqrt_transfrom <- fread(paste0(sqrt_morgan_path, "CTRP_AAC_MORGAN_1024_inference_results.csv"))
  ctrp_sqrt_transfrom$transform_type <- "SqrtTransform"
  
  ctrp_morgan_only <- fread(paste0(morgan_only_path, "CTRP_AAC_MORGAN_1024_inference_results.csv"))
  ctrp_morgan_only[, pred_var_per_cell := sd(predicted), by = "cell_name"]
  ctrp_morgan_only[, pred_var_per_drug := sd(predicted), by = "cpd_name"]
  ctrp_morgan_only[, targ_var_per_cell := sd(target), by = "cell_name"]
  ctrp_morgan_only[, targ_var_per_drug := sd(target), by = "cpd_name"]
  ctrp_morgan_only$transform_type <- "MorganOnly"
  
  min(ctrp_morgan_only$predicted)
  max(ctrp_morgan_only$predicted)
  
  ctrp_nontransform <- fread(paste0(nontransform_morgan_path, "CTRP_AAC_MORGAN_1024_inference_results.csv"))
  ctrp_nontransform[, pred_var_per_cell := sd(predicted), by = "cell_name"]
  ctrp_nontransform[, pred_var_per_drug := sd(predicted), by = "cpd_name"]
  ctrp_nontransform[, targ_var_per_cell := sd(target), by = "cell_name"]
  ctrp_nontransform[, targ_var_per_drug := sd(target), by = "cpd_name"]
  ctrp_nontransform$transform_type <- "NoTransform"
  cor(ctrp_nontransform$predicted, ctrp_nontransform$target, method = 'pearson')
  
  ctrp_nontran_onehot <- fread(paste0(nontransform_onehot_path, "CTRP_AAC_MORGAN_1024_inference_results.csv"))
  cor.test(ctrp_nontran_onehot$cell_name, ctrp_nontran_onehot$predicted, method = "kendall")
  pred_aov <- aov(predicted ~ cell_name, ctrp_nontran_onehot)
  summary(pred_aov)
  target_aov <- aov(target ~ cell_name, ctrp_nontran_onehot)
  summary(target_aov)
  drug_pred_aov <- aov(target ~ cpd_name, ctrp_nontran_onehot)
  summary(drug_pred_aov)
  
  ctrp_log_transfrom <- fread(paste0(log_morgan_path, "CTRP_AAC_MORGAN_1024_inference_results.csv"))
  ctrp_log_transfrom$transform_type <- "LogTransform"
  
  all_ctrp <- rbindlist(list(ctrp_sqrt_transfrom, ctrp_nontransform, ctrp_log_transfrom, ctrp_morgan_only))
  
  ggplot(all_ctrp) +
    # geom_bar(aes(x = target, y = RMSE_loss), position = "identity", width = 0.01, alpha=0.2) +
    stat_summary_bin(aes(x = target, y = RMSE_loss), fun = "mean", colour = "red", geom = "bar", size = 1, binwidth = 0.01)
  ggplot(all_ctrp) +
    # geom_density(aes(x = predicted, group = drug_type, fill = drug_type), alpha=.2) +
    geom_freqpoly(aes(x = predicted, group = transform_type, color = transform_type), binwidth = 0.01) +
    # geom_density(alpha = 0.2, aes(x = target, fill = "Targets")) + 
    geom_freqpoly(aes(x = target, color = "Targets"), binwidth = 0.01) + 
    stat_summary_bin(aes(x = target, y = RMSE_loss*100000), fun = "mean", geom = "bar", size = 1, binwidth = 0.01, alpha = 0.2) +
    facet_wrap(~transform_type) +
    ggtitle(label = title,
            subtitle = subtitle) +
    xlab("Area Above Curve") + ylab("Density")
  ggsave(filename = paste0(plot_path, plot_filename))
}

plot_filename = "ResponseOnly_drug_exp_train_CTRPv2_AAC_Distribution_RMSE_SplitByDrug_NoBottleNeck_Transform_Comparison.pdf"
plot_grid_mono <- function(model_type, data_type, split, bottleneck) {
  morgan_path <- paste0("Data/CV_Results/HyperOpt_DRP_", model_type, "_drug", data_type,
                 "_HyperOpt_DRP_CTRP_1024_", model_type, "_EncoderTrain_Split_", split, "_", bottleneck, "_NoTCGAPretrain_MergeBySum_RMSELoss_MorganDrugs_drug", data_type, "/")
  onehot_path <- paste0("Data/CV_Results/HyperOpt_DRP_", model_type, "_drug", data_type,
                 "_HyperOpt_DRP_CTRP_1024_", model_type, "_EncoderTrain_Split_", split, "_", bottleneck, "_NoTCGAPretrain_MergeBySum_RMSELoss_OneHotDrugs_drug", data_type, "/")
  
  if (split == "CELL_LINE") {
    plot_path <- "Plots/Dataset_Exploration/PredictedvsTarget_AAC_Distribution/Split_by_Cell/"
    plot_split_name <- "SplitByCell"
    title_split_name <- "Cell Line"
  } else if (split == "DRUG") {
    plot_path <- "Plots/Dataset_Exploration/PredictedvsTarget_AAC_Distribution/Split_by_Drug/"
    plot_split_name <- "SplitByDrug"
    title_split_name <- "Drug"
    
  } else {
    plot_path <- "Plots/Dataset_Exploration/PredictedvsTarget_AAC_Distribution/Split_by_Both/"
    plot_split_name <- "SplitByBoth"
    title_split_name <- "Cell Line & Drug"
    
  }
  if (bottleneck == "WithBottleNeck") {
    subtitle_bottleneck_name <- "With Bottleneck"
    
  } else  {
    subtitle_bottleneck_name <- "No Bottleneck"
  }
  
  dir.create(plot_path)
  plot_filename <- paste0(model_type, "_drug", data_type, "_train_CTRPv2_AAC_Distribution_RMSE_", plot_split_name, "_", bottleneck, ".pdf")
  title <- paste0("Predicted vs Target AAC Distribution for CTRPv2 | CV ", title_split_name, " Splitting")
  subtitle <- paste0("Model Type: ", model_type, " | Data: Drug + ", gsub("_", "", toupper(data_type)), " | Loss x 1e5 bins")
  plot_aac_dist(morgan_path = morgan_path, onehot_path = onehot_path, plot_path = plot_path, title = title, subtitle = subtitle, plot_filename = plot_filename)
  
}

# "Predicted vs Target AAC Distribution for CTRPv2 | Data Types: Drug + Exp"
# paste0("Dataset Length: ", nrow(all_ctrp_exp)/2, " | Model Type: ResponseOnly | CV Split by Drug")
# "Plots/Dataset_Exploration/CTRPv2_AAC_Distribution_Response_Only_Pred_vs_Target_drug_exp.pdf"
model_types <- c("FullModel", "ResponseOnly")
data_types <- c("mut", "exp", "prot", "mirna", "metab", "rppa", "hist")
data_types <- paste0("_", data_types)
data_types <- c("", data_types)
# splits <- c("CELL_LINE", "DRUG", "BOTH")
splits <- c("DRUG")
# bottlenecking <- c("WithBottleNeck", "NoBottleNeck")
bottlenecking <- c("NoBottleNeck")
# drug_types <- c("OneHotDrugs")
grid <- expand.grid(model_types, data_types, splits, bottlenecking)

for (i in 1:nrow(grid)) {
  plot_grid_mono(model_type = grid[i, 1], data_type = grid[i, 2], split = grid[i, 3], bottleneck = grid[i, 4])
}

# ==== GDSC2 ====
gdsc2 <- fread("Data/DRP_Training_Data/GDSC2_AAC_SMILES.txt")
ggplot(gdsc2, aes(x = area_above_curve)) +
  geom_density() +
  geom_vline(aes(xintercept = mean(area_above_curve)), color="blue", linetype="dashed", size=1) +
  geom_vline(aes(xintercept = median(area_above_curve)), color="blue", linetype="dashed", size=1) +
  scale_x_continuous(breaks=c(0, round(median(gdsc2$area_above_curve), 3), round(mean(gdsc2$area_above_curve), 3), 0.25, 0.5, 0.75, 1)) +
  annotate(x=mean(gdsc2$area_above_curve), y=6,label="Mean",vjust=1.5,geom="text", angle = 90) + 
  annotate(x=median(gdsc2$area_above_curve), y=6,label="Median",vjust=1.5,geom="text", angle = 90) + 
  ggtitle(label = "Area Above Curve Distribution for GDSC2", subtitle = paste0("Dataset Length: ", nrow(gdsc2))) +
  xlab("Area Above Curve") + ylab("Density")
ggsave(filename = "Plots/Dataset_Exploration/GDSC2_AAC_Distribution.pdf")

# ==== GDSC1 ====
gdsc1 <- fread("Data/DRP_Training_Data/GDSC1_AAC_SMILES.txt")
ggplot(gdsc1, aes(x = area_above_curve)) +
  geom_density() +
  geom_vline(aes(xintercept = mean(area_above_curve)), color="blue", linetype="dashed", size=1) +
  geom_vline(aes(xintercept = median(area_above_curve)), color="blue", linetype="dashed", size=1) +
  scale_x_continuous(breaks=c(0, round(median(gdsc1$area_above_curve), 3), round(mean(gdsc1$area_above_curve), 3), 0.25, 0.5, 0.75, 1)) +
  annotate(x=mean(gdsc1$area_above_curve), y=6,label="Mean",vjust=1.5,geom="text", angle = 90) + 
  annotate(x=median(gdsc1$area_above_curve), y=6,label="Median",vjust=1.5,geom="text", angle = 90) + 
  ggtitle(label = "Area Above Curve Distribution for GDSC1", subtitle = paste0("Dataset Length: ", nrow(gdsc2))) +
  xlab("Area Above Curve") + ylab("Density")
ggsave(filename = "Plots/Dataset_Exploration/GDSC1_AAC_Distribution.pdf")

# TODO GEOM Density log distribution

