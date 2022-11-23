# interpretation_analysis.R
require(data.table)
setDTthreads(8)
require(ggfortify)
require(umap)
exp_path = "Data/CV_Results/HyperOpt_DRP_ResponseOnly_gnndrug_exp_HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_exp/"
prot_path = "Data/CV_Results/HyperOpt_DRP_ResponseOnly_gnndrug_prot_HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_BOTH_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_prot/"
dir.create("Plots/DRP")
dir.create("Plots/DRP/Lineage_Results")

targeted_drugs <- c("Idelalisib", "Olaparib", "Venetoclax", "Crizotinib", "Regorafenib", 
                    "Tretinoin", "Bortezomib", "Cabozantinib", "Dasatinib", "Erlotinib", 
                    "Sonidegib", "Vandetanib", "Axitinib", "Ibrutinib", "Gefitinib", 
                    "Nilotinib", "Tamoxifen", "Bosutinib", "Pazopanib", "Lapatinib", 
                    "Dabrafenib", "Bexarotene", "Temsirolimus", "Belinostat", 
                    "Sunitinib", "Vorinostat", "Trametinib", "Fulvestrant", "Sorafenib", 
                    "Vemurafenib", "Alpelisib")

ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")

length(targeted_drugs)

length(unique(ctrp[cpd_name %in% targeted_drugs]$ccl_name))  # 842 cell lines tested with targeted drugs
length(unique(ctrp[cpd_name %in% targeted_drugs & area_above_curve >= 0.7]$ccl_name))  # 302 of them with AAC >= 0.7
nrow(unique(ctrp[cpd_name %in% targeted_drugs & area_above_curve >= 0.7]))  # resulting in 395 potential samples
# Load cell line and interpretation results ===============================
exp_data <- fread(paste0(exp_path, "integrated_gradients_results.csv"))
prot_data <- fread(paste0(prot_path, "integrated_gradients_results.csv"))
cell_line_data <- fread("Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv")
# cur_data <- fread(paste0(path, "GDSC2_AAC_MORGAN_512_inference_results.csv"))
# cur_cv <- fread(paste0(path, "CV_results.csv"))

dim(cur_data)
exp_data[1:5, 1:10]
max(exp_data[1:5, -c(1:7)])
exp_data[RMSE_loss < 0.2][1:5, 1:5]
prot_data[RMSE_loss < 0.1][1:5, 1:5]
prot_data[RMSE_loss < 0.1][, 1:6]
cur_data$RMSE_loss[1]
cur_data$DeepLIFT_delta[1]

exp_data <- merge(exp_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")
prot_data <- merge(prot_data, cell_line_data[, c("stripped_cell_line_name", "lineage")], by.x = "cell_name", by.y = "stripped_cell_line_name")

unique(prot_data$cpd_name)
# cur_data[, total_drug_attrib := sum(.SD), .SDcols = drug_cols, by = ]
setcolorder(prot_data, 'lineage')
prot_data[lineage == "lung" & RMSE_loss < 0.3][, 1:6]
prot_data[lineage == "lung"][, 1:6]
setcolorder(exp_data, 'lineage')

cur_data[1:5, 1:10]
exp_data$V1 <- NULL
prot_data$V1 <- NULL
# drug_cols = colnames(cur_data)[7:518]
prot_cols = colnames(prot_data)[8:ncol(prot_data)]
exp_cols = colnames(exp_data)[8:ncol(exp_data)]
col_types <- sapply(cur_data[, ..prot_cols], class)
which(col_types == "character")
unique(col_types)

mean(exp_data$RMSE_loss)
mean(prot_data$RMSE_loss)
# cur_data[lineage %like% 'blood'][, 1:8]
exp_data[cpd_name %like% 'Paclitaxel' & target >= 0.9 & RMSE_loss <= 0.1][, 1:8]
prot_data[cpd_name %like% 'Paclitaxel' & target >= 0.9 & RMSE_loss <= 0.1][, 1:8]
exp_data[cpd_name %like% 'Paclitaxel' & target >= 0.9 & RMSE_loss <= 0.1]$cell_name %in% prot_data[cpd_name %like% 'Paclitaxel' & target >= 0.9 & RMSE_loss <= 0.1]$cell_name
cur_data[cpd_name %like% 'Paclitaxel' & cell_name == "GA10"][, 1:8]


# Distinguish positive and negative attributions
exp_temp <- exp_data[cpd_name %like% 'Paclitaxel' & cell_name == "697"]
prot_temp <- prot_data[cpd_name %like% 'Paclitaxel' & cell_name == "697"]

# temp <- cur_data[RMSE_loss <= 0.1]
# temp[1:100, 1:8]

prot_temp <- melt(prot_temp[1, ..prot_cols])
prot_pos_temp <- prot_temp[value > 0]
top_10 <- quantile(prot_pos_temp$value, 0.9)
quantile(prot_pos_temp$value)
quantile(prot_pos_temp$value)[4]  # %75

# Top Prots
prot_pos_temp[value > quantile(prot_pos_temp$value)[4]]
prot_top_5 <- prot_pos_temp[value > quantile(prot_pos_temp$value, 0.95)]
setorder(prot_top_5, -value)
prot_top_5$variable <- gsub("prot_", "", prot_top_5$variable)
top_5_prots <- setNames(prot_top_5$value, prot_top_5$variable)

# Bottom Prots
# temp <- melt(temp[1, ..prot_cols])
neg_temp <- prot_temp[value < 0]
bottom_5 <- neg_temp[value < quantile(neg_temp$value, 0.05)]
bottom_5$value <- abs(bottom_5$value)
setorder(bottom_5, -value)
bottom_5$variable <- gsub("prot_", "", bottom_5$variable)
bottom_5_prots <- setNames(bottom_5$value, bottom_5$variable)

prot_top_5[variable %like% "Q02548"] # PAX5 for leukemia
prot_temp[variable %like% "Q02548"]
prot_temp[variable %like% "PAX5"]
prot_temp[variable %like% "NBN"]
prot_temp[variable %like% "GNB1"]
prot_temp[variable %like% "FLT3"]
prot_temp[variable %like% "ETV6"]
prot_temp[variable %like% "ACTB"]

prot_top_5[variable %like% "FLT3"]
prot_top_5[variable %like% "PAX5"]
prot_top_5[variable %like% "NBN"]
prot_top_5[variable %like% "GNB1"]
prot_top_5[variable %like% "ETV6"]
prot_top_5[variable %like% "ACTB"]

exp_temp <- melt(exp_temp[1, ..exp_cols])
exp_pos_temp <- exp_temp[value > 0]
top_10 <- quantile(exp_pos_temp$value, 0.9)
quantile(exp_pos_temp$value)
quantile(exp_pos_temp$value)[4]  # %75
exp_pos_temp[value > quantile(exp_pos_temp$value)[4]]
exp_top_5 <- exp_pos_temp[value > quantile(exp_pos_temp$value, 0.95)]
setorder(exp_top_5, -value)
exp_top_5$variable <- gsub("exp_", "", exp_top_5$variable)
top_5_exps <- setNames(exp_top_5$value, exp_top_5$variable)

exp_temp[variable %like% "PAX5"]
exp_temp[variable %like% "NBN"]
exp_temp[variable %like% "GNB1"]
exp_temp[variable %like% "FLT3"]
exp_temp[variable %like% "ETV6"]
exp_temp[variable %like% "ACTB"]

exp_top_5[variable %like% "FLT3"]
exp_top_5[variable %like% "PAX5"]
exp_top_5[variable %like% "NBN"]
exp_top_5[variable %like% "GNB1"]
exp_top_5[variable %like% "ETV6"]
exp_top_5[variable %like% "ACTB"]

exp_top_5[1:10,]
temp <- cur_data[RMSE_loss <= 0.1]

temp[1, 1:8]
temp[variable %like% "XPO1"]

temp <- melt(temp[1, ..prot_cols])
neg_temp <- temp[value < 0]
bottom_5 <- neg_temp[value < quantile(neg_temp$value, 0.05)]
bottom_5$value <- abs(bottom_5$value)
setorder(bottom_5, -value)
bottom_5$variable <- gsub("exp_", "", bottom_5$variable)
bottom_5_prots <- setNames(bottom_5$value, bottom_5$variable)

# ==== clusterProfiler ====
# BiocManager::install("clusterProfiler")
# BiocManager::install("pathview")
# BiocManager::install("enrichplot")

require(clusterProfiler)
require(pathview)
organism = "org.Hs.eg.db"
# BiocManager::install(organism, character.only = TRUE)
library(organism, character.only = TRUE)
keytypes(get(organism))
# org.Hs.eg.db
top_gse_prot <- gseGO(geneList=top_5_prots, 
             ont ="ALL", 
             keyType = "UNIPROT", 
             # nPerm = 10000,
             minGSSize = 3, 
             maxGSSize = 800, 
             pvalueCutoff = 0.05, 
             verbose = TRUE, 
             OrgDb = get(organism), 
             pAdjustMethod = "none")
p_top_prot <- ridgeplot(top_gse_prot) + labs(x = "enrichment distribution") + ggtitle("Top 5% Protein Attributions GSE",
                                                                            subtitle = "Cell-line 697 (lymphoblastic leukemia) + Paclitaxel\nTarget: 0.97, Predicted: 0.94")
ggsave("Plots/Interpretation/IntegratedGradients/GSE/gnndrug_prot_697_Paclitaxel_GSE_top_5.pdf", p_top_prot, 
       width = 10, units = "in")


bottom_gse_prot <- gseGO(geneList=bottom_5_prots, 
                  ont ="ALL", 
                  keyType = "UNIPROT", 
                  # nPerm = 10000,
                  minGSSize = 3, 
                  maxGSSize = 800, 
                  pvalueCutoff = 0.05, 
                  verbose = TRUE, 
                  OrgDb = get(organism), 
                  pAdjustMethod = "none")
p_bottom_prot <- ridgeplot(bottom_gse_prot) + labs(x = "enrichment distribution") + ggtitle("Bottom 5% Protein Attributions GSE",
                                                                                  subtitle = "Cell-line 697 (lymphoblastic leukemia) + Paclitaxel\nTarget: 0.97, Predicted: 0.94")
ggsave("Plots/Interpretation/IntegratedGradients/GSE/gnndrug_prot_697_Paclitaxel_GSE_bottom_5.pdf", p_bottom_prot, 
       width = 20, units = "in")

require(cowplot)
cowplot::plot_grid(p_top_prot, p_bottom_prot, ncol = 2)
dir.create("Plots/Interpretation")
dir.create("Plots/Interpretation/IntegratedGradients")
dir.create("Plots/Interpretation/IntegratedGradients/GSE")
ggsave("Plots/Interpretation/IntegratedGradients/GSE/gnndrug_prot_697_Paclitaxel_GSE.pdf", width = 20, units = "in")


gse_exp <- gseGO(geneList=top_5_exps, 
                 ont ="ALL", 
                 keyType = "SYMBOL", 
                 nPerm = 10000,
                 minGSSize = 3, 
                 maxGSSize = 800, 
                 pvalueCutoff = 0.05, 
                 verbose = TRUE, 
                 OrgDb = get(organism), 
                 pAdjustMethod = "none")
p_top_exp <- ridgeplot(gse_exp) + labs(x = "enrichment distribution") + ggtitle("Top 5% RNA-Seq Attributions GSE")

require(cowplot)
cowplot::plot_grid(p_top_prot, p_top_exp, ncol = 2)
dir.create("Plots/Interpretation")
dir.create("Plots/Interpretation/IntegratedGradients")
dir.create("Plots/Interpretation/IntegratedGradients/GSE")
ggsave("Plots/Interpretation/IntegratedGradients/GSE/gnndrug_exp_5637_leptomycin_b_GSE.pdf", width = 20, units = "in")
ggsave("Plots/Interpretation/IntegratedGradients/GSE/gnndrug_exp_vs_prot_697_paclitaxel_GSE.pdf", width = 20, units = "in")
max(cur_data$MSE_loss)
min(cur_data$MSE_loss)
mean(cur_data$MSE_loss)
quantile(cur_data$MSE_loss)

# Which lineages are easier to learn compared to others?
easy_samples <- cur_data[MSE_loss < quantile(cur_data$MSE_loss)[4]][, 1:6]
hard_samples <- cur_data[MSE_loss > quantile(cur_data$MSE_loss)[4]][, 1:6]
easy_samples$type <- "easy"
hard_samples$type <- "hard"
easy_hard <- rbindlist(list(easy_samples[, c("lineage", "type")], hard_samples[, c("lineage", "type")]))
ggplot(data = easy_hard) + geom_bar(mapping = aes(x = lineage, fill = type), stat = "count", position = "stack") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# easy_hard <- within(easy_hard, type <- factor(type, levels = names(sort(table(type), decreasing = T))))
ggplot(data = easy_hard) + geom_bar(mapping = aes(x = reorder(lineage,lineage,
                                                              function(x)-length(x)), fill = type)) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text()) + 
  xlab("Cell Line Lineage") + ylab("Cell Line x Drug Count")

# cur_data[, total_drug_attrib := sum(.SD), .SDcols = drug_cols, by = ]
quantile(cur_data[2, ..drug_cols])
max(cur_data[2, ..drug_cols])
min(cur_data[2, ..drug_cols])



# Add max/min for each attribute for each data type
cur_data[, max_drug := max(.SD), .SDcols = drug_cols, by = c("cell_name", "cpd_name")]
cur_data[, min_drug := min(.SD), .SDcols = drug_cols, by = c("cell_name", "cpd_name")]
cur_data[, max_prot := max(.SD), .SDcols = prot_cols, by = c("cell_name", "cpd_name")]
cur_data[, min_prot := min(.SD), .SDcols = prot_cols, by = c("cell_name", "cpd_name")]

cur_data[, "max_drug"]
cur_data[, "min_drug"]
cur_data[, "min_prot"]
cur_data[, "max_prot"]

# Plot histogram encompassing all positions of the drug data

plot(cur_data[2, ..drug_cols])



pca_res <- prcomp(cur_data[, ..drug_cols], scale. = TRUE)
umap_drug <- umap(cur_data[, ..drug_cols])
autoplot(pca_res, data = cur_data[, c("MSE_loss", "lineage", drug_cols), with = F], colour = "MSE_loss")
autoplot(pca_res, data = cur_data[, c("MSE_loss", "lineage", drug_cols), with = F], colour = "lineage")

cur_data[, ..prot_cols][,1]

pca_prot <- prcomp(cur_data[, ..prot_cols], scale. = TRUE)
autoplot(pca_prot, data = cur_data[, c("MSE_loss", "lineage", prot_cols), with = F], colour = "lineage")
autoplot(pca_prot, data = cur_data[, c("MSE_loss", "lineage", prot_cols), with = F], colour = "MSE_loss")
