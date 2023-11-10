# Samplewise_Omicwise_Performance_Comparison.R
require(data.table)
setDTthreads(8)
require(ggplot2)
require(gt)
require(flextable)
require(magrittr)
require(scales)
require(officer)

rsq <- function (x, y) cor(x, y, method = "pearson") ^ 2
rmse <- function(x, y) sqrt(mean((x - y)^2))
mae <- function(x, y) mean(abs(x - y))

# Setup ====
# Consider a specific lineage (e.g. lung), and compare the predictions in samples with targeted therapies when using
# different omic data types

targeted_drugs <- c("Idelalisib", "Olaparib", "Venetoclax", "Crizotinib", "Regorafenib", 
                    "Tretinoin", "Bortezomib", "Cabozantinib", "Dasatinib", "Erlotinib", 
                    "Sonidegib", "Vandetanib", "Axitinib", "Ibrutinib", "Gefitinib", 
                    "Nilotinib", "Tamoxifen", "Bosutinib", "Pazopanib", "Lapatinib", 
                    "Dabrafenib", "Bexarotene", "Temsirolimus", "Belinostat", 
                    "Sunitinib", "Vorinostat", "Trametinib", "Fulvestrant", "Sorafenib", 
                    "Vemurafenib", "Alpelisib")


drug_by_cancer <-
  list(
    Leukemia = c(
      "Idelalisib",
      "Venetoclax",
      "Tretinoin",
      "Dasatinib",
      "Ibrutinib",
      "Nilotinib",
      "Bosutinib"
    ),
    Lymphoma = c("Bortezomib", "Bexarotene", "Belinostat", "Vorinostat"),
    `Sarcoma` = c("Pazopanib"),
    `Breast Cancer` = c(
      "Olaparib",
      "Tamoxifen",
      "Lapatinib",
      "Fulvestrant",
      "Alpelisib"
    ),
    `Lung Cancer` = c("Crizotinib", "Erlotinib", "Gefitinib", "Dabrafenib"),
    `Colon/Colorectal Cancer` =  c("Regorafenib"),
    `Thyroid Cancer` = c("Cabozantinib", "Vandetanib"),
    `Skin Cancer` = c("Sonidegib", "Trametinib", "Vemurafenib"),
    `Kidney Cancer` = c("Axitinib", "Temsirolimus", "Sorafenib"),
    `Pancreatic Cancer` = c("Sunitinib")
  )


require(plyr)
cancer_by_drug <- as.data.table(plyr::ldply(drug_by_cancer, rbind))
cancer_by_drug <- melt(cancer_by_drug, id.vars = ".id")
cancer_by_drug <- na.omit(cancer_by_drug, "value")
cancer_by_drug <- cancer_by_drug[, c(1,3)]
colnames(cancer_by_drug) <- c("assigned_disease", "cpd_name")

drug_info <- fread("Data/DRP_Training_Data/CTRP_DRUG_INFO.csv")
cell_info <- fread("Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv")
ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")

# Read Data ====
# Select per fold validation files
all_cv_files <- list.files("Data/CV_Results/", recursive = T,
                           pattern = ".*final_validation.*", full.names = T)
# ".+drug_.{3,5}_HyperOpt.+"
bimodal_cv_files <- grep(pattern = ".ResponseOnly_.*drug_\\w{3,11}_HyperOpt.+", all_cv_files, value = T)
# cur_cv_files <- grep(pattern = ".ResponseOnly_.*drug_\\w{3,5}_.+", cur_cv_files, value = T)
# cur_cv_files <- grep(pattern = ".ResponseOnly_+drug_exp_HyperOpt.+", cur_cv_files, value = T)
# cur_cv_files_2 <- grep(pattern = ".Baseline_ElasticNet.+", all_cv_files, value = T)
# final_cv_files <- c(bimodal_cv_files, cur_cv_files_2)
final_cv_files <- bimodal_cv_files
# cur_cv_files <- grep(pattern = ".+drug_.{6,11}_HyperOpt.+", cur_cv_files, value = T)
length(final_cv_files)
# sum(grepl(".*ElasticNet.*", final_cv_files))
# Read all data
all_results <- vector(mode = "list", length = length(final_cv_files))
gc()
for (i in 1:length(final_cv_files)) {
  cur_res <- fread(final_cv_files[i])
  if (!grepl(".*Baseline_ElasticNet.*", final_cv_files[i])) {
    data_types <- gsub(".+ResponseOnly_\\w*drug_(.+)_HyperOpt.+", "\\1", final_cv_files[i])
    data_types <- toupper(data_types)
    merge_method <- gsub(".+MergeBy(\\w+)_.*RMSE.+", "\\1", final_cv_files[i])
    loss_method <- gsub(".+_(.*)RMSE.+", "\\1RMSE", final_cv_files[i])
    drug_type <- gsub(".+ResponseOnly_(\\w*)drug.+_HyperOpt.+", "\\1drug", final_cv_files[i])
    drug_type <- toupper(drug_type)
    split_method <- gsub(".+Split_(\\w+)_NoBottleNeck.+", "\\1", final_cv_files[i])
    # data_types <- strsplit(data_types, "_")[[1]]
    # cur_res$epoch <- as.integer(epoch)
    cur_res$data_types <- data_types
    cur_res$merge_method <- merge_method
    cur_res$loss_type <- loss_method
    cur_res$drug_type <- drug_type
    cur_res$split_method <- split_method
  } else {
    data_types <- gsub(".+ResponseOnly_drug_(\\w+)_Baseline_.+", "\\1", final_cv_files[i])
    data_types <- toupper(data_types)
    cur_res$data_types <- data_types
    cur_res$merge_method <- "Merge By Early Concat"
    cur_res$loss_type <- "UnBase Model + LDS"
    cur_res$drug_type <- "1024-bit ECFP"
    cur_res$split_method <- "Split By Both"
  }
  
  cur_fold <- gsub(".+CV_Index_(\\d)_.+", "\\1", final_cv_files[i])
  cur_res$fold <- cur_fold
  
  all_results[[i]] <- cur_res
}
all_results <- rbindlist(all_results, fill = T)
if (any(all_results$merge_method == "Merge By Early Concat")) {
  all_results[is.na(rmse_loss), RMSELoss := abs(target - predicted), by = .I]
  all_results[!is.na(rmse_loss), RMSELoss := rmse_loss, by = .I]
  all_results$rmse_loss <- NULL
} else {
  all_results[, RMSELoss := abs(target - predicted), by = .I]
}

# all_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold")]
all_results$V1 <- NULL

all_results[drug_type == "DRUG"]$drug_type <- "1024-bit ECFP"
all_results[drug_type == "GNNDRUG"]$drug_type <- "Base Model + GNN"

all_results[split_method == "BOTH"]$split_method <- "Split By Both"
all_results[split_method == "DRUG"]$split_method <- "Split By Drug Scaffold"
all_results[split_method == "CELL_LINE"]$split_method <- "Split By Cell Line"

all_results[merge_method == "Concat"]$merge_method <- "Merge By Concat"
# all_results[merge_method == "MergeByEarlyConcat"]$merge_method <- "Merge By Early Concat"
all_results[merge_method == "LMF"]$merge_method <- "Base Model + LMF"
all_results[merge_method == "Sum"]$merge_method <- "Merge By Sum"

all_results[loss_type == "RMSE"]$loss_type <- "UnBase Model + LDS"
all_results[loss_type == "WeightedRMSE"]$loss_type <- "Base Model + LDS"


all_results[, Targeted := ifelse(cpd_name %in% targeted_drugs, "Targeted Drug", "Untargeted Drug")]

all_results[, TargetRange := ifelse(target >= 0.7, "Target Above 0.7", "Target Below 0.7")]

fwrite(all_results, "Data/all_results.csv")


# Compare samples predicted using different omic data types ====
dir.create("Data/InferenceResults")
all_results <- fread("Data/all_results.csv")
trifecta_results <- unique(all_results[merge_method == "Base Model + LMF" &
              loss_type == "Base Model + LDS" &
              drug_type == "Base Model + GNN",
              c("cpd_name", "cell_name", "target", "predicted", "RMSELoss", "data_types", "split_method")])
trifecta_results <- unique(trifecta_results, by = c("cpd_name", "cell_name", "data_types"))
colnames(trifecta_results)[4] <- "pred"
colnames(trifecta_results)[5] <- "RMSE"

trifecta_results_wide <-
  dcast(data = trifecta_results, formula = cpd_name + cell_name + split_method + target ~ data_types,
        value.var = c("pred", "RMSE"), fill = NA)
# Save 
# fwrite(trifecta_results_wide, "Data/InferenceResults/Trifecta_Results_Wide_SplitByEach.csv")

trifecta_results_wide <- fread("Data/InferenceResults/Trifecta_Results_Wide_SplitByEach.csv")
trifecta_results_wide[target > 0.7]

# Are there data types that are beneficial for specific cell lines, lineages or drug types?
# Best performing drug per cell line and per lineage

# Add cell line lineage data
trifecta_results_wide <- merge(trifecta_results_wide, cell_info[, 2:5], by.x = "cell_name", by.y = "stripped_cell_line_name")
setcolorder(trifecta_results_wide, c("cell_name", "cpd_name", "primary_disease", "lineage", "lineage_subtype", "split_method", "target"))

# Get minimum RMSE across all omic data types and combos for each drug x cell line x split method
trifecta_results_wide[, lowest_rmse := min(.SD, na.rm = T),
                      by = c("cell_name", "cpd_name", "split_method"), .SDcols = patterns("RMSE")]

# Get highest AAC for each drug x cell line
trifecta_results_wide[, highest_aac := max(target),
                      by = c("cell_name", "cpd_name")]

# Top cell lines for each drug
# trifecta_results_wide[, c("top_cell_1", "top_cell_2", "top_cell_3") := tail(.SD, 3),
#                       .SDcols = c("cell_name", names(trifecta_results_wide) %like% "RMSE"), by = c("cpd_name", "split_method")]

trifecta_results_wide[target > 0.7 & cpd_name %in% targeted_drugs, tail(.SD, 3),
                      .SDcols = c("cell_name", "lowest_rmse"), by = c("cpd_name", "split_method")]


# Top cell lines for each drug
setorder(ctrp, -area_above_curve)
setkey(ctrp, "cpd_name")
ctrp <- merge(ctrp, cancer_by_drug, by = "cpd_name")
top_cells_per_drug_min_aac <- ctrp[area_above_curve >= 0.5 & cpd_name %in% targeted_drugs, head(.SD, 10), by = "cpd_name"][, c("cpd_name", "assigned_disease", "ccl_name", "primary_disease", "area_above_curve")]
top_cells_per_drug_matching_disease <- ctrp[(assigned_disease == primary_disease) & cpd_name %in% targeted_drugs, head(.SD, 10), by = "cpd_name"][, c("cpd_name", "assigned_disease", "ccl_name", "primary_disease", "area_above_curve")]
# max(ctrp[cpd_name == "Bosutinib"]$area_above_curve)
top_cells_per_drug <- unique(rbindlist(list(top_cells_per_drug_min_aac, top_cells_per_drug_matching_disease)))
# top_cells_per_drug <- merge(top_cells_per_drug, cancer_by_drug, by = "cpd_name")
setcolorder(top_cells_per_drug, c("cpd_name", "assigned_disease"))

set_flextable_defaults(
  font.size = 10, theme_fun = theme_vanilla,
  padding = 6,
  background.color = "#EFEFEF")

colourer <- col_numeric(
  palette = c("transparent", "red"),
  domain = c(0, 1))

top_cells_per_drug[, area_above_curve := round(area_above_curve, 3)]
colnames(top_cells_per_drug) <- c("Prescribed Drug(s)", "Cancer", "Cell Line", "Cell Line Primary Disease", "AAC")
setcolorder(top_cells_per_drug, c("Cancer", "Prescribed Drug(s)"))
setkey(top_cells_per_drug, Cancer, `Prescribed Drug(s)`)
ft <- flextable(top_cells_per_drug)
final_ft <- ft %>%
  merge_v(j = c("Cancer", "Prescribed Drug(s)", "Cell Line Primary Disease")) %>%
  border_inner(border = fp_border(color="gray", width = 1)) %>%
  border_outer(part="all", border = fp_border(color="gray", width = 2))

final_ft <- autofit(final_ft)
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/Dataset_Exploration/drug_by_cell_line_vs_target_disease_table.docx")


# Are there cancer types where the targeted drug is not prescribed for, but the model was able to predict
# a true high AAC?
# i.e. can the model give faithful drug repurposing recommendations?
# Can you show that previous repurposing recommendations were not possible since everyone was using EXP only?


targeted_results <- merge(top_cells_per_drug, trifecta_results_wide,
      by.x = c("Prescribed Drug(s)", "Cell Line"), by.y = c("cpd_name", "cell_name"))
targeted_results <- targeted_results[, !names(targeted_results) %like% "RMSE", with = F]
targeted_results$primary_disease <- NULL
# targeted_results$lowest_rmse <- NULL
targeted_results$highest_aac <- NULL
targeted_results$target <- NULL

pred_cols <- names(targeted_results)[names(targeted_results) %like% "pred"]
colnames(targeted_results)[9:ncol(targeted_results)]

melt_targeted_results <- melt(targeted_results, id.vars = colnames(targeted_results)[1:8])
# melt_targeted_results[, `Max AAC` := max(AAC), by = c("Prescribed Drug(s)", "Cell Line", "split_method")]
melt_targeted_results[, value := round(value, 3)]
# melt_targeted_results[, `Max AAC` := round(`Max AAC`, 3)]
wide_targeted_results <- dcast(melt_targeted_results, ... ~ split_method + variable)

# df_header <- as.data.table(expand.grid(unique(targeted_results$split_method), pred_cols), stringsAsFactors = FALSE)
# df_header[, header_id := paste(Var1, Var2, sep = "_")]
# setcolorder(df_header, "header_id")
# df_header <- rbindlist(list(no_header, df_header), use.names = T)
# setcolorder(df_header, c("header_id", "Var1", "Var2"))
# df_header <- as.data.frame(df_header, stringsAsFactors = F)


split_by_both_cols <- names(wide_targeted_results)[names(wide_targeted_results) %like% "Split By Both"]
split_by_both_cols <- split_by_both_cols[split_by_both_cols != "Split By Both_lowest_rmse"]
split_by_cell_cols <- names(wide_targeted_results)[names(wide_targeted_results) %like% "Split By Cell Line"]
split_by_cell_cols <- split_by_cell_cols[split_by_cell_cols != "Split By Cell Line_lowest_rmse"]
split_by_drug_cols <- names(wide_targeted_results)[names(wide_targeted_results) %like% "Split By Drug Scaffold"]
split_by_drug_cols <- split_by_drug_cols[split_by_drug_cols != "Split By Drug Scaffold_lowest_rmse"]

# wide_targeted_results$`Split By Both_highest_aac`
# wide_targeted_results$`Split By Drug Scaffold_highest_aac`
# wide_targeted_results$`Split By Cell Line_highest_aac`


setcolorder(wide_targeted_results, c("Prescribed Drug(s)", "Cancer", "Cell Line",
                                     "Cell Line Primary Disease", "lineage", "lineage_subtype", "AAC",
                                     "Split By Both_lowest_rmse", "Split By Drug Scaffold_lowest_rmse",
                                     "Split By Cell Line_lowest_rmse",
                                     split_by_both_cols, split_by_cell_cols, split_by_drug_cols))

colnames(wide_targeted_results)[5] <- "Lineage"
colnames(wide_targeted_results)[6] <- "Lineage Subtype"
wide_targeted_results$Lineage <- tools::toTitleCase(gsub("_", " ", wide_targeted_results$Lineage))
wide_targeted_results$`Lineage Subtype` <- tools::toTitleCase(gsub("_", " ", wide_targeted_results$`Lineage Subtype`))

# stringsAsFactors = False is required for flextables to work normally 
no_header <- data.frame(header_id = colnames(wide_targeted_results)[1:10],
                        Var1 = colnames(wide_targeted_results)[1:10],
                        Var2 = colnames(wide_targeted_results)[1:10], stringsAsFactors = FALSE)
temp <- expand.grid(unique(targeted_results$split_method), pred_cols)
df_header <- data.frame('header_id' = paste(temp$Var1, temp$Var2, sep = "_"),
                        'Var1' = temp$Var1,
                        'Var2' = temp$Var2,
                        stringsAsFactors = F)
# target_order <- c("Split By Both", "Split By Cell Line", "Split By Drug Scaffold")
# df_header[match(target_order, df_header$Var1), ]
df_header <- df_header[with(df_header, order(Var1)), ]
df_header <- rbind(no_header, df_header)

all(df_header$header_id == colnames(wide_targeted_results))
# final_ft <- ft %>%
#   merge_v(j = c("Cancer", "Prescribed Drug(s)", "Cell Line Primary Disease")) %>%
#   border_inner(border = fp_border(color="gray", width = 1)) %>%
#   border_outer(part="all", border = fp_border(color="gray", width = 2))
# 
# final_ft <- autofit(final_ft)

setcolorder(top_cells_per_drug, c("Cancer", "Prescribed Drug(s)"))
setkey(wide_targeted_results, Cancer, `Prescribed Drug(s)`)


# temp <- as.data.frame(wide_targeted_results, stringsAsFactors = F)
ft <- flextable(wide_targeted_results, col_keys = df_header$header_id)
final_ft <- set_header_df(ft, mapping = df_header, key = "header_id") %>%
  merge_v(part = "header") %>%
  merge_v(part = "body") %>%
  merge_h(part = "header", i = 1) %>%
  # theme_booktabs(bold_header = TRUE) %>% 
  align(align = "center", part = "all") %>%
  border_inner(border = fp_border(color="gray", width = 1)) %>%
  border_outer(part="all", border = fp_border(color="gray", width = 2))

  
final_ft <- autofit(final_ft)
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/Dataset_Exploration/drug_by_cell_line_vs_target_disease_by_all_omic_combos_table.docx")


# Summarize Repurposable Drugs ====
require(data.table)
setDTthreads(8)
require(ggplot2)
require(gt)
require(flextable)
require(magrittr)
require(scales)
require(officer)
drug_info <- fread("Data/DRP_Training_Data/CTRP_DRUG_INFO.csv")
cell_info <- fread("Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv")
ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")

# Must summarize to e.g. top 3 omic data types with lowest RMSEs, and write their splitting method,
# predicted AACs and RMSEs in the same cell, preferably (per each cell line and drug combo)
all_results <- fread("Data/all_results.csv")
trifecta_results <- unique(all_results[merge_method == "Base Model + LMF" &
                                         loss_type == "Base Model + LDS" &
                                         drug_type == "Base Model + GNN",
                                       c("cpd_name", "cell_name", "target", "predicted", "RMSELoss",
                                         "data_types", "split_method")])
rm(all_results)
gc()

trifecta_results <- trifecta_results[split_method == "Split By Cell Line"]
# trifecta_results <- unique(trifecta_results, by = c("cpd_name", "cell_name", "data_types"))
colnames(trifecta_results)[4] <- "pred"
colnames(trifecta_results)[5] <- "RMSE"

trifecta_results[, target := round(target, 3)]
trifecta_results[, pred := round(pred, 3)]
trifecta_results[, RMSE := round(RMSE, 3)]

trifecta_results <- merge(trifecta_results, cell_info[, 2:5], by.x = "cell_name", by.y = "stripped_cell_line_name")
trifecta_results <- merge(trifecta_results, cancer_by_drug, by = "cpd_name", all.x = T)

setcolorder(trifecta_results, c("cpd_name", "assigned_disease", "cell_name", "primary_disease", "lineage", "lineage_subtype", "split_method", "target"))

targeted_results <- trifecta_results[cpd_name %in% targeted_drugs]
# Order by RMSE
setorder(targeted_results, RMSE)
setkey(targeted_results, cpd_name, cell_name)

# Subset by lowest RMSEs
targeted_subset <- targeted_results[, head(.SD, 5), by = c("cpd_name", "cell_name")]
# targeted_subset <- targeted_subset[(target >= 0.5) | (assigned_disease == primary_disease)]
# It would be interesting to add inter-dataset concordance as a column to this data,
# although I'm not sure how relevant it would be


# Put top results in a single cell of a table 
targeted_subset[, cell_content := paste0(split_method, "\n", data_types, "\n", target, "\n", pred, "\n", RMSE)]
targeted_subset[, full_cell := paste(.SD, collapse = "\n"), by = c("cpd_name", "cell_name"), .SDcols = "cell_content"]

targeted_sub_sub <- unique(targeted_subset[, c("cpd_name", "assigned_disease", "cell_name", "primary_disease",
                                               "lineage", "lineage_subtype", "full_cell")])

targeted_subset$cell_content <- NULL
targeted_subset$full_cell <- NULL
# Are there cancer types where the targeted drug is not prescribed for, but the model was able to predict
# a true high AAC?
# i.e. can the model give faithful drug repurposing recommendations?
# Can you show that previous repurposing recommendations were not possible since everyone was using EXP only?

# Find each drug's highest AAC on it's assigned disease (NOTE the one liner)
targeted_subset[, highest_drug_match_disease_aac := max(.SD[assigned_disease == primary_disease]$target),
                by = "cpd_name"]

# targeted_subset[, avg_drug_match_disease_aac := mean(.SD[assigned_disease == primary_disease]$target),
#                 by = "cpd_name"]
# NOTE: Have the option of choosing AAC larger by a certain amount
better_than_assigned <- targeted_subset[target >= highest_drug_match_disease_aac]
uniqueN(better_than_assigned[, c("cpd_name", "cell_name")])  # 600 drug and cell line combinations
uniqueN(better_than_assigned[, c("cpd_name")])  # 31 repurposable drugs
better_than_assigned <- targeted_subset[target >= highest_drug_match_disease_aac + 0.1]
uniqueN(better_than_assigned[, c("cpd_name", "cell_name")])  # 182
uniqueN(better_than_assigned[, c("cpd_name")])  # 17 repurposable drugs
better_than_assigned <- targeted_subset[target >= highest_drug_match_disease_aac + 0.2]
uniqueN(better_than_assigned[, c("cpd_name", "cell_name")])  # 81
uniqueN(better_than_assigned[, c("cpd_name")])  # 14
unique(better_than_assigned[, c("cpd_name")])  # 14 repurposable drugs


# No minimum difference -> 2991 rows
# Minimum difference of 0.1 -> 910 rows
# Minimum difference of 0.2 -> 405 rows

better_than_assigned <- targeted_subset[target >= highest_drug_match_disease_aac]

# Repurposable drugs that have x amount higher AAC in unassigned cancers, and at least one
# of our models can predict with MAE loss less than 0.2 while seeing that cell line for the 
# first time
unique(better_than_assigned[target > highest_drug_match_disease_aac & RMSE <= 0.2]$cpd_name)

# Subset by THE lowest RMSE model to save space
setorder(better_than_assigned, RMSE)
setkey(better_than_assigned, cpd_name, cell_name)

# Save different subsets
# Only subset by MAE loss less than 0.2
better_than_assigned_subset <- better_than_assigned[target >= highest_drug_match_disease_aac]
# Select top 5 models
# final_data <- better_than_assigned_subset[, head(.SD, 5), by = c("cpd_name", "cell_name")]
final_data <- better_than_assigned_subset
setorder(final_data, -target)
setcolorder(final_data, c("assigned_disease", "cpd_name", "highest_drug_match_disease_aac",
                          "cell_name", "primary_disease", "lineage", "lineage_subtype"))
# Save final data
fwrite(final_data, "Data/repurposable_drugs_table.csv")

# final_data <- fread("Data/repurposable_drugs_table.csv")
# Subset by MAE loss less than 0.2 and AAC more than prescribed at least 0.2
# better_than_assigned_subset <- better_than_assigned[target >= highest_drug_match_disease_aac + 0.2 &
                                                      # RMSE <= 0.2]
better_than_assigned_subset <- better_than_assigned[target >= highest_drug_match_disease_aac &
                                                      RMSE <= 0.2]

final_data <- better_than_assigned_subset[, head(.SD, 1), by = c("cpd_name", "cell_name")]
final_data[cpd_name == "Ibrutinib"]
setorder(final_data, -target)
setcolorder(final_data, c("assigned_disease", "cpd_name", "highest_drug_match_disease_aac",
                          "cell_name", "primary_disease", "lineage", "lineage_subtype"))
# Save final data
fwrite(final_data, "Data/high_aac_lowest_mae_repurposable_drugs_table.csv")

# Put top results in a single cell of a table 
final_data[, data_types := gsub("_", " + ", data_types, fixed = T)]
final_data[, cell_content := paste0(data_types, "\n", pred)]

# Find percentage of entries that didn't have EXP in their data
final_data[!(data_types %like% "EXP")]  # 351 rows
nrow(final_data)  # 585 rows

# flextable(final_data)

setcolorder(final_data, c("primary_disease", "lineage_subtype", "cell_name", "target",
                          "highest_drug_match_disease_aac", "assigned_disease", "cpd_name",
                          "cell_content"))

final_data$data_types <- NULL
# All the data is from split by cell line 
final_data$split_method <- NULL
final_data$pred <- NULL
final_data$RMSE <- NULL
final_data$lineage <- NULL

setorder(final_data, primary_disease, lineage_subtype, cell_name)
colnames(final_data) <- c("Primary Disease", "Lineage Subtype", "Cell Line", "AAC",
                          "Highest AAC", "Cancer", "Drug",
                          "Top Model:\nData Type(s)\nPrediction")
final_data <- unique(final_data)
final_data$`Lineage Subtype` <- tools::toTitleCase(gsub("_", " ", final_data$`Lineage Subtype`))

colnames(final_data)[1:4] <- paste0("Empirical_", colnames(final_data)[1:4])
colnames(final_data)[5:7] <- paste0("Prescribed_", colnames(final_data)[5:7])
colnames(final_data)[8] <- paste0(colnames(final_data)[8], "_", colnames(final_data)[8])

header_id_colnames <- colnames(final_data)
header_split <- stringr::str_split(header_id_colnames, "_", simplify = T)
df_header <- data.frame('header_id' = header_id_colnames,
                         'Var1' = header_split[,1],  # Empirical vs Prescribed
                         'Var2' = header_split[,2],  # Empirical vs Prescribed
                         stringsAsFactors = F)
# no_header <- data.table(header_id = "Top Model:\nData Type(s)\nPrediction",
#                         Var1 = "Top Model:\nData Type(s)\nPrediction")
# df_header <- rbind(no_header, cur_header)

# NOTE: Cancers of the brain, the eye, the esophagus, the thyroid gland, and the skin of the
# head and neck are not usually classified as head and neck cancers.


ft <- flextable(final_data, col_keys = df_header$header_id)
final_ft <- set_header_df(ft, mapping = df_header, key = "header_id") %>%
  merge_v(part = "header") %>%
  merge_h(part = "header") %>%
  merge_v(part = "body") %>%
  # theme_booktabs(bold_header = TRUE) %>%
  align(align = "center", part = "all") %>%
  border_inner(border = fp_border(color="gray", width = 1)) %>%
  border_outer(part="all", border = fp_border(color="gray", width = 2)) %>%
  bold(bold = T, part = "header")

# ft <- flextable(final_data)
# final_ft <- ft %>%
#   # merge_v(j = c("Cancer", "Prescribed Drug(s)", "Cell Line Primary Disease")) %>%
#   merge_v() %>%
#   border_inner(border = fp_border(color="gray", width = 1)) %>%
#   border_outer(part="all", border = fp_border(color="gray", width = 2)) %>%
#   align(align = "center", part = "all")

final_ft <- autofit(final_ft)

read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Top_repurposable_Drugs_table.docx")


# Important data types per cell line / lineage / drug ====
# Are there data types that are beneficial for specific cell lines, lineages or drug types?
# Best performing drug per cell line and per lineage
# Best splitting method per cell line, per drug, per lineage, and overall, especially to predict higher AAC targets
require(data.table)
setDTthreads(8)
require(ggplot2)
require(gt)
require(flextable)
require(magrittr)
require(scales)
require(officer)
drug_info <- fread("Data/DRP_Training_Data/CTRP_DRUG_INFO.csv")
cell_info <- fread("Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv")

all_results <- fread("Data/all_results.csv")
trifecta_results <- unique(all_results[merge_method == "Base Model + LMF" &
                                         loss_type == "Base Model + LDS" &
                                         drug_type == "Base Model + GNN",
                                       c("cpd_name", "cell_name", "target", "predicted", "RMSELoss",
                                         "data_types", "split_method")])
rm(all_results)
gc()

trifecta_results <- unique(trifecta_results, by = c("cpd_name", "cell_name", "data_types"))
colnames(trifecta_results)[4] <- "pred"
colnames(trifecta_results)[5] <- "RMSE"

trifecta_results[, target := round(target, 3)]
trifecta_results[, pred := round(pred, 3)]
trifecta_results[, RMSE := round(RMSE, 3)]

trifecta_results <- merge(trifecta_results, cell_info[, 2:5], by.x = "cell_name", by.y = "stripped_cell_line_name")
# trifecta_results <- merge(trifecta_results, cancer_by_drug, by = "cpd_name", all.x = T)

# setcolorder(trifecta_results, c("cpd_name", "assigned_disease", "cell_name", "primary_disease", "lineage", "lineage_subtype", "split_method", "target"))
setcolorder(trifecta_results, c("cpd_name", "cell_name", "primary_disease", "lineage", "lineage_subtype", "split_method", "target"))


trifecta_results <- trifecta_results[split_method != "Split By Both"]

# What cell lines never have an AAC >= 0.5?
sensitive_cells <- (trifecta_results[target >= 0.5]$cell_name)
all_cells <- unique(trifecta_results$cell_name)
setdiff(all_cells, sensitive_cells)
# 12 cell lines never have AAC >= 0.5
# [1] "A498"     "CAPAN2"   "CCFSTTG1" "HPAFII"   "KMM1"     "KPL1"     "NCIH2172" "NCIH441"  "NMCG1"    "SNU410"  
# [11] "SW1783"   "TOLEDO"  

# Best data types for each cell line (top 3), will consider only data types that have some response, so AAC >= 0.5
trifecta_results[, top_mean_rmse_data_type_per_cell := mean(.SD[target > 0.5]$RMSE),
                 by = c("data_types", "cell_name", "split_method")]
trifecta_results[split_method == "Split By Cell Line" & cell_name== "HUCCT1"]
table(trifecta_results[split_method == "Split By Cell Line" & cell_name== "HUCCT1"]$data_types)

# Sort by mean RMSE per cell line and split_method
trifecta_results <- trifecta_results[!is.na(top_mean_rmse_data_type_per_cell)]
setorder(trifecta_results, split_method, primary_disease, cell_name, top_mean_rmse_data_type_per_cell)
unique(trifecta_results[, head(.SD, 3), by = "cell_name",
                 .SDcols = c("primary_disease", "data_types", "top_mean_rmse_data_type_per_cell")])

# Subset for (per) cell line related data
data_types_by_cell <- unique(trifecta_results[, c("cell_name", "primary_disease", "split_method", "data_types",
                             "top_mean_rmse_data_type_per_cell")])
setorder(data_types_by_cell, split_method, primary_disease, cell_name, top_mean_rmse_data_type_per_cell)
top_data_types_by_cell <- data_types_by_cell[, head(.SD, 3), by = "cell_name"]

unique(trifecta_results[, head(.SD, 3), by = "cell_name"])


# Do the same for lineages (top 3, response above 0.5 AAC)
trifecta_results[, top_mean_rmse_data_type_per_lineage := mean(.SD[target > 0.5]$RMSE),
                 by = c("data_types", "primary_disease", "split_method")]

# Sort by mean RMSE per cell line and split_method
trifecta_results <- trifecta_results[!is.na(top_mean_rmse_data_type_per_lineage)]
setorder(trifecta_results, split_method, primary_disease, cell_name, top_mean_rmse_data_type_per_lineage)
unique(trifecta_results[, head(.SD, 3), by = "cell_name",
                        .SDcols = c("primary_disease", "data_types", "top_mean_rmse_data_type_per_lineage")])

# Subset for (per) cell line related data
data_types_by_lineage <- unique(trifecta_results[, c("cell_name", "primary_disease", "split_method", "data_types",
                                                  "top_mean_rmse_data_type_per_lineage")])
setorder(data_types_by_lineage, primary_disease, split_method, cell_name, top_mean_rmse_data_type_per_lineage)
top_data_types_by_lineage <- data_types_by_lineage[, head(.SD, 3), by = c("primary_disease", "split_method")]

top_data_types_by_lineage$cell_name <- NULL
top_data_types_by_lineage <- unique(top_data_types_by_lineage)
top_data_types_by_lineage[, top_mean_rmse_data_type_per_lineage := round(top_mean_rmse_data_type_per_lineage, 3)]
# setorder(data_types_by_lineage, primary_disease, split_method, top_mean_rmse_data_type_per_lineage)
setorder(data_types_by_lineage, primary_disease, split_method, `data_types`)
colnames(top_data_types_by_lineage) <- c("Primary Disease", "Split Method", "Data Type(s)", "Mean RMSE per data type per lineage (in samples with AAC >= 0.5)")

ft <- flextable(top_data_types_by_lineage)

final_ft <- ft %>%
  # merge_v(j = c("Cancer", "Prescribed Drug(s)", "Cell Line Primary Disease")) %>%
  merge_v() %>%
  border_inner(border = fp_border(color="gray", width = 1)) %>%
  border_outer(part="all", border = fp_border(color="gray", width = 2)) %>%
  align(align = "center", part = "all")

final_ft <- autofit(final_ft)

read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/Dataset_Exploration/Best_data_types_for_each_lineage_table.docx")


# Final Comparison Table ====
## Bold Table Function ====
setup_bold_table <- function(cur_table, header_df, bold_df=NULL) {
  final_ft <- set_header_df(cur_table, mapping = header_df, key = "header_id") %>%
    merge_v(part = "header") %>%
    # merge_v(part = "body") %>%
    merge_h(part = "header", i = 1:3) %>%
    # theme_booktabs(bold_header = TRUE) %>%
    align(align = "center", part = "all") %>%
    border_inner(border = fp_border(color="gray", width = 1)) %>%
    border_outer(part="all", border = fp_border(color="gray", width = 2)) %>%
    bold(bold = T, part = "header")
  
  if (!is.null(bold_df)) {
    for (i in 1:nrow(bold_df)) {
      cur_data_type <- bold_df[i,]$`Omic Type(s)`
      cur_variable <- as.character(bold_df[i,]$variable)
      cur_pattern <- bold_df[`Omic Type(s)` == cur_data_type &
                               variable == cur_variable]$best_cv_mean
      # cur_pattern <- paste0(cur_pattern, " ±")
      i_formula <- as.formula(paste0("~ grepl(x = `", cur_variable, "`, pattern = '", cur_pattern, "', fixed = T)"))
      j_formula <- as.formula(paste0("~`", cur_variable, "`"))
      final_ft <- final_ft %>%
        bold(i = i_formula,
             j = j_formula) %>%
        highlight(i = i_formula,
                  j = j_formula,
                  color = "yellow",
                  part = "body")
    }
  }
  return(final_ft)
}


## Bimodal Cases ====
require(data.table)
setDTthreads(8)
require(ggplot2)
require(gt)
require(flextable)
require(magrittr)
require(scales)
require(officer)

set_flextable_defaults(
  font.size = 10, theme_fun = theme_vanilla,
  padding = 6,
  background.color = "#EFEFEF")

### Single Method ==== 
# all_results <- fread("Data/all_results.csv")

# Note the change from MAE to RMSE
# all_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "Targeted", "bottleneck")]
all_results[, loss_by_config := rmse(target, predicted), by = c("data_types", "merge_method", "loss_type",
                                                                "drug_type", "split_method", "TargetRange",
                                                                "Targeted", "bottleneck")]
all_results <- unique(all_results[, c("data_types", "merge_method", "loss_type",
                                      "drug_type", "split_method", "TargetRange",
                                      "Targeted", "bottleneck", "loss_by_config")])
# all_results_long_copy <- melt(unique(all_results[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config", "TargetRange", "Targeted", "bottleneck")]),
#                               id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "Targeted", "bottleneck"))

# all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange", "Targeted", "bottleneck")]
# all_results_long_copy[, cv_sd := sd(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange", "Targeted", "bottleneck")]

# all_results_long_copy$value <- NULL
# all_results_long_copy$variable <- NULL
# all_results_long_copy$fold <- NULL
# all_results_long_copy <- unique(all_results_long_copy)

rm(all_results)
gc()

bimodal_results <- all_results[nchar(data_types) < 6]
bimodal_results[merge_method == "Base Model"]$merge_method <- "Concat"
bimodal_results[merge_method == "Base Model + Sum"]$merge_method <- "Sum"
bimodal_results[merge_method == "Base Model + LMF"]$merge_method <- "LMF"
bimodal_results[loss_type == "Base Model"]$loss_type <- "non-LDS"
bimodal_results[loss_type == "Base Model + LDS"]$loss_type <- "LDS"
bimodal_results[drug_type == "Base Model"]$drug_type <- "ECFP"
bimodal_results[drug_type == "Base Model + GNN"]$drug_type <- "GNN"
bimodal_results[split_method == "Split By Both Cell Line & Drug Scaffold"]$split_method <- "Group Both"
bimodal_results[split_method == "Split By Drug Scaffold"]$split_method <- "Group Scaffold"
bimodal_results[split_method == "Split By Cell Line"]$split_method <- "Group Cell"
bimodal_results[TargetRange == "Target Above 0.7"]$TargetRange <- ">= 0.7"
bimodal_results[TargetRange == "Target Below 0.7"]$TargetRange <- "< 0.7"
bimodal_results[Targeted == "Targeted Drug"]$Targeted <- "Targeted"
bimodal_results[Targeted == "Untargeted Drug"]$Targeted <- "Untargeted"
bimodal_results[bottleneck == "No Data Bottleneck"]$bottleneck <- "No Bottleneck"
bimodal_results[bottleneck == "With Data Bottleneck"]$bottleneck <- "With Bottleneck"

# Consider only single technique bimodal results for now
bi_single_tech <- bimodal_results[
  (loss_type == "LDS" & merge_method == "Concat" & drug_type == "ECFP") |  # LDS
  (loss_type == "non-LDS" & merge_method == "LMF" & drug_type == "ECFP") |  # LMF
  (loss_type == "non-LDS" & merge_method == "Concat" & drug_type == "GNN") |  # GNN
  (loss_type == "non-LDS" & merge_method == "Concat" & drug_type == "ECFP")  # Baseline
]

# Subset for no bottleneck
bi_single_tech <- bi_single_tech[bottleneck == "No Bottleneck"]
bi_single_tech$bottleneck <- NULL

bi_single_tech[(loss_type == "LDS" & merge_method == "Concat" & drug_type == "ECFP"), Method := "LDS"]
bi_single_tech[(loss_type == "non-LDS" & merge_method == "LMF" & drug_type == "ECFP"), Method := "LMF"]
bi_single_tech[(loss_type == "non-LDS" & merge_method == "Concat" & drug_type == "GNN"), Method := "GNN"]
bi_single_tech[(loss_type == "non-LDS" & merge_method == "Concat" & drug_type == "ECFP"), Method := "Baseline"]
bi_single_tech$loss_type <- NULL
bi_single_tech$merge_method <- NULL
bi_single_tech$drug_type <- NULL

# bi_single_tech[, cv_mean := round(cv_mean, 3)]
# bi_single_tech[, cv_sd := round(cv_sd, 3)]
bi_single_tech[, loss_by_config := round(loss_by_config, 3)]
# bi_single_tech$Result <- paste(bi_single_tech$cv_mean, bi_single_tech$cv_sd, sep = " ± ")
bi_single_tech$Result <- as.character(bi_single_tech$loss_by_config)
# bi_single_tech$cv_mean <- NULL
# bi_single_tech$cv_sd <- NULL
bi_single_tech$loss_by_config <- NULL
bi_single_tech <- unique(bi_single_tech)
bi_single_tech <- dcast(bi_single_tech, ... ~ split_method + Method + Targeted + TargetRange,
                        value.var = "Result")

# Create Header for FlexTable
colnames(bi_single_tech)[1] <- "Data Type(s)"
header_id_colnames <- colnames(bi_single_tech)[-1]
header_split <- stringr::str_split(header_id_colnames, "_", simplify = T)
cur_header <- data.frame('header_id' = header_id_colnames,
                        'Var1' = header_split[,1],  # Grouping method
                        'Var2' = header_split[,2],  # Technique used
                        'Var3' = header_split[,3],  # Targeted or Untargeted
                        'Var4' = header_split[,4],  # AAC range
                        stringsAsFactors = F)
# target_order <- c("Split By Both", "Split By Cell Line", "Split By Drug Scaffold")
# df_header[match(target_order, df_header$Var1), ]
# df_header <- df_header[with(df_header, order(Var1)), ]
no_header <- data.table(header_id = "Data Type(s)",
                        Var1 = "Data Type(s)", Var2 = "Data Type(s)",
                        Var3 = "Data Type(s)", Var4 = "Data Type(s)")
df_header <- rbind(no_header, cur_header)

# all(df_header$header_id == colnames(wide_targeted_results))
# final_ft <- ft %>%
#   merge_v(j = c("Cancer", "Prescribed Drug(s)", "Cell Line Primary Disease")) %>%
#   border_inner(border = fp_border(color="gray", width = 1)) %>%
#   border_outer(part="all", border = fp_border(color="gray", width = 2))
# 
# final_ft <- autofit(final_ft)

# setcolorder(top_cells_per_drug, c("Cancer", "Prescribed Drug(s)"))
# setkey(wide_targeted_results, Cancer, `Prescribed Drug(s)`)

flextable(bi_single_tech)
# temp <- as.data.frame(wide_targeted_results, stringsAsFactors = F)
ft <- flextable(bi_single_tech, col_keys = df_header$header_id)
final_ft <- set_header_df(ft, mapping = df_header, key = "header_id") %>%
  merge_v(part = "header") %>%
  merge_v(part = "body") %>%
  merge_h(part = "header", i = 1:3) %>%
  # theme_booktabs(bold_header = TRUE) %>% 
  align(align = "center", part = "all") %>%
  border_inner(border = fp_border(color="gray", width = 1)) %>%
  border_outer(part="all", border = fp_border(color="gray", width = 2)) %>%
  bold(bold = T, part = "header")


final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Bimodal_byGroup_byMethod_byTargeted_byRange_table.docx")


#### Group By Both ====
df_header <- df_header[Var4 %like% ">= 0.7|Data Type"]
group_both_subset <- bi_single_tech[, .SD, .SDcols = patterns("Data Type|Group Both")]
# Subset by AAC >= 0.7
group_both_subset <- group_both_subset[, .SD, .SDcols = patterns("Data Type|>= 0.7")]
header_both_subset <- df_header[Var1 == "Group Both" | Var1 == "Data Type(s)"]

group_both_cv_means <- melt(group_both_subset,
                            measure.vars = colnames(group_both_subset)[-1])
# Subset for those with higher AACs
group_both_cv_means <- group_both_cv_means[variable %like% ">= 0.7"]

group_both_cv_means$cv_mean <- as.numeric(gsub(" ± .+", "", group_both_cv_means$value))
group_both_cv_means[, c("Group", "Method", "Targeted", "Range") := tstrsplit(variable, "_")]

# Everything except the method used, so we can compare method performance
group_both_cv_means[, best_cv_mean := as.character(min(cv_mean)),
                    by = c("Data Type(s)", "Group", "Targeted")]

group_both_cv_means <- group_both_cv_means[cv_mean == as.numeric(best_cv_mean)]
bold_df <- unique(group_both_cv_means[, c("Data Type(s)", "variable", "best_cv_mean")])


ft <- flextable(group_both_subset, col_keys = header_both_subset$header_id)

final_ft <- setup_bold_table(ft, header_both_subset, bold_df)

final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Bimodal_GroupByBoth_table.docx")


#### Group By Cell Line ====
cur_group_subset <- bi_single_tech[, .SD, .SDcols = patterns("Data Type|Group Cell")]
cur_group_subset <- cur_group_subset[, .SD, .SDcols = patterns("Data Type|>= 0.7")]

cur_header_subset <- df_header[Var1 == "Group Cell" | Var1 == "Data Type(s)"]

cur_group_cv_means <- melt(cur_group_subset,
                            measure.vars = colnames(cur_group_subset)[-1])
# Subset for those with higher AACs
cur_group_cv_means <- cur_group_cv_means[variable %like% ">= 0.7"]
cur_group_cv_means$cv_mean <- as.numeric(gsub(" ± .+", "", cur_group_cv_means$value))
cur_group_cv_means[, c("Group", "Method", "Targeted", "Range") := tstrsplit(variable, "_")]

# Everything except the method used, so we can compare method performance
cur_group_cv_means[, best_cv_mean := as.character(min(cv_mean)),
                    by = c("Data Type(s)", "Group", "Targeted")]

cur_group_cv_means <- cur_group_cv_means[cv_mean == as.numeric(best_cv_mean)]
bold_df <- unique(cur_group_cv_means[, c("Data Type(s)", "variable", "best_cv_mean")])


ft <- flextable(cur_group_subset, col_keys = cur_header_subset$header_id)


final_ft <- setup_bold_table(ft, cur_header_subset, bold_df)
final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Bimodal_GroupByCellLine_table.docx")


#### Group By Drug Scaffold ====
cur_group_subset <- bi_single_tech[, .SD, .SDcols = patterns("Data Type|Group Scaffold")]
cur_group_subset <- cur_group_subset[, .SD, .SDcols = patterns("Data Type|>= 0.7")]
cur_header_subset <- df_header[Var1 == "Group Scaffold" | Var1 == "Data Type(s)"]


cur_group_cv_means <- melt(cur_group_subset,
                           measure.vars = colnames(cur_group_subset)[-1])
# Subset for those with higher AACs
cur_group_cv_means <- cur_group_cv_means[variable %like% ">= 0.7"]
cur_group_cv_means$cv_mean <- as.numeric(gsub(" ± .+", "", cur_group_cv_means$value))
cur_group_cv_means[, c("Group", "Method", "Targeted", "Range") := tstrsplit(variable, "_")]

# Everything except the method used, so we can compare method performance
cur_group_cv_means[, best_cv_mean := as.character(min(cv_mean)),
                   by = c("Data Type(s)", "Group", "Targeted")]

cur_group_cv_means <- cur_group_cv_means[cv_mean == as.numeric(best_cv_mean)]
bold_df <- unique(cur_group_cv_means[, c("Data Type(s)", "variable", "best_cv_mean")])


ft <- flextable(cur_group_subset, col_keys = cur_header_subset$header_id)


final_ft <- setup_bold_table(ft, cur_header_subset, bold_df)
final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Bimodal_GroupByScaffold_table.docx")


#### Best Model for Each Omic Data ====
bimodal_results[merge_method == "Merge By Early Concat"]$merge_method <- "Elastic Net"
bimodal_results_copy <- bimodal_results

# Assign model names
bimodal_results_copy[(loss_type == "LDS" & merge_method == "Concat" & drug_type == "ECFP"), Method := "LDS"]
bimodal_results_copy[(loss_type == "non-LDS" & merge_method == "LMF" & drug_type == "ECFP"), Method := "LMF"]
bimodal_results_copy[(loss_type == "non-LDS" & merge_method == "Concat" & drug_type == "GNN"), Method := "GNN"]
bimodal_results_copy[(loss_type == "non-LDS" & merge_method == "Concat" & drug_type == "ECFP"), Method := "Baseline"]
bimodal_results_copy[(loss_type == "LDS" & merge_method == "LMF" & drug_type == "ECFP"), Method := "LDS+LMF"]
bimodal_results_copy[(loss_type == "non-LDS" & merge_method == "LMF" & drug_type == "GNN"), Method := "LMF+GNN"]
bimodal_results_copy[(loss_type == "LDS" & merge_method == "Concat" & drug_type == "GNN"), Method := "LDS+GNN"]
bimodal_results_copy[(loss_type == "LDS" & merge_method == "LMF" & drug_type == "GNN"), Method := "LDS+LMF+GNN"]

# bimodal_results_copy[(loss_type == "non-LDS" & merge_method == "Sum" & drug_type == "ECFP"), Method := "LDS+Sum+GNN"]
# bimodal_results_copy[(loss_type == "LDS" & merge_method == "Sum" & drug_type == "GNN"), Method := "LDS+Sum+GNN"]

bimodal_results_copy[(merge_method == "Elastic Net"), Method := "Elastic Net"]

bimodal_results_copy$loss_type <- NULL
bimodal_results_copy$merge_method <- NULL
bimodal_results_copy$drug_type <- NULL

# Subset for upper ranges, remove bottleneck column
bimodal_results_copy <- bimodal_results_copy[TargetRange == ">= 0.7"]
bimodal_results_copy$bottleneck <- NULL

# Find the lowest CV mean by data types and drug type
bimodal_results_copy[, best_cv_mean := min(cv_mean), by = c("data_types", "split_method", "Targeted")]

# Subset models for those with best CV means
bimodal_results_copy <- bimodal_results_copy[cv_mean == best_cv_mean]
colnames(bimodal_results_copy)[1] <- "Data Type(s)"
bimodal_results_copy[, variable := paste(split_method, Targeted, TargetRange, sep = "_")]

# Find the best model overall for grouping method and drug type
bimodal_results_copy[, best_overall := min(cv_mean), by = c("split_method", "Targeted")]
bolf_df <- bimodal_results_copy[cv_mean == best_overall]
bold_df <- unique(bolf_df[, c("Data Type(s)", "variable", "best_overall")])
bold_df$best_overall <- round(bold_df$best_overall, 3)
colnames(bold_df)[3] <- "best_cv_mean"
bimodal_results_copy$variable <- NULL
bimodal_results_copy$best_overall <- NULL

bimodal_results_copy[, cv_mean := round(cv_mean, 3)]
bimodal_results_copy[, cv_sd := round(cv_sd, 3)]
bimodal_results_copy$Result <- paste(bimodal_results_copy$cv_mean, bimodal_results_copy$cv_sd, sep = " ± ")
bimodal_results_copy$cv_mean <- NULL
bimodal_results_copy$cv_sd <- NULL
bimodal_results_copy$best_cv_mean <- NULL
bimodal_results_copy <- unique(bimodal_results_copy)
bimodal_results_copy$Result <- paste(bimodal_results_copy$Method, bimodal_results_copy$Result, sep = "\n")
bimodal_results_copy$Method <- NULL

bimodal_results_copy <- dcast(bimodal_results_copy, ... ~ split_method + Targeted + TargetRange,
                              value.var = "Result")

header_id_colnames <- colnames(bimodal_results_copy)[-1]
header_split <- stringr::str_split(header_id_colnames, "_", simplify = T)
cur_header <- data.frame('header_id' = header_id_colnames,
                         'Var1' = header_split[,1],  # Grouping method
                         'Var2' = header_split[,2],  # Targeted or Untargeted
                         'Var3' = header_split[,3],  # AAC range
                         stringsAsFactors = F)
no_header <- data.table(header_id = "Data Type(s)",
                        Var1 = "Data Type(s)", Var2 = "Data Type(s)",
                        Var3 = "Data Type(s)")
df_header <- rbind(no_header, cur_header)

ft <- flextable(bimodal_results_copy, col_keys = df_header$header_id)

final_ft <- setup_bold_table(ft, df_header, bold_df)

final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Bimodal_Best_By_DataType_table.docx")


### Two Methods ====
require(data.table)
setDTthreads(8)
require(ggplot2)
require(gt)
require(flextable)
require(magrittr)
require(scales)
require(officer)
rsq <- function (x, y) cor(x, y, method = "pearson") ^ 2
rmse <- function(x, y) sqrt(mean((x - y)^2))
mae <- function(x, y) mean(abs(x - y))


set_flextable_defaults(
  font.size = 10, theme_fun = theme_vanilla,
  padding = 6,
  background.color = "#EFEFEF")

all_results <- fread("Data/all_results.csv")
all_results <- all_results[nchar(data_types) <= 5]


all_results[, loss_by_config := rmse(target, predicted), by = c("data_types", "merge_method", "loss_type",
                                                                "drug_type", "split_method", "TargetRange",
                                                                "Targeted", "bottleneck")]
all_results <- unique(all_results[, c("data_types", "merge_method", "loss_type",
                                      "drug_type", "split_method", "TargetRange",
                                      "Targeted", "bottleneck", "loss_by_config")])

# all_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "Targeted", "bottleneck")]
# all_results_long_copy <- melt(unique(all_results[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config", "TargetRange", "Targeted", "bottleneck")]),
#                               id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "Targeted", "bottleneck"))
# all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange", "Targeted", "bottleneck")]
# all_results_long_copy[, cv_sd := sd(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange", "Targeted", "bottleneck")]

# all_results_long_copy$value <- NULL
# all_results_long_copy$variable <- NULL
# all_results_long_copy$fold <- NULL
# all_results_long_copy <- unique(all_results_long_copy)

# rm(all_results)
gc()

bimodal_results <- all_results[nchar(data_types) < 6]
bimodal_results[merge_method == "Base Model"]$merge_method <- "Concat"
bimodal_results[merge_method == "Base Model + Sum"]$merge_method <- "Sum"
bimodal_results[merge_method == "Base Model + LMF"]$merge_method <- "LMF"
bimodal_results[loss_type == "Base Model"]$loss_type <- "non-LDS"
bimodal_results[loss_type == "Base Model + LDS"]$loss_type <- "LDS"
bimodal_results[drug_type == "Base Model"]$drug_type <- "ECFP"
bimodal_results[drug_type == "Base Model + GNN"]$drug_type <- "GNN"
bimodal_results[split_method == "Split By Both Cell Line & Drug Scaffold"]$split_method <- "Group Both"
bimodal_results[split_method == "Split By Drug Scaffold"]$split_method <- "Group Scaffold"
bimodal_results[split_method == "Split By Cell Line"]$split_method <- "Group Cell"
bimodal_results[TargetRange == "Target Above 0.7"]$TargetRange <- ">= 0.7"
bimodal_results[TargetRange == "Target Below 0.7"]$TargetRange <- "< 0.7"
bimodal_results[Targeted == "Targeted Drug"]$Targeted <- "Targeted"
bimodal_results[Targeted == "Untargeted Drug"]$Targeted <- "Untargeted"
bimodal_results[bottleneck == "No Data Bottleneck"]$bottleneck <- "No Bottleneck"
bimodal_results[bottleneck == "With Data Bottleneck"]$bottleneck <- "With Bottleneck"

#### Grouping Preparation ====
# Consider only single technique bimodal results for now
bi_two_tech <- bimodal_results[
  (loss_type == "LDS" & merge_method == "LMF" & drug_type == "ECFP") |  # LDS + LMF
    (loss_type == "non-LDS" & merge_method == "LMF" & drug_type == "GNN") |  # LMF + GNN
    (loss_type == "LDS" & merge_method == "Concat" & drug_type == "GNN") |  # LDS + GNN
    (loss_type == "non-LDS" & merge_method == "Concat" & drug_type == "ECFP")  # Baseline
]
# Subset for no bottleneck
bi_two_tech <- bi_two_tech[bottleneck == "No Bottleneck"]
bi_two_tech$bottleneck <- NULL

# Assign Model Names
bi_two_tech[(loss_type == "LDS" & merge_method == "LMF" & drug_type == "ECFP"), Method := "LDS+LMF"]
bi_two_tech[(loss_type == "non-LDS" & merge_method == "LMF" & drug_type == "GNN"), Method := "LMF+GNN"]
bi_two_tech[(loss_type == "LDS" & merge_method == "Concat" & drug_type == "GNN"), Method := "LDS+GNN"]
bi_two_tech[(loss_type == "non-LDS" & merge_method == "Concat" & drug_type == "ECFP"), Method := "Baseline"]
bi_two_tech$loss_type <- NULL
bi_two_tech$merge_method <- NULL
bi_two_tech$drug_type <- NULL

bi_two_tech[, cv_mean := round(cv_mean, 3)]
bi_two_tech[, cv_sd := round(cv_sd, 3)]
bi_two_tech$Result <- paste(bi_two_tech$cv_mean, bi_two_tech$cv_sd, sep = " ± ")
bi_two_tech$cv_mean <- NULL
bi_two_tech$cv_sd <- NULL
bi_two_tech <- unique(bi_two_tech)
bi_two_tech <- dcast(bi_two_tech, ... ~ split_method + Method + Targeted + TargetRange,
                        value.var = "Result")

# Make Header Table for FlexTable
colnames(bi_two_tech)[1] <- "Data Type(s)"
header_id_colnames <- colnames(bi_two_tech)[-1]
header_split <- stringr::str_split(header_id_colnames, "_", simplify = T)
cur_header <- data.frame('header_id' = header_id_colnames,
                         'Var1' = header_split[,1],  # Grouping method
                         'Var2' = header_split[,2],  # Technique used
                         'Var3' = header_split[,3],  # Targeted or Untargeted
                         'Var4' = header_split[,4],  # AAC range
                         stringsAsFactors = F)
no_header <- data.table(header_id = "Data Type(s)",
                        Var1 = "Data Type(s)", Var2 = "Data Type(s)",
                        Var3 = "Data Type(s)", Var4 = "Data Type(s)")
df_header <- rbind(no_header, cur_header)


# setup_bold_table <- function(cur_table, header_df, bold_df) {
#   final_ft <- set_header_df(cur_table, mapping = header_df, key = "header_id") %>%
#     merge_v(part = "header") %>%
#     merge_v(part = "body") %>%
#     merge_h(part = "header", i = 1:3) %>%
#     align(align = "center", part = "all") %>%
#     border_inner(border = fp_border(color="gray", width = 1)) %>%
#     border_outer(part="all", border = fp_border(color="gray", width = 2)) %>%
#     bold(bold = T, part = "header")
#   
#   for (i in 1:nrow(bold_df)) {
#     cur_data_type <- bold_df[i,]$`Data Type(s)`
#     cur_variable <- as.character(bold_df[i,]$variable)
#     cur_pattern <- bold_df[`Data Type(s)` == cur_data_type &
#                              variable == cur_variable]$best_cv_mean
#     i_formula <- as.formula(paste0("~ grepl(x = `", cur_variable, "`, pattern = ", cur_pattern, ", fixed = T)"))
#     j_formula <- as.formula(paste0("~`", cur_variable, "`"))
#     final_ft <- final_ft %>%
#       bold(i = i_formula,
#            j = j_formula)
#   }
#   return(final_ft)
# }

#### Group By Both ====
df_header <- df_header[Var4 %like% ">= 0.7|Data Type"]
group_both_subset <- bi_two_tech[, .SD, .SDcols = patterns("Data Type|Group Both")]
# Subset by AAC >= 0.7
group_both_subset <- group_both_subset[, .SD, .SDcols = patterns("Data Type|>= 0.7")]
header_both_subset <- df_header[Var1 == "Group Both" | Var1 == "Data Type(s)"]

group_both_cv_means <- melt(group_both_subset,
                            measure.vars = colnames(group_both_subset)[-1])
# Subset for those with higher AACs
group_both_cv_means <- group_both_cv_means[variable %like% ">= 0.7"]

group_both_cv_means$cv_mean <- as.numeric(gsub(" ± .+", "", group_both_cv_means$value))
group_both_cv_means[, c("Group", "Method", "Targeted", "Range") := tstrsplit(variable, "_")]

# Everything except the method used, so we can compare method performance
group_both_cv_means[, best_cv_mean := as.character(min(cv_mean)),
                    by = c("Data Type(s)", "Group", "Targeted")]

group_both_cv_means <- group_both_cv_means[cv_mean == as.numeric(best_cv_mean)]
bold_df <- unique(group_both_cv_means[, c("Data Type(s)", "variable", "best_cv_mean")])


ft <- flextable(group_both_subset, col_keys = header_both_subset$header_id)

final_ft <- setup_bold_table(ft, header_both_subset, bold_df)

final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Bimodal_TwoMethods_GroupByBoth_table.docx")


#### Group By Cell Line ====
cur_group_subset <- bi_two_tech[, .SD, .SDcols = patterns("Data Type|Group Cell")]
cur_group_subset <- cur_group_subset[, .SD, .SDcols = patterns("Data Type|>= 0.7")]

cur_header_subset <- df_header[Var1 == "Group Cell" | Var1 == "Data Type(s)"]

cur_group_cv_means <- melt(cur_group_subset,
                           measure.vars = colnames(cur_group_subset)[-1])
# Subset for those with higher AACs
cur_group_cv_means <- cur_group_cv_means[variable %like% ">= 0.7"]
cur_group_cv_means$cv_mean <- as.numeric(gsub(" ± .+", "", cur_group_cv_means$value))
cur_group_cv_means[, c("Group", "Method", "Targeted", "Range") := tstrsplit(variable, "_")]

# Everything except the method used, so we can compare method performance
cur_group_cv_means[, best_cv_mean := as.character(min(cv_mean)),
                   by = c("Data Type(s)", "Group", "Targeted")]

cur_group_cv_means <- cur_group_cv_means[cv_mean == as.numeric(best_cv_mean)]
bold_df <- unique(cur_group_cv_means[, c("Data Type(s)", "variable", "best_cv_mean")])


ft <- flextable(cur_group_subset, col_keys = cur_header_subset$header_id)


final_ft <- setup_bold_table(ft, cur_header_subset, bold_df)
final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Bimodal_TwoMethods_GroupByCellLine_table.docx")


#### Group By Drug Scaffold ====
cur_group_subset <- bi_two_tech[, .SD, .SDcols = patterns("Data Type|Group Scaffold")]
cur_group_subset <- cur_group_subset[, .SD, .SDcols = patterns("Data Type|>= 0.7")]
cur_header_subset <- df_header[Var1 == "Group Scaffold" | Var1 == "Data Type(s)"]


cur_group_cv_means <- melt(cur_group_subset,
                           measure.vars = colnames(cur_group_subset)[-1])
# Subset for those with higher AACs
cur_group_cv_means <- cur_group_cv_means[variable %like% ">= 0.7"]
cur_group_cv_means$cv_mean <- as.numeric(gsub(" ± .+", "", cur_group_cv_means$value))
cur_group_cv_means[, c("Group", "Method", "Targeted", "Range") := tstrsplit(variable, "_")]

# Everything except the method used, so we can compare method performance
cur_group_cv_means[, best_cv_mean := as.character(min(cv_mean)),
                   by = c("Data Type(s)", "Group", "Targeted")]

cur_group_cv_means <- cur_group_cv_means[cv_mean == as.numeric(best_cv_mean)]
bold_df <- unique(cur_group_cv_means[, c("Data Type(s)", "variable", "best_cv_mean")])


ft <- flextable(cur_group_subset, col_keys = cur_header_subset$header_id)


final_ft <- setup_bold_table(ft, cur_header_subset, bold_df)
final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Bimodal_TwoMethods_GroupByScaffold_table.docx")

### Best Model for Each Omic Data ====
# bimodal_results[merge_method == "Merge By Early Concat"]$merge_method <- "Elastic Net"
all_results <- fread("Data/all_results.csv")
shared_combos <- fread("Data/shared_unique_combinations.csv")
shared_combos[, unique_samples := paste0(cpd_name, "_", cell_name)]

# Subset for bimodal results
bimodal_results <- all_results[nchar(data_types) <= 5]
# Subset by all shared samples all each data type
bimodal_results[, unique_samples := paste0(cpd_name, "_", cell_name)]
bimodal_results <- bimodal_results[unique_samples %in% shared_combos$unique_samples]

# Calculate total RMSE by configuration
bimodal_results[, loss_by_config := rmse(target, predicted), by = c("data_types", "merge_method", "loss_type",
                                                                "drug_type", "split_method", "TargetRange",
                                                                "Targeted", "bottleneck")]
bimodal_results <- unique(bimodal_results[, c("data_types", "merge_method", "loss_type",
                                      "drug_type", "split_method", "TargetRange",
                                      "Targeted", "bottleneck", "loss_by_config")])

bimodal_results_copy <- bimodal_results

# Assign model names
# Baseline and Elastic Net
bimodal_results_copy[(loss_type == "Base Model" & merge_method == "Merge By Early Concat" & drug_type == "Base Model"), Method := "ElasticNet"]
bimodal_results_copy[(loss_type == "Base Model" & merge_method == "Base Model" & drug_type == "Base Model"), Method := "Baseline"]
# Single Technique
bimodal_results_copy[(loss_type == "Base Model + LDS" & merge_method == "Base Model" & drug_type == "Base Model"), Method := "LDS"]
bimodal_results_copy[(loss_type == "Base Model" & merge_method == "Base Model + LMF" & drug_type == "Base Model"), Method := "LMF"]
bimodal_results_copy[(loss_type == "Base Model" & merge_method == "Base Model + Sum" & drug_type == "Base Model"), Method := "Sum"]
bimodal_results_copy[(loss_type == "Base Model" & merge_method == "Base Model" & drug_type == "Base Model + GNN"), Method := "GNN"]
# Two Techniques
bimodal_results_copy[(loss_type == "Base Model + LDS" & merge_method == "Base Model + LMF" & drug_type == "Base Model"), Method := "LDS+LMF"]
bimodal_results_copy[(loss_type == "Base Model" & merge_method == "Base Model + LMF" & drug_type == "Base Model + GNN"), Method := "LMF+GNN"]
bimodal_results_copy[(loss_type == "Base Model + LDS" & merge_method == "Base Model" & drug_type == "Base Model + GNN"), Method := "LDS+GNN"]
# Three Techniques
bimodal_results_copy[(loss_type == "Base Model + LDS" & merge_method == "Base Model + Sum" & drug_type == "Base Model + GNN"), Method := "LDS+Sum+GNN"]
bimodal_results_copy[(loss_type == "Base Model + LDS" & merge_method == "Base Model + LMF" & drug_type == "Base Model + GNN"), Method := "LDS+LMF+GNN"]

bimodal_results_copy$loss_type <- NULL
bimodal_results_copy$merge_method <- NULL
bimodal_results_copy$drug_type <- NULL

# Subset for upper ranges, remove bottleneck column
bimodal_results_copy <- bimodal_results_copy[TargetRange == "Target Above 0.7"]
bimodal_results_copy <- bimodal_results_copy[bottleneck == "No Data Bottleneck"]
table(bimodal_results_copy$model_name)

bimodal_results_copy$bottleneck <- NULL

# Find the lowest RMSE loss by data types and drug type
# bimodal_results_copy[, best_cv_mean := min(cv_mean), by = c("data_types", "split_method", "Targeted")]
bimodal_results_copy[, best_cv_mean := min(loss_by_config), by = c("data_types", "split_method", "Targeted")]

# Subset models for those with best CV means
# bimodal_results_copy <- bimodal_results_copy[cv_mean == best_cv_mean]
bimodal_results_copy <- bimodal_results_copy[loss_by_config == best_cv_mean]
colnames(bimodal_results_copy)[1] <- "Omic Type(s)"
bimodal_results_copy[, variable := paste(split_method, Targeted, TargetRange, sep = "_")]

# Find the best model overall for grouping method and drug type
# bimodal_results_copy[, best_overall := min(cv_mean), by = c("split_method", "Targeted")]
bimodal_results_copy[, best_overall := min(loss_by_config), by = c("split_method", "Targeted")]
# bolf_df <- bimodal_results_copy[cv_mean == best_overall]
bolf_df <- bimodal_results_copy[loss_by_config == best_overall]
bold_df <- unique(bolf_df[, c("Omic Type(s)", "variable", "best_overall")])
bold_df$best_overall <- round(bold_df$best_overall, 3)
colnames(bold_df)[3] <- "best_cv_mean"
bimodal_results_copy$variable <- NULL
bimodal_results_copy$best_overall <- NULL

# bimodal_results_copy[, cv_mean := round(cv_mean, 3)]
# bimodal_results_copy[, cv_sd := round(cv_sd, 3)]
bimodal_results_copy[, loss_by_config := round(loss_by_config, 3)]
# bimodal_results_copy$Result <- paste(bimodal_results_copy$cv_mean, bimodal_results_copy$cv_sd, sep = " ± ")
bimodal_results_copy$Result <- as.character(bimodal_results_copy$loss_by_config)
# bimodal_results_copy$cv_mean <- NULL
# bimodal_results_copy$cv_sd <- NULL
bimodal_results_copy$loss_by_config <- NULL
bimodal_results_copy$best_cv_mean <- NULL
bimodal_results_copy <- unique(bimodal_results_copy)
bimodal_results_copy$Result <- paste(bimodal_results_copy$Method, bimodal_results_copy$Result, sep = "\n")
bimodal_results_copy$Method <- NULL

bimodal_results_copy <- dcast(bimodal_results_copy, ... ~ split_method + Targeted + TargetRange,
                              value.var = "Result")

header_id_colnames <- colnames(bimodal_results_copy)[-1]
header_split <- stringr::str_split(header_id_colnames, "_", simplify = T)
cur_header <- data.frame('header_id' = header_id_colnames,
                         'Var1' = header_split[,1],  # Grouping method
                         'Var2' = header_split[,2],  # Targeted or Untargeted
                         'Var3' = header_split[,3],  # AAC range
                         stringsAsFactors = F)
no_header <- data.table(header_id = "Omic Type(s)",
                        Var1 = "Omic Type(s)", Var2 = "Omic Type(s)",
                        Var3 = "Omic Type(s)")
df_header <- rbind(no_header, cur_header)

bimodal_results_copy[, "Omic Type(s)" := factor(bimodal_results_copy$`Omic Type(s)`,
                                             levels = c("MUT", "CNV", "EXP", "PROT",
                                                        "MIRNA", "METAB", "HIST", "RPPA"))]
bar_level_df <- data.frame(temp = c("MUT", "CNV", "EXP", "PROT",
                                  "MIRNA", "METAB", "HIST", "RPPA"))
colnames(bar_level_df) <- "Omic Type(s)"
bimodal_results_copy <- left_join(bar_level_df,  
                      bimodal_results_copy,
                      by = "Omic Type(s)")
bimodal_results_copy <- as.data.table(bimodal_results_copy)

ft <- flextable(bimodal_results_copy, col_keys = df_header$header_id)

final_ft <- setup_bold_table(cur_table = ft, header_df = df_header, bold_df = bold_df)

final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Bimodal_Best_By_DataType_table.docx")

## Trimodal Tables ====
require(data.table)
setDTthreads(8)
require(ggplot2)
require(gt)
require(flextable)
require(magrittr)
require(scales)
require(officer)
rmse <- function(x, y) sqrt(mean((x - y)^2))

set_flextable_defaults(
  font.size = 10, theme_fun = theme_vanilla,
  padding = 6,
  background.color = "#EFEFEF")

all_results <- fread("Data/all_results.csv")
all_results <- all_results[str_count(data_types, "_") == 1]

shared_combos <- fread("Data/shared_unique_combinations.csv")
shared_combos[, unique_samples := paste0(cpd_name, "_", cell_name)]

# Subset for bimodal results
# Subset by all shared samples all each data type
all_results[, unique_samples := paste0(cpd_name, "_", cell_name)]
all_results <- all_results[unique_samples %in% shared_combos$unique_samples]

all_results[, loss_by_config := rmse(target, predicted), by = c("data_types", "merge_method", "loss_type",
                                                                "drug_type", "split_method", "TargetRange",
                                                                "Targeted", "bottleneck")]
all_results <- unique(all_results[, c("data_types", "merge_method", "loss_type",
                                      "drug_type", "split_method", "TargetRange",
                                      "Targeted", "bottleneck", "loss_by_config")])
gc()

uniqueN(all_results$data_types)
# all_results[merge_method == "Merge By Concat"]$merge_method <- "Concat"
# all_results[merge_method == "Merge By Sum"]$merge_method <- "Sum"
# all_results[merge_method == "Base Model + LMF"]$merge_method <- "LMF"
# all_results[loss_type == "UnBase Model + LDS"]$loss_type <- "non-LDS"
# all_results[loss_type == "Base Model + LDS"]$loss_type <- "LDS"
# all_results[drug_type == "1024-bit ECFP"]$drug_type <- "ECFP"
# all_results[drug_type == "Base Model + GNN"]$drug_type <- "GNN"
# all_results[split_method == "Split By Both Cell Line & Drug Scaffold"]$split_method <- "Group Both"
# all_results[split_method == "Split By Drug Scaffold"]$split_method <- "Group Scaffold"
# all_results[split_method == "Split By Cell Line"]$split_method <- "Group Cell"
# all_results[split_method == "Split By Cancer Type"]$split_method <- "Group Cancer Type"
# all_results[TargetRange == "Target Above 0.7"]$TargetRange <- ">= 0.7"
# all_results[TargetRange == "Target Below 0.7"]$TargetRange <- "< 0.7"
# all_results[Targeted == "Targeted Drug"]$Targeted <- "Targeted"
# all_results[Targeted == "Untargeted Drug"]$Targeted <- "Untargeted"
# all_results[bottleneck == "No Data Bottleneck"]$bottleneck <- "No Bottleneck"
# all_results[bottleneck == "With Data Bottleneck"]$bottleneck <- "With Bottleneck"

### Best Model for Each Omic Data ====
trimodal_results_copy <- all_results
# Subset for no bottleneck
unique(trimodal_results_copy$bottleneck)
trimodal_results_copy <- trimodal_results_copy[bottleneck == "No Data Bottleneck"]
trimodal_results_copy$bottleneck <- NULL

unique(trimodal_results_copy[, c("merge_method", "loss_type", "drug_type")])

# Assign model names
trimodal_results_copy[(loss_type == "Base Model + LDS" & merge_method == "Base Model + LMF" & drug_type == "Base Model + GNN"), Method := "Trifecta"]
trimodal_results_copy[(loss_type == "Base Model" & merge_method == "Base Model" & drug_type == "Base Model"), Method := "Baseline"]
# Subset for baseline and trifecta models
trimodal_results_copy <- trimodal_results_copy[!is.na(Method)]
# Remove splitting by lineage (only done for trifecta models)
unique(trimodal_results_copy$split_method)
# trimodal_results_copy <- trimodal_results_copy[split_method != "Split By Lineage"]
# Subset TargetRange
trimodal_results_copy <- trimodal_results_copy[TargetRange == "Target Above 0.7"]

trimodal_results_copy$loss_type <- NULL
trimodal_results_copy$merge_method <- NULL
trimodal_results_copy$drug_type <- NULL

# Find the lowest CV mean by data types and drug type
trimodal_results_copy[, best_cv_mean := min(loss_by_config),
                      by = c("data_types", "split_method", "Targeted")]

# Subset models for those with best CV means
trimodal_results_copy <- trimodal_results_copy[loss_by_config == best_cv_mean]
colnames(trimodal_results_copy)[1] <- "Omic Type(s)"
trimodal_results_copy[, variable := paste(split_method, Targeted, TargetRange, sep = "_")]

# Find the best model overall for grouping method and drug type
trimodal_results_copy[, best_overall := min(loss_by_config), by = c("split_method", "Targeted")]
bolf_df <- trimodal_results_copy[loss_by_config == best_overall]

bold_df <- unique(bolf_df[, c("Omic Type(s)", "variable", "best_overall")])
bold_df$best_overall <- round(bold_df$best_overall, 3)
colnames(bold_df)[3] <- "best_cv_mean"
trimodal_results_copy$variable <- NULL
trimodal_results_copy$best_overall <- NULL

# trimodal_results_copy[, cv_mean := round(cv_mean, 3)]
trimodal_results_copy[, loss_by_config := round(loss_by_config, 3)]
# trimodal_results_copy[, cv_sd := round(cv_sd, 3)]
# trimodal_results_copy$Result <- paste(trimodal_results_copy$cv_mean, trimodal_results_copy$cv_sd, sep = " ± ")
trimodal_results_copy[, Result := as.character(loss_by_config)]
# trimodal_results_copy$cv_mean <- NULL
# trimodal_results_copy$cv_sd <- NULL
trimodal_results_copy <- unique(trimodal_results_copy)
# trimodal_results_copy <- dcast(trimodal_results_copy, ... ~ split_method + Method + Targeted + TargetRange,
#                         value.var = "Result")

# bimodal_results_copy[, cv_mean := round(cv_mean, 3)]
# bimodal_results_copy[, cv_sd := round(cv_sd, 3)]
# bimodal_results_copy$Result <- paste(bimodal_results_copy$cv_mean, bimodal_results_copy$cv_sd, sep = " ± ")
# bimodal_results_copy$cv_mean <- NULL
# bimodal_results_copy$cv_sd <- NULL
trimodal_results_copy$best_cv_mean <- NULL
trimodal_results_copy$loss_by_config <- NULL
# bimodal_results_copy <- unique(bimodal_results_copy)
trimodal_results_copy$Result <- paste(trimodal_results_copy$Method, trimodal_results_copy$Result, sep = "\n")
trimodal_results_copy$Method <- NULL

trimodal_results_copy_wide <- dcast(trimodal_results_copy, ... ~ split_method + Targeted + TargetRange,
                              value.var = "Result")

header_id_colnames <- colnames(trimodal_results_copy_wide)[-1]
header_split <- stringr::str_split(header_id_colnames, "_", simplify = T)
cur_header <- data.frame('header_id' = header_id_colnames,
                         'Var1' = header_split[,1],  # Grouping method
                         'Var2' = header_split[,2],  # Targeted or Untargeted
                         'Var3' = header_split[,3],  # AAC range
                         stringsAsFactors = F)
no_header <- data.table(header_id = "Omic Type(s)",
                        Var1 = "Omic Type(s)", Var2 = "Omic Type(s)",
                        Var3 = "Omic Type(s)")
df_header <- rbind(no_header, cur_header)

all_tri_omic_combos_el <- utils::combn(c("MUT", 'CNV', 'EXP', 'PROT', 'MIRNA', 'METAB', 'HIST', 'RPPA'), 2, simplify = T)
all_tri_omic_combos_el <- t(all_tri_omic_combos_el)
all_tri_omic_combos_el <- as.data.table(all_tri_omic_combos_el)

bar_level_df <- data.frame(temp = paste0(all_tri_omic_combos_el$V1, "_", all_tri_omic_combos_el$V2))

colnames(bar_level_df) <- "Omic Type(s)"
trimodal_results_copy_wide <- left_join(bar_level_df,  
                                        trimodal_results_copy_wide,
                                  by = "Omic Type(s)")
trimodal_results_copy_wide <- as.data.table(trimodal_results_copy_wide)

ft <- flextable(trimodal_results_copy_wide, col_keys = df_header$header_id)

final_ft <- setup_bold_table(ft, df_header, bold_df)

final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Trimodal_Best_By_DataType_table.docx")


## Multimodal Tables ====
require(data.table)
setDTthreads(8)
require(ggplot2)
require(gt)
require(flextable)
require(magrittr)
require(scales)
require(officer)
rmse <- function(x, y) sqrt(mean((x - y)^2))

set_flextable_defaults(
  font.size = 10, theme_fun = theme_vanilla,
  padding = 6,
  background.color = "#EFEFEF")

all_results <- fread("Data/all_results.csv")
all_results <- all_results[str_count(data_types, "_") > 1]

shared_combos <- fread("Data/shared_unique_combinations.csv")
shared_combos[, unique_samples := paste0(cpd_name, "_", cell_name)]

# Subset for bimodal results
# Subset by all shared samples all each data type
all_results[, unique_samples := paste0(cpd_name, "_", cell_name)]
all_results <- all_results[unique_samples %in% shared_combos$unique_samples]

all_results[, loss_by_config := rmse(target, predicted), by = c("data_types", "merge_method", "loss_type",
                                                                "drug_type", "split_method", "TargetRange",
                                                                "Targeted", "bottleneck")]
all_results <- unique(all_results[, c("data_types", "merge_method", "loss_type",
                                      "drug_type", "split_method", "TargetRange",
                                      "Targeted", "bottleneck", "loss_by_config")])
gc()

# all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange", "Targeted", "bottleneck")]
# all_results_long_copy[, cv_sd := sd(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange", "Targeted", "bottleneck")]

# all_results_long_copy$value <- NULL
# all_results_long_copy$variable <- NULL
# all_results_long_copy$fold <- NULL
# all_results_long_copy <- unique(all_results_long_copy)

# quadmodal_results <- all_results_long_copy[nchar(data_types) > 11]

# quadmodal_results[data_types == "CNV_EXP_METAB"]
# quadmodal_results[data_types == "CNV_EXP_PROT"]
# quadmodal_results[data_types == "CNV_EXP_PROT_METAB"]
# quadmodal_results[data_types == "CNV_EXP_PROT_MIRNA_METAB_HIST_RPPA"]
# quadmodal_results[data_types == "MUT_CNV_EXP_PROT_MIRNA_METAB_HIST_RPPA"]

# quadmodal_results[merge_method == "Merge By Concat"]$merge_method <- "Concat"
# quadmodal_results[merge_method == "Merge By Sum"]$merge_method <- "Sum"
# quadmodal_results[merge_method == "Base Model + LMF"]$merge_method <- "LMF"
# quadmodal_results[loss_type == "UnBase Model + LDS"]$loss_type <- "non-LDS"
# quadmodal_results[loss_type == "Base Model + LDS"]$loss_type <- "LDS"
# quadmodal_results[drug_type == "1024-bit ECFP"]$drug_type <- "ECFP"
# quadmodal_results[drug_type == "Base Model + GNN"]$drug_type <- "GNN"
# quadmodal_results[split_method == "Split By Both"]$split_method <- "Group Both"
# quadmodal_results[split_method == "Split By Drug Scaffold"]$split_method <- "Group Scaffold"
# quadmodal_results[split_method == "Split By Cell Line"]$split_method <- "Group Cell"
# quadmodal_results[TargetRange == "Target Above 0.7"]$TargetRange <- ">= 0.7"
# quadmodal_results[TargetRange == "Target Below 0.7"]$TargetRange <- "< 0.7"
# quadmodal_results[Targeted == "Targeted Drug"]$Targeted <- "Targeted"
# quadmodal_results[Targeted == "Untargeted Drug"]$Targeted <- "Untargeted"
# quadmodal_results[bottleneck == "No Data Bottleneck"]$bottleneck <- "No Bottleneck"
# quadmodal_results[bottleneck == "With Data Bottleneck"]$bottleneck <- "With Bottleneck"


### Best Model for Each Omic Data ====
quadmodal_results_copy <- all_results
quadmodal_results_copy[, data_types := gsub("_", " + ", data_types)]
# Subset for no bottleneck
quadmodal_results_copy <- quadmodal_results_copy[bottleneck == "No Data Bottleneck"]
quadmodal_results_copy$bottleneck <- NULL


unique(quadmodal_results_copy[, c("merge_method", "loss_type", "drug_type")])

# Assign model names
quadmodal_results_copy[(loss_type == "Base Model + LDS" & merge_method == "Base Model + LMF" & drug_type == "Base Model + GNN"), Method := "Trifecta"]
quadmodal_results_copy[(loss_type == "Base Model" & merge_method == "Base Model" & drug_type == "Base Model"), Method := "Baseline"]
# Subset for baseline and trifecta models
quadmodal_results_copy <- quadmodal_results_copy[!is.na(Method)]

table(quadmodal_results_copy$data_types)
table(quadmodal_results_copy$split_method)

# Remove splitting by lineage (it was only done for trimodal trifecta models)
quadmodal_results_copy <- quadmodal_results_copy[split_method != "Split By Cancer Type"]
# Subset TargetRange
quadmodal_results_copy <- quadmodal_results_copy[TargetRange == "Target Above 0.7"]

table(quadmodal_results_copy$data_types)
table(quadmodal_results_copy$split_method)

quadmodal_results_copy$loss_type <- NULL
quadmodal_results_copy$merge_method <- NULL
quadmodal_results_copy$drug_type <- NULL

# Find the lowest CV mean by data types and drug type
quadmodal_results_copy[, best_cv_mean := min(loss_by_config), by = c("data_types", "split_method", "Targeted")]

# Subset models for those with best CV means
quadmodal_results_copy <- quadmodal_results_copy[loss_by_config == best_cv_mean]
colnames(quadmodal_results_copy)[1] <- "Omic Type(s)"
quadmodal_results_copy[, variable := paste(split_method, Targeted, TargetRange, sep = "_")]

# Find the best model overall for grouping method and drug type
quadmodal_results_copy[, best_overall := min(loss_by_config), by = c("split_method", "Targeted")]
bolf_df <- quadmodal_results_copy[loss_by_config == best_overall]

bold_df <- unique(bolf_df[, c("Omic Type(s)", "variable", "best_overall")])
bold_df$best_overall <- round(bold_df$best_overall, 3)
colnames(bold_df)[3] <- "best_cv_mean"
quadmodal_results_copy$variable <- NULL
quadmodal_results_copy$best_overall <- NULL

quadmodal_results_copy[, loss_by_config := round(loss_by_config, 3)]
# quadmodal_results_copy[, cv_mean := round(cv_mean, 3)]
# quadmodal_results_copy[, cv_sd := round(cv_sd, 3)]
# quadmodal_results_copy$Result <- paste(quadmodal_results_copy$cv_mean, quadmodal_results_copy$cv_sd, sep = " ± ")
quadmodal_results_copy[, Result := as.character(loss_by_config)]
quadmodal_results_copy$loss_by_config <- NULL
# quadmodal_results_copy$cv_mean <- NULL
# quadmodal_results_copy$cv_sd <- NULL
quadmodal_results_copy <- unique(quadmodal_results_copy)
# quadmodal_results_copy <- dcast(quadmodal_results_copy, ... ~ split_method + Method + Targeted + TargetRange,
#                         value.var = "Result")

# bimodal_results_copy[, cv_mean := round(cv_mean, 3)]
# bimodal_results_copy[, cv_sd := round(cv_sd, 3)]
# bimodal_results_copy$Result <- paste(bimodal_results_copy$cv_mean, bimodal_results_copy$cv_sd, sep = " ± ")
# bimodal_results_copy$cv_mean <- NULL
# bimodal_results_copy$cv_sd <- NULL
quadmodal_results_copy$best_cv_mean <- NULL
# bimodal_results_copy <- unique(bimodal_results_copy)
quadmodal_results_copy$Result <- paste(quadmodal_results_copy$Method, quadmodal_results_copy$Result, sep = "\n")
quadmodal_results_copy$Method <- NULL

quadmodal_results_copy_wide <- dcast(quadmodal_results_copy, ... ~ split_method + Targeted + TargetRange,
                               value.var = "Result")

header_id_colnames <- colnames(quadmodal_results_copy_wide)[-1]
header_split <- stringr::str_split(header_id_colnames, "_", simplify = T)
cur_header <- data.frame('header_id' = header_id_colnames,
                         'Var1' = header_split[,1],  # Grouping method
                         'Var2' = header_split[,2],  # Targeted or Untargeted
                         'Var3' = header_split[,3],  # AAC range
                         stringsAsFactors = F)
no_header <- data.table(header_id = "Omic Type(s)",
                        Var1 = "Omic Type(s)", Var2 = "Omic Type(s)",
                        Var3 = "Omic Type(s)")
df_header <- rbind(no_header, cur_header)

ft <- flextable(quadmodal_results_copy_wide, col_keys = df_header$header_id)

final_ft <- setup_bold_table(ft, df_header, bold_df)

final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Multimodal_Best_By_DataType_table.docx")


# Comparison of Best Models from Each Omic Combination ====
require(data.table)
setDTthreads(8)
require(ggplot2)
require(tidytext)
rmse <- function(x, y) sqrt(mean((x - y)^2))

# For each cell line and drug combination, identify the best performing model overall per Splitting Method

# require(flextable)
# require(magrittr)
# require(scales)
# require(officer)

set_flextable_defaults(
  font.size = 10, theme_fun = theme_vanilla,
  padding = 6,
  background.color = "#EFEFEF")

all_results <- fread("Data/all_results.csv")


shared_combos <- fread("Data/shared_unique_combinations.csv")
shared_combos[, unique_samples := paste0(cpd_name, "_", cell_name)]

# Subset for bimodal results
# Subset by all shared samples all each data type
all_results[, unique_samples := paste0(cpd_name, "_", cell_name)]
all_results <- all_results[unique_samples %in% shared_combos$unique_samples]

all_results <- all_results[TargetRange == "Target Above 0.7"]
# Subset all results
all_results <- all_results[bottleneck != "With Data Bottleneck"]
all_results$bottleneck <- NULL

all_results[, loss_by_config := rmse(target, predicted), by = c("data_types", "merge_method", "loss_type",
                                                                "drug_type", "split_method", "TargetRange",
                                                                "Targeted")]
all_results <- unique(all_results[, c("data_types", "merge_method", "loss_type",
                                      "drug_type", "split_method", "TargetRange",
                                      "Targeted", "loss_by_config")])
gc()

# Assign Model Names
# Baseline and Elastic Net
all_results[(loss_type == "Base Model" & merge_method == "Merge By Early Concat" & drug_type == "Base Model"), model_name := "ElasticNet"]
all_results[(loss_type == "Base Model" & merge_method == "Base Model" & drug_type == "Base Model"), model_name := "Baseline"]
# Single Technique
all_results[(loss_type == "Base Model + LDS" & merge_method == "Base Model" & drug_type == "Base Model"), model_name := "LDS"]
all_results[(loss_type == "Base Model" & merge_method == "Base Model + LMF" & drug_type == "Base Model"), model_name := "LMF"]
all_results[(loss_type == "Base Model" & merge_method == "Base Model + Sum" & drug_type == "Base Model"), model_name := "Sum"]
all_results[(loss_type == "Base Model" & merge_method == "Base Model" & drug_type == "Base Model + GNN"), model_name := "GNN"]
# Two Techniques
all_results[(loss_type == "Base Model + LDS" & merge_method == "Base Model + LMF" & drug_type == "Base Model"), model_name := "LDS+LMF"]
all_results[(loss_type == "Base Model" & merge_method == "Base Model + LMF" & drug_type == "Base Model + GNN"), model_name := "LMF+GNN"]
all_results[(loss_type == "Base Model + LDS" & merge_method == "Base Model" & drug_type == "Base Model + GNN"), model_name := "LDS+GNN"]
# Three Techniques
all_results[(loss_type == "Base Model + LDS" & merge_method == "Base Model + Sum" & drug_type == "Base Model + GNN"), model_name := "LDS+Sum+GNN"]
all_results[(loss_type == "Base Model + LDS" & merge_method == "Base Model + LMF" & drug_type == "Base Model + GNN"), model_name := "Trifecta"]

sum(is.na(all_results$model_name))
all_results[is.na(model_name)]
table(all_results$model_name)

# Find the lowest CV mean by data types and drug type
all_results[, best_cv_mean := min(loss_by_config), by = c("data_types", "split_method", "Targeted")]

# Subset models for those with best CV means
all_results <- all_results[loss_by_config == best_cv_mean]
colnames(all_results)[1] <- "Omic Type(s)"
all_results[, variable := paste(split_method, Targeted, TargetRange, sep = "_")]

# Find the best model overall for grouping method and drug type
all_results[, best_overall := min(loss_by_config), by = c("split_method", "Targeted")]
all_results <- all_results[loss_by_config == best_overall]

# bold_df <- unique(bolf_df[, c("Omic Type(s)", "variable", "best_overall")])
# bold_df$best_overall <- round(bold_df$best_overall, 3)
# colnames(bold_df)[3] <- "best_cv_mean"

all_results[, loss_by_config := round(loss_by_config, 3)]
all_results[, Result := as.character(loss_by_config)]
all_results[, "Omic Type(s)" := gsub("_", "+", `Omic Type(s)`)]
all_results[ , Result := paste(`Omic Type(s)`, model_name, Result, sep = "\n")]
all_results$loss_by_config <- NULL
all_results$variable <- NULL
all_results$best_overall <- NULL
all_results$best_cv_mean <- NULL
# bimodal_results_copy <- unique(bimodal_results_copy)
all_results$model_name <- NULL
all_results$`Omic Type(s)` <- NULL
all_results$merge_method <- NULL
all_results$loss_type <- NULL
all_results$drug_type <- NULL

all_results <- all_results[split_method != "Split By Cancer Type"]
all_results_wide <- dcast(all_results, ... ~ split_method + Targeted + TargetRange,
                                     value.var = "Result")
# colnames(all_results_wide)[4] <- "Result"
header_id_colnames <- colnames(all_results_wide)[-1]
header_split <- stringr::str_split(header_id_colnames, "_", simplify = T)
cur_header <- data.frame('header_id' = header_id_colnames,
                         'Var1' = header_split[,1],  # Grouping method
                         'Var2' = header_split[,2],  # Targeted or Untargeted
                         'Var3' = header_split[,3],  # AAC range
                         stringsAsFactors = F)
no_header <- data.table(header_id = "Omic Type(s)",
                        Var1 = "Omic Type(s)", Var2 = "Omic Type(s)",
                        Var3 = "Omic Type(s)")
df_header <- rbind(no_header, cur_header)

# ft <- flextable(all_results_wide, col_keys = df_header$header_id)
ft <- flextable(all_results_wide, col_keys = df_header$header_id)

final_ft <- setup_bold_table(ft, df_header)

final_ft <- autofit(final_ft)
dir.create("Plots/CV_Tables")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/CV_Tables/Best_Model_By_SplitMethod_table.docx")


# Baseline  ElasticNet         GNN         LDS     LDS+GNN     LDS+LMF LDS+LMF+GNN LDS+Sum+GNN         LMF     LMF+GNN 
# 25397897      419855     6113536     6679193     4872726     4882025    31924617     2475755     5054873     4690297 
# Sum 
# 4882025 

# all_results$loss_type <- NULL
# all_results$merge_method <- NULL
# all_results$drug_type <- NULL
gc()


# Find the lowest CV mean by data types and drug type
all_results[, best_loss := min(RMSELoss), by = c("cpd_name", "cell_name", "split_method")]

quantile(all_results$target)
# Subset models for those with best CV means
all_the_best <- all_results[RMSELoss == best_loss]
# colnames(bimodal_results_copy)[1] <- "Data Type(s)"
# bimodal_results_copy[, variable := paste(split_method, Targeted, TargetRange, sep = "_")]

uniqueN(all_the_best[, c("cpd_name", "cell_name")])  # 309,594

sub <- unique(all_the_best[, c("cpd_name", "cell_name", "target")])
cur_quantile_func <- ecdf(unlist(sub[, "target", with = F]))

cur_quantile_func(0.7)  # 98.6th percentile...
quantile(sub$target)

# 0%        25%        50%        75%       100% 
# 0.00000000 0.02496015 0.09060940 0.21159975 0.99529000 

table(all_the_best$split_method)
table(all_the_best$model_name)

## Entire range results ====
all_the_best[, model_and_data := paste0(model_name, "-", data_types)]
all_the_best[, group_n := .N, by = "split_method"]
all_the_best[, model_and_data_freq := .N / group_n, by = c("split_method", "model_and_data")]
all_the_best[, model_freq := .N / group_n, by = c("split_method", "model_name")]
all_the_best[, data_freq := .N / group_n, by = c("split_method", "data_types")]

unique_sub <- unique(all_the_best[, c("split_method", "model_and_data_freq",
                        "model_name", "data_types", "model_and_data",
                        "model_freq", "data_freq")])

unique_sub <- unique_sub[split_method != "Split By Cancer Type"]
unique_sub$model_and_data <- NULL
unique_sub$model_and_data_freq <- NULL
# unique_sub$model_name <- NULL
# unique_sub$model_freq <- NULL

unique_sub$data_types <- NULL
unique_sub$data_freq <- NULL


unique_sub <- unique(unique_sub)
best_molten <- melt(unique_sub,
     id.vars = c("split_method", "model_name"), 
     measure.vars = "model_freq")

table(best_molten$variable)

require(tidytext)

# Plot top model types
ggplot(data = best_molten) +
  geom_bar(mapping = aes(x = reorder_within(x = model_name, by = -value, within = split_method), y = value, fill = model_name),
           stat = "identity", position = position_dodge2(width = 0.9, preserve = "single"),
           show.legend = F) +
  facet_wrap(~split_method, scales = "free_x") +
  geom_text(position = position_dodge2(width = 0.9, preserve = "single"),
            aes(x = reorder_within(x = model_name, by = -value, within = split_method), y = value+0.01, label=model_name, hjust=0), angle=90) +
  scale_y_continuous(limits = c(0, 0.65), breaks = seq(0, .65, by = 0.05)) +
  theme(axis.text.x = element_blank(),
        axis.ticks = element_blank()) +
  ylab("Frequency") +
  xlab("Model Name")

ggsave("Plots/CV_Results/Best_Model_Frequency_by_Split_Method_BarPlot.pdf")

unique_sub <- unique(all_the_best[, c("split_method", "model_and_data_freq",
                                      "model_name", "data_types", "model_and_data",
                                      "model_freq", "data_freq")])

unique_sub <- unique_sub[split_method != "Split By Cancer Type"]
unique_sub$model_and_data <- NULL
unique_sub$model_and_data_freq <- NULL
unique_sub$model_name <- NULL
unique_sub$model_freq <- NULL

# unique_sub$data_types <- NULL
# unique_sub$data_freq <- NULL

unique_sub <- unique(unique_sub)
best_molten <- melt(unique_sub,
                    id.vars = c("split_method", "data_types"), 
                    measure.vars = "data_freq")

# Plot top data types
ggplot(data = best_molten) +
  geom_bar(mapping = aes(x = reorder_within(x = data_types, by = -value, within = split_method), y = value, fill = data_types),
           stat = "identity", position = position_dodge2(width = 0.9, preserve = "single"),
           show.legend = F) +
  facet_wrap(~split_method, scales = "free_x") +
  geom_text(position = position_dodge2(width = 0.9, preserve = "single"),
            aes(x = reorder_within(x = data_types, by = -value, within = split_method), y = value+0.01, label=data_types, hjust=0), angle=90) +
  scale_y_continuous(limits = c(0, 0.65), breaks = seq(0, .65, by = 0.05)) +
  theme(axis.text.x = element_blank(),
        axis.ticks = element_blank()) +
  ylab("Frequency") +
  xlab("Model Name")

ggsave("Plots/CV_Results/Best_Data_Type_Frequency_by_Split_Method_BarPlot.pdf",
       width = 20)


## Upper Range Results ====
### Targeted Results ====
upper_aac_results <- all_results[target >= 0.7 & Targeted == "Targeted Drug"]
# Find the lowest CV mean by data types and drug type
upper_aac_results[, best_loss := min(RMSELoss),
                  by = c("cpd_name", "cell_name", "split_method", "Targeted")]

# Subset models for those with best CV means
upper_all_the_best <- upper_aac_results[RMSELoss == best_loss]
# colnames(bimodal_results_copy)[1] <- "Data Type(s)"
# bimodal_results_copy[, variable := paste(split_method, Targeted, TargetRange, sep = "_")]

table(upper_all_the_best$split_method)
table(upper_all_the_best$model_name)

upper_all_the_best[, model_and_data := paste0(model_name, "-", data_types)]
upper_all_the_best[, group_n := .N, by = "split_method"]
# upper_all_the_best[, model_and_data_freq := .N / group_n, by = c("split_method", "model_and_data")]
upper_all_the_best[, model_freq := .N / group_n, by = c("split_method", "model_name")]
upper_all_the_best[, data_freq := .N / group_n, by = c("split_method", "data_types")]

unique_sub <- unique(upper_all_the_best[, c("split_method",
                                      "model_name", "data_types",
                                      "model_freq", "data_freq")])

unique_sub <- unique_sub[split_method != "Split By Cancer Type"]
unique_sub$model_and_data <- NULL
# unique_sub$model_and_data_freq <- NULL
# unique_sub$model_name <- NULL
# unique_sub$model_freq <- NULL

unique_sub$data_types <- NULL
unique_sub$data_freq <- NULL

unique_sub <- unique(unique_sub)
best_molten <- melt(unique_sub,
                    id.vars = c("split_method", "model_name"), 
                    measure.vars = "model_freq")

table(best_molten$variable)

# Plot top model types
ggplot(data = best_molten) +
  geom_bar(mapping = aes(x = reorder_within(x = model_name, by = -value, within = split_method), y = value, fill = model_name),
           stat = "identity", position = position_dodge2(width = 0.9, preserve = "single"),
           show.legend = F) +
  facet_wrap(~split_method, scales = "free_x") +
  geom_text(position = position_dodge2(width = 0.9, preserve = "single"),
            aes(x = reorder_within(x = model_name, by = -value, within = split_method), y = value+0.01, label=model_name, hjust=0), angle=90) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.05)) +
  theme(text = element_text(size = 14, face = "bold"),
        axis.text.x = element_blank(),
        axis.ticks = element_blank()) +
  ylab("Frequency") +
  xlab("Model Name")

ggsave("Plots/CV_Results/Best_Model_Frequency_by_Split_Method_UpperAAC_0.7_Targeted_BarPlot.pdf")

unique_sub <- unique(upper_all_the_best[, c("split_method",
                                      "model_name", "data_types", "model_and_data",
                                      "model_freq", "data_freq")])

unique_sub <- unique_sub[split_method != "Split By Cancer Type"]
unique_sub$model_and_data <- NULL
# unique_sub$model_and_data_freq <- NULL
unique_sub$model_name <- NULL
unique_sub$model_freq <- NULL

# unique_sub$data_types <- NULL
# unique_sub$data_freq <- NULL

unique_sub <- unique(unique_sub)
best_molten <- melt(unique_sub,
                    id.vars = c("split_method", "data_types"), 
                    measure.vars = "data_freq")
best_molten <- best_molten[order(-value), head(.SD, 1000), by = "split_method"]
# setorder(best_molten, -value)
# Plot top data types
best_molten[, split_method := factor(split_method,
                                     levels = c("Split By Cell Line",
                                                "Split By Drug Scaffold",
                                                "Split By Both Cell Line & Drug Scaffold"))]

ggplot(data = best_molten) +
  geom_bar(mapping = aes(x = reorder_within(x = data_types, by = -value, within = split_method), y = value, fill = data_types),
           stat = "identity", position = position_dodge2(width = 0.9, preserve = "single"),
           show.legend = F) +
  facet_wrap(~split_method, scales = "free_x", ncol = 1) +
  geom_text(position = position_dodge2(width = 0.9, preserve = "single"),
            aes(x = reorder_within(x = data_types, by = -value, within = split_method), y = value+0.01, label=data_types, hjust=0), angle=90) +
  scale_y_continuous(limits = c(0, 0.3), breaks = seq(0, .3, by = 0.05)) +
  theme(text = element_text(size = 20, face = "bold"),
        axis.text.x = element_blank(),
        axis.ticks = element_blank()) +
  ylab("Frequency") +
  xlab("Model Name")

ggsave("Plots/CV_Results/Best_Data_Type_Frequency_by_Split_Method_UpperAAC_0.7_Targeted_BarPlot.pdf",
       height = 20, width = 16)

### Untargeted Drugs ====
upper_aac_results <- all_results[target >= 0.7 & Targeted == "Untargeted Drug"]
# Find the lowest CV mean by data types and drug type
upper_aac_results[, best_loss := min(RMSELoss),
                  by = c("cpd_name", "cell_name", "split_method", "Targeted")]

# Subset models for those with best CV means
upper_all_the_best <- upper_aac_results[RMSELoss == best_loss]
# colnames(bimodal_results_copy)[1] <- "Data Type(s)"
# bimodal_results_copy[, variable := paste(split_method, Targeted, TargetRange, sep = "_")]

table(upper_all_the_best$split_method)
table(upper_all_the_best$model_name)

upper_all_the_best[, model_and_data := paste0(model_name, "-", data_types)]
upper_all_the_best[, group_n := .N, by = "split_method"]
# upper_all_the_best[, model_and_data_freq := .N / group_n, by = c("split_method", "model_and_data")]
upper_all_the_best[, model_freq := .N / group_n, by = c("split_method", "model_name")]
upper_all_the_best[, data_freq := .N / group_n, by = c("split_method", "data_types")]

unique_sub <- unique(upper_all_the_best[, c("split_method",
                                            "model_name", "data_types",
                                            "model_freq", "data_freq")])

unique_sub <- unique_sub[split_method != "Split By Cancer Type"]
unique_sub$model_and_data <- NULL
# unique_sub$model_and_data_freq <- NULL
# unique_sub$model_name <- NULL
# unique_sub$model_freq <- NULL

unique_sub$data_types <- NULL
unique_sub$data_freq <- NULL

unique_sub <- unique(unique_sub)
best_molten <- melt(unique_sub,
                    id.vars = c("split_method", "model_name"), 
                    measure.vars = "model_freq")

table(best_molten$variable)

# Plot top model types
ggplot(data = best_molten) +
  geom_bar(mapping = aes(x = reorder_within(x = model_name, by = -value, within = split_method), y = value, fill = model_name),
           stat = "identity", position = position_dodge2(width = 0.9, preserve = "single"),
           show.legend = F) +
  facet_wrap(~split_method, scales = "free_x") +
  geom_text(position = position_dodge2(width = 0.9, preserve = "single"),
            aes(x = reorder_within(x = model_name, by = -value, within = split_method), y = value+0.01, label=model_name, hjust=0), angle=90) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.05)) +
  theme(text = element_text(size = 14, face = "bold"),
        axis.text.x = element_blank(),
        axis.ticks = element_blank()) +
  ylab("Frequency") +
  xlab("Model Name")

ggsave("Plots/CV_Results/Best_Model_Frequency_by_Split_Method_UpperAAC_0.7_Untargeted_BarPlot.pdf")

unique_sub <- unique(upper_all_the_best[, c("split_method",
                                            "model_name", "data_types", "model_and_data",
                                            "model_freq", "data_freq")])

unique_sub <- unique_sub[split_method != "Split By Cancer Type"]
unique_sub$model_and_data <- NULL
# unique_sub$model_and_data_freq <- NULL
unique_sub$model_name <- NULL
unique_sub$model_freq <- NULL

# unique_sub$data_types <- NULL
# unique_sub$data_freq <- NULL

unique_sub <- unique(unique_sub)
best_molten <- melt(unique_sub,
                    id.vars = c("split_method", "data_types"), 
                    measure.vars = "data_freq")
best_molten <- best_molten[order(-value), head(.SD, 1000), by = "split_method"]
# setorder(best_molten, -value)
# Plot top data types
best_molten[, split_method := factor(split_method,
                                     levels = c("Split By Cell Line",
                                                "Split By Drug Scaffold",
                                                "Split By Both Cell Line & Drug Scaffold"))]

ggplot(data = best_molten) +
  geom_bar(mapping = aes(x = reorder_within(x = data_types, by = -value, within = split_method), y = value, fill = data_types),
           stat = "identity", position = position_dodge2(width = 0.9, preserve = "single"),
           show.legend = F) +
  facet_wrap(~split_method, scales = "free_x", ncol = 1) +
  geom_text(position = position_dodge2(width = 0.9, preserve = "single"),
            aes(x = reorder_within(x = data_types, by = -value, within = split_method), y = value+0.01, label=data_types, hjust=0), angle=90) +
  scale_y_continuous(limits = c(0, 0.3), breaks = seq(0, .3, by = 0.05)) +
  theme(text = element_text(size = 20, face = "bold"),
        axis.text.x = element_blank(),
        axis.ticks = element_blank()) +
  ylab("Frequency") +
  xlab("Model Name")

ggsave("Plots/CV_Results/Best_Data_Type_Frequency_by_Split_Method_UpperAAC_0.7_Untargeted_BarPlot.pdf",
       height = 20, width = 16)


table(all_the_best[split_method == "Split By Cell Line"]$model_name)
table(all_the_best[split_method == "Split By Both Cell Line & Drug Scaffold"]$model_name)
table(all_the_best[split_method == "Split By Cancer Type"]$model_name)
table(all_the_best[split_method == "Split By Drug Scaffold"]$model_name)

table(all_the_best[split_method == "Split By Both Cell Line & Drug Scaffold"]$model_name)

cur_freqs <- unique(all_the_best[split_method == "Split By Both Cell Line & Drug Scaffold"][, c("model_name", "model_freq")])
colnames(cur_freqs) <- c("label", "freq")
cols <- rainbow(nrow(cur_freqs))
cur_freqs$percent = round(100*cur_freqs$freq/sum(cur_freqs$freq), digits = 1)
cur_freqs$final_label = paste(cur_freqs$label," (", cur_freqs$percent,"%)", sep = "")
model_both_pie <- pie(cur_freqs$freq, labels = cur_freqs$final_label, col = cols)


cur_freqs <- unique(all_the_best[split_method == "Split By Cell Line"][, c("model_name", "model_freq")])
colnames(cur_freqs) <- c("label", "freq")
cols <- rainbow(nrow(cur_freqs))
cur_freqs$percent = round(100*cur_freqs$freq/sum(cur_freqs$freq), digits = 1)
cur_freqs$final_label = paste(cur_freqs$label," (", cur_freqs$percent,"%)", sep = "")
model_cell_pie <- pie(cur_freqs$freq, labels = cur_freqs$final_label, col = cols)

cur_freqs <- unique(all_the_best[split_method == "Split By Drug Scaffold"][, c("model_name", "model_freq")])
colnames(cur_freqs) <- c("label", "freq")
cols <- rainbow(nrow(cur_freqs))
cur_freqs$percent = round(100*cur_freqs$freq/sum(cur_freqs$freq), digits = 1)
cur_freqs$final_label = paste(cur_freqs$label," (", cur_freqs$percent,"%)", sep = "")
model_drug_pie <- pie(cur_freqs$freq, labels = cur_freqs$final_label, col = cols, )

require(cowplot)

cowplot::plot_grid(model_both_pie, model_cell_pie, model_drug_pie)

table(all_the_best$model_and_data)
cur_freqs <- unique(all_the_best[split_method == "Split By Cell Line"][, c("model_and_data", "model_and_data_freq")])
cols <- rainbow(nrow(cur_freqs))
pie(cur_freqs$model_and_data_freq, labels = cur_freqs$model_and_data, col = cols)

table(all_the_best[split_method == "Split By Cell Line"]$model_name)
cur_freqs <- unique(all_the_best[split_method == "Split By Cell Line"][, c("model_name", "model_freq")])
cols <- rainbow(nrow(cur_freqs))
pie(cur_freqs$model_freq, labels = cur_freqs$model_name, col = cols)

cur_freqs <- unique(all_the_best[split_method == "Split By Cell Line"][, c("data_types", "data_freq")])
cols <- rainbow(nrow(cur_freqs))
pie(cur_freqs$data_freq, labels = cur_freqs$data_types, col = cols)
# MUT is surprisingly a good data type???
all_the_best[data_types == "MUT"]


cur_freqs <- unique(all_the_best[split_method == "Split By Both Cell Line & Drug Scaffold"][, c("model_and_data", "freq")])
cols <- rainbow(nrow(cur_freqs))


all_the_best[split_method == "Split By Drug Scaffold"]
all_the_best[split_method == "Split By Cell Line"]
all_the_best[split_method == "Split By Cancer Type"]

# Find the best model overall for grouping method and drug type
bimodal_results_copy[, best_overall := min(cv_mean), by = c("split_method", "Targeted")]
bolf_df <- bimodal_results_copy[cv_mean == best_overall]
bold_df <- unique(bolf_df[, c("Data Type(s)", "variable", "best_overall")])
bold_df$best_overall <- round(bold_df$best_overall, 3)
colnames(bold_df)[3] <- "best_cv_mean"
bimodal_results_copy$variable <- NULL
bimodal_results_copy$best_overall <- NULL

# all_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "Targeted", "bottleneck")]

# For each cell line and drug combination, identify the best performing model overall per Splitting Method

all_results_long_copy <- melt(unique(all_results[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config", "TargetRange", "Targeted", "bottleneck")]),
                              id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "Targeted", "bottleneck"))

# rm(all_results)
# gc()

all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange", "Targeted", "bottleneck")]
all_results_long_copy[, cv_sd := sd(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange", "Targeted", "bottleneck")]

all_results_long_copy$value <- NULL
all_results_long_copy$variable <- NULL
all_results_long_copy$fold <- NULL
all_results_long_copy <- unique(all_results_long_copy)

quadmodal_results <- all_results_long_copy[nchar(data_types) > 11]
table(quadmodal_results$data_types)
quadmodal_results[data_types == "CNV_EXP_METAB"]
quadmodal_results[data_types == "CNV_EXP_PROT"]
quadmodal_results[data_types == "CNV_EXP_PROT_METAB"]
quadmodal_results[data_types == "CNV_EXP_PROT_MIRNA_METAB_HIST_RPPA"]
quadmodal_results[data_types == "MUT_CNV_EXP_PROT_MIRNA_METAB_HIST_RPPA"]

quadmodal_results[merge_method == "Merge By Concat"]$merge_method <- "Concat"
quadmodal_results[merge_method == "Merge By Sum"]$merge_method <- "Sum"
quadmodal_results[merge_method == "Base Model + LMF"]$merge_method <- "LMF"
quadmodal_results[loss_type == "UnBase Model + LDS"]$loss_type <- "non-LDS"
quadmodal_results[loss_type == "Base Model + LDS"]$loss_type <- "LDS"
quadmodal_results[drug_type == "1024-bit ECFP"]$drug_type <- "ECFP"
quadmodal_results[drug_type == "Base Model + GNN"]$drug_type <- "GNN"
quadmodal_results[split_method == "Split By Both"]$split_method <- "Group Both"
quadmodal_results[split_method == "Split By Drug Scaffold"]$split_method <- "Group Scaffold"
quadmodal_results[split_method == "Split By Cell Line"]$split_method <- "Group Cell"
quadmodal_results[TargetRange == "Target Above 0.7"]$TargetRange <- ">= 0.7"
quadmodal_results[TargetRange == "Target Below 0.7"]$TargetRange <- "< 0.7"
quadmodal_results[Targeted == "Targeted Drug"]$Targeted <- "Targeted"
quadmodal_results[Targeted == "Untargeted Drug"]$Targeted <- "Untargeted"
quadmodal_results[bottleneck == "No Data Bottleneck"]$bottleneck <- "No Bottleneck"
quadmodal_results[bottleneck == "With Data Bottleneck"]$bottleneck <- "With Bottleneck"
