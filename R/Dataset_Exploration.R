# Dataset_exploration.R

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("PharmacoGx", version = "3.8")


library(PharmacoGx)
library(data.table)
??PharmacoSet

availablePSets()
ccle <- PharmacoGx::downloadPSet("CCLE_2013")
gdsc <- PharmacoGx::downloadPSet("GDSC_2013")
gcsi <- PharmacoGx::downloadPSet("gCSI")

common = intersectPSet(pSets = list(ccle, gdsc, gcsi),
                       intersectOn = c("drugs", "cell.lines"), strictIntersect = T, nthread = 2)
intersectPSet(pSets = list(ccle, gdsc),
                       intersectOn = "cell.lines", strictIntersect = F, nthread = 2)
fNames(ccle, 'rna')
fNames(gdsc, 'rna')
fNames(gcsi, 'rnaseq')

sensitivityMeasures(ccle)
unique(ccle@sensitivity$info[,-(1:2)])
ccle@sensitivity$raw
ccle@sensitivity$profiles

ccle_sum <- summarizeSensitivityProfiles(pSet = ccle, sensitivity.measure = "auc_published",
                                         drugs = "lapatinib")
ccle_sum[1:5]
# common = intersectPSet(pSets = list(ccle, gdsc),
#                        intersectOn = c("drugs", "cell.lines"), strictIntersect = T, nthread = 2)

# common_drugs = intersectPSet(pSets = list(ccle, gdsc, gcsi),
#                        intersectOn = c("drugs", "cell.lines"),
#                        drugs = c("Erlotinib", "Lapatinib", "Paclitaxel"), nthread = 2)
# common_drugs$CCLE@drug
# common_drugs$GDSC@drug
# common_drugs$GDSC@cell$tissueid
common$CCLE@drug$drug.name
common$CCLE@cell$cellid[common$CCLE@cell$tissueid == "breast"]

common$CCLE@drug$drug.name[common$CCLE@cell$tissueid == "breast"]

sensitivityMeasures(pSet = gdsc)
gdsc_auc <-
  summarizeSensitivityProfiles(
    pSet = gdsc,
    sensitivity.measure = "auc_published",
    summary.stat = "median",
    fill.missing = T
  )
ccle_auc <-
  summarizeSensitivityProfiles(
    pSet = ccle,
    sensitivity.measure = "auc_published",
    summary.stat = "median",
    fill.missing = T
  )
gcsi_auc <-
  summarizeSensitivityProfiles(
    pSet = gcsi,
    sensitivity.measure = "auc_recomputed",
    summary.stat = "median",
    fill.missing = T
  )

# Find breast tissue cell lines with Lapatinib tested on them
ccle@drug$drug.name == "Lapatinib"
cell_drug <- sensNumber(ccle)
cell_drug <- as.data.table(cell_drug, keep.rownames = T)
breast_cells <- ccle@cell$cellid[ccle@cell$tissueid == "breast"]
# All the cell lines in CCLE that have a breast origin and test Lapatinib
cell_drug[rn %in% breast_cells & lapatinib == 1]$rn

drugDoseResponseCurve(drug = "Lapatinib", cellline = "HARA",
                      pSets = ccle)
drugDoseResponseCurve(drug = "lapatinib", cellline = "HARA",
                      pSets = ccle)



ach <- fread("Data/Achilles/D2_combined_gene_dep_scores.csv")
ach <- fread("Data/Achilles/RNAseq_lRPKM_data.csv")
dim(ach)
ach[1:5, 1:5]
# ==== Read DepMap data ====
ccle_rna <- fread("Data/DepMap/CCLE_depMap_19Q1_TPM.csv")
ccle_transcripts <- fread("Data/DepMap/CCLE_depMap_19Q1_TPM_transcripts.csv")
dim(ccle_transcripts)
ccle_drug_data <- fread("Data/DepMap/CCLE_NP24.2009_Drug_data_2015.02.24.csv")
ccle_line_info <- fread("Data/DepMap/DepMap-2019q1-celllines_v2.csv")
dim(ccle_rna)

ccle_line_info

depmap_mutation <- fread("Data/DepMap/depmap_19Q1_mutation_calls.csv")
# Percentage DepMap cell line mutation data shared with CCLE
sum(ccle_rna$V1 %in% depmap_mutation$DepMap_ID) / length(ccle_rna$V1)
# Percentage GDSC cell lines in DepMap mutation data
sum(colnames(gdsc_auc)[-1] %in% depmap_mutation$DepMap_ID) / (ncol(gdsc_auc)-1)

gdsc_auc <- fread("Data/DepMap/GDSC_AUC.csv")
dim(gdsc_auc)
gdsc_auc$V1

ccle_rna[1:5, 1:5]
length(unique(ccle_line_info$DepMap_ID))

ccle_linenames <- gsub(pattern = "_.*", replacement = "", x = ccle_line_info[DepMap_ID %in% ccle_rna$V1]$CCLE_Name)
line_name_id <- data.table(DepMap_ID = ccle_line_info$DepMap_ID,
                           Name = gsub(pattern = "_.*", replacement = "", x = ccle_line_info$CCLE_Name))
# Percentage CTRPv2 cell lines with CCLE RNA expression data
sum(ctrp_cell_info$ccl_name %in% ccle_linenames) / length(ctrp_cell_info$ccl_name)
# Percentage CTRPv2 cell lines with DepMap mutation data
ctrp_depmap_id <- line_name_id[Name %in% ctrp_cell_info$ccl_name]$DepMap_ID
sum(ctrp_depmap_id %in% depmap_mutation$DepMap_ID) / length(ctrp_depmap_id)


sum(ctrp_cell_info$ccl_name %in% ccle_linenames) / length(ctrp_cell_info$ccl_name)

# Percentage GDSC cell lines in DepMap CCLE expression data
sum(colnames(gdsc_auc)[-1] %in% ccle_rna$V1) / length(colnames(gdsc_auc)[-1])

sum(ccle_line_info$DepMap_ID %in% ccle_rna$V1)

sum(colnames(gdsc_auc)[-1] %in% ccle_line_info$DepMap_ID) / (ncol(gdsc_auc)-1)
length(unique(colnames(gdsc_auc)))-1
length(unique(ccle_line_info$DepMap_ID))

ctrp_columns <- fread("Data/DepMap/CTRPv2.0_2015_ctd2_ExpandedDataset/v20._COLUMNS.txt")
unique(ctrp_columns[COLUMN_HEADER == "master_cpd_id", "COLUMN_DESCRIPTION"])
unique(ctrp_columns[COLUMN_HEADER == "experiment_id", "COLUMN_DESCRIPTION"])
unique(ctrp_columns[COLUMN_HEADER == "cpd_pred_pv", "COLUMN_DESCRIPTION"])
unique(ctrp_columns[COLUMN_HEADER == "cpd_avg_pv", "COLUMN_DESCRIPTION"])
unique(ctrp_columns[COLUMN_HEADER == "master_ccl_id", "COLUMN_DESCRIPTION"])
unique(ctrp_columns[COLUMN_HEADER == "baseline_signal", "COLUMN_DESCRIPTION"])


ctrp_plate <- fread("Data/DepMap/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.meta.per_assay_plate.txt")
unique(ctrp_plate$assay_plate_barcode)
ctrp_data <- fread("Data/DepMap/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.data.per_cpd_post_qc.txt")
ctrp_experiment <- fread("Data/DepMap/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.meta.per_experiment.txt")
ctrp_drug_data <- fread("Data/DepMap/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.data.per_cpd_avg.txt")
ctrp_line_info <- fread("Data/DepMap/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.meta.per_cell_line.txt")

ctrp_data[experiment_id == 1 & master_cpd_id == 1788]

length(unique(ctrp_drug_data$assay_plate_barcode))
ctrp_drug_data[experiment_id == 1]
ctrp_experiment$experiment_id

ctrp_master <- merge(x = ctrp_data, y = ctrp_drug_data, by = "experiment_id")
ctrp_drug_info <- fread("Data/DepMap/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.meta.per_compound.txt")


