# repurposable_drugs_interpretation.R
require(data.table)
ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")
# gdsc2 <- fread("Data/DRP_Training_Data/GDSC2_AAC_SMILES.txt")
get_all_interpret <- function(data_types, split_type) {
  cv_path = paste0("Data/CV_Results/HyperOpt_DRP_ResponseOnly_", data_types, "_HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_", split_type, "_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_", data_types, "/")
  interpret_paths = list.files(path = cv_path, pattern = ".*final_interpretation.*", full.names = T)
  all_files = vector(mode = "list", length = length(interpret_paths))
  for (i in 1:length(interpret_paths)) {
    cur_file <- fread(interpret_paths[i])
    all_files[[i]] <- cur_file
  }
  final_set = rbindlist(all_files)
  return(final_set)
}
get_top_attrs <- function(integ_data, compound_name, cell_line_name) {
  
  cur_integ <- integ_data[cpd_name == compound_name & cell_name == cell_line_name]
  cur_integ$V1 <- NULL
  cur_integ_melt <- melt(cur_integ, id.vars = c("cpd_name", "cell_name", "target", "predicted", "RMSE_loss", "interpret_delta"))
  setorder(cur_integ_melt, -value)
  return(cur_integ_melt)
}

path_dict <- vector("list", length = 8)
path_dict[['MUT']] <- "Data/DRP_Training_Data/DepMap_21Q2_Mutations_by_Cell.csv"
path_dict[['CNV']] <- "Data/DRP_Training_Data/DepMap_21Q2_CopyNumber.csv"
path_dict[['EXP']] <- "Data/DRP_Training_Data/DepMap_21Q2_Expression.csv"
path_dict[['PROT']] <- "Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv"
path_dict[['MIRNA']] <- "Data/DRP_Training_Data/DepMap_2019_miRNA.csv"
path_dict[['METAB']] <- "Data/DRP_Training_Data/DepMap_2019_Metabolomics.csv"
path_dict[['HIST']] <- "Data/DRP_Training_Data/DepMap_2019_ChromatinProfiling.csv"
path_dict[['RPPA']] <- "Data/DRP_Training_Data/DepMap_2019_RPPA.csv"

get_expression_in_cell_line <- function(data_type, cell_line_name, variable_name) {
  cur_path <- path_dict[[data_type]]
  cur_depmap <- fread(cur_path)
  # dim(depmap_mirna)
  # depmap_mirna[1:10, 1:10]
  # Get primary disease of the given cell line
  cur_cancer_type <- unique(cur_depmap[stripped_cell_line_name == cell_line_name]$primary_disease)
  cur_col_idx <- which(colnames(cur_depmap) == variable_name)
  print(cur_depmap[stripped_cell_line_name == cell_line_name, c(1:2, cur_col_idx), with = F])
  
  if (data_type != "MUT") {
    cur_value <- unlist(cur_depmap[stripped_cell_line_name == cell_line_name, cur_col_idx, with = F])
    cur_quantile_func <- ecdf(unlist(cur_depmap[, cur_col_idx, with = F]))
    print("Expression Percentile in All Cell Lines")
    print(cur_quantile_func(cur_value))
    
    cur_quantile_func <- ecdf(unlist(cur_depmap[primary_disease == cur_cancer_type, cur_col_idx, with = F]))
    print("Expression Percentile in Cell Lines from the Same Cancer Type")
    print(cur_quantile_func(cur_value))
  } else {
    print(cur_depmap[primary_disease == cur_cancer_type, cur_col_idx, with = F])
  }
  
}
# require(clusterProfiler)
# require(pathview)
# organism = "org.Hs.eg.db"
# # BiocManager::install(organism, character.only = TRUE)
# library(organism, character.only = TRUE)
# keytypes(get(organism))
# org.Hs.eg.db

# ggsave("Plots/Interpretation/IntegratedGradients//gnndrug_prot_697_Paclitaxel_GSE_bottom_5.pdf", p_bottom_prot, 
#        width = 20, units = "in")
# Find the best re-purposable drugs, perform IntegratedGradients Interpretation

# final_data <- fread("Data/repurposable_drugs_table.csv")
final_data <- fread("Data/repurposable_drugs_table.csv")

# Regorafenib ----
# Regorafenib (BAY 73-4506, Stivarga®) is an oral diphenylurea multi-kinase inhibitor
# that targets angiogenic (VEGFR1-3, TIE2), stromal (PDGFR-β, FGFR), and oncogenic
# receptor tyrosine kinases (KIT, RET, and RAF).

final_data[cpd_name == "Regorafenib"]
# Assigned for Colorectal Cancer, Good for Leukemia (AML)
# Currently in Phase I clinical trials (Study of Regorafenib in Patients With Advanced Myeloid Malignancies)
# https://clinicaltrials.gov/ct2/show/NCT03042689
# EOL1 cell line AAC: 0.992
# MV411 cell line AAC: 0.524
# Best data type for EOL1 is MUT_PROT

mut_prot_integ <- get_all_interpret(data_types = "gnndrug_mut_prot", split_type = "CELL_LINE")
dim(mut_prot_integ)
min(mut_prot_integ$target)
max(mut_prot_integ$target)
max(mut_prot_integ$RMSE_loss)

# mirna_metab_integ <- fread("Data/CV_Results/HyperOpt_DRP_ResponseOnly_gnndrug_mirna_metab_HyperOpt_DRP_CTRP_ResponseOnly_EncoderTrain_Split_CELL_LINE_NoBottleNeck_NoTCGAPretrain_MergeByLMF_WeightedRMSELoss_GNNDrugs_gnndrug_mirna_metab/inter")
colnames(mut_prot_integ)[1:10]
unique(mut_prot_integ$cell_name)

## EOL1 ----

### AAC vs EXP/CNV ====
mut_prot_integ <- get_all_interpret(data_types = "gnndrug_mut_prot", split_type = "CELL_LINE")
cnv_exp_integ <- get_all_interpret(data_types = "gnndrug_cnv_exp", split_type = "CELL_LINE")
regorafenib_eol1_attrs <- get_top_attrs(mut_prot_integ, compound_name = "Regorafenib",
                                        cell_line_name = "EOL1")
regorafenib_eol1_attrs_cnv_exp <- get_top_attrs(cnv_exp_integ, compound_name = "Regorafenib",
                                        cell_line_name = "EOL1")

regorafenib_eol1_attrs[variable %like% "VEGF"]  # very small value...
regorafenib_eol1_attrs_cnv_exp[variable %like% "VEGF"]  # very small value...
regorafenib_eol1_attrs[variable %like% "TIE2"]  # very small value...
regorafenib_eol1_attrs[variable %like% "PDFGFR"]  # very small value...
regorafenib_eol1_attrs[variable %like% "FGFR1"]  # very small value...
regorafenib_eol1_attrs[variable %like% "FGFR2"]  # very small value...
regorafenib_eol1_attrs[variable %like% "FGFR3"]  # very small value...
regorafenib_eol1_attrs[variable %like% "FGFR4"]  # very small value...
regorafenib_eol1_attrs[variable %like% "FGFRL1"]  # very small value...
regorafenib_eol1_attrs[variable %like% "KIT"]  # very small value...
regorafenib_eol1_attrs[variable %like% "RET"]  # very small value...
regorafenib_eol1_attrs[variable %like% "RAF"]  # very small value...

# NONE OF THE KINASES ARE ACTUALLY IN THE PROTEIN QUANTIFICATION DATA
regorafenib_eol1_attrs[variable %like% "P17948"]
regorafenib_eol1_attrs[variable %like% "P35968"]
regorafenib_eol1_attrs[variable %like% "P35916"]
regorafenib_eol1_attrs[variable %like% "Q02763"]
regorafenib_eol1_attrs[variable %like% "P16234"]
regorafenib_eol1_attrs[variable %like% "P10721"]
regorafenib_eol1_attrs[variable %like% "P07949"]

MUT <- fread("Data/DRP_Training_Data/DepMap_21Q2_Mutations_by_Cell.csv")
CNV <- fread("Data/DRP_Training_Data/DepMap_21Q2_CopyNumber.csv")
EXP <- fread("Data/DRP_Training_Data/DepMap_21Q2_Expression.csv")

# Check EXP data for expression
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "VEGFA")
# 96th percentile in all, 96 in leukemia
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "TIE2")
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "PDFGFR")
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "FGFR1")
# 37th percentile in all, 54 in leukemia
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "FGFR2")
# 23th percentile in all, 57 in leukemia
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "FGFR3")
# 67th percentile in all, 85 in leukemia
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "FGFR4")
# 7th percentile in all, 18 in leukemia
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "FGFRL1")
# 49th percentile in all, 78 in leukemia
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "KIT")
# 68th percentile in all, 54 in leukemia
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "RET")
# 92th percentile in all, 86th in leukemia
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "RAF1")
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "BRAF")

# Check PROT data for expression
get_expression_in_cell_line(data_type = "PROT", cell_line_name = "EOL1", variable_name = "P17948")  # VEGFR1
get_expression_in_cell_line(data_type = "PROT", cell_line_name = "EOL1", variable_name = "P35968")  # VEGFR2
get_expression_in_cell_line(data_type = "PROT", cell_line_name = "EOL1", variable_name = "P35916")  # VEGFR3
get_expression_in_cell_line(data_type = "PROT", cell_line_name = "EOL1", variable_name = "Q02763")  # TIE2
get_expression_in_cell_line(data_type = "PROT", cell_line_name = "EOL1", variable_name = "P16234")  # PDFGFR
get_expression_in_cell_line(data_type = "PROT", cell_line_name = "EOL1", variable_name = "P10721")  # KIT
get_expression_in_cell_line(data_type = "PROT", cell_line_name = "EOL1", variable_name = "P07949")  # RET

# Check MUT data for mutations 
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "VEGFA")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "TIE2")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "PDFGFR")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "FGFR1")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "FGFR2")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "FGFR3")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "FGFR4")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "FGFRL1")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "KIT")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "RET")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "RAF1")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "BRAF")


# Expression of BTK is in the 46th percentile (among all cell lines)
get_expression_in_cell_line(data_type = "CNV", cell_line_name = "EFM192A", variable_name = "BTK")
# Expression of BTK is in the 46th percentile (among all cell lines)

drug_name <- "Regorafenib"
gene_name <- "BTK"
cell_line_name <- "EFM192A"
# Find expression of BTK in all cell lines that ibrutinib was tested in, plot it against AAC 
cur_cell_aac <- unique(ctrp[cpd_name == drug_name][, c("ccl_name", "area_above_curve")])
cur_exp_subset <- EXP[stripped_cell_line_name %in% unique(cur_cell_aac$ccl_name)][, c("stripped_cell_line_name",
                                                                                      gene_name), with = F]
exp_aac_subset <- merge(cur_exp_subset, cur_cell_aac, by.x = "stripped_cell_line_name", by.y = "ccl_name")
colnames(exp_aac_subset)[2] <- "value"

# Find copy number of BTK in all cell lines that ibrutinib was tested in, plot it against AAC 
cur_cnv_subset <- CNV[stripped_cell_line_name %in% unique(cur_cell_aac$ccl_name)][, c("stripped_cell_line_name",
                                                                                      gene_name), with = F]
cnv_aac_subset <- merge(cur_cnv_subset, cur_cell_aac, by.x = "stripped_cell_line_name", by.y = "ccl_name")
colnames(cnv_aac_subset)[2] <- "value"

require(ggplot2)
require(patchwork)
p_exp_aac <- ggplot(data = exp_aac_subset) +
  geom_point(aes(x = area_above_curve, y = value)) +
  xlab("Area Above Curve") +
  ylab("Gene Expression") +
  annotate(geom = "point",
           x = exp_aac_subset[stripped_cell_line_name == cell_line_name]$area_above_curve,
           y = exp_aac_subset[stripped_cell_line_name == cell_line_name]$value,
           colour = "orange", size = 1) + 
  annotate(
    geom = "curve",
    x = 0.6, y = 2,
    xend = exp_aac_subset[stripped_cell_line_name == cell_line_name]$area_above_curve - 0.005,
    yend = exp_aac_subset[stripped_cell_line_name == cell_line_name]$value + 0.2, 
    curvature = -.3, arrow = arrow(length = unit(2, "mm"))
  ) +
  annotate(geom = "text", x = 0.6, y = 2.3, label = "EFM192A", size = 6,
  ) +
  theme(text = element_text(size = 14, face = "bold"))

p_cnv_aac <- ggplot(data = cnv_aac_subset) +
  geom_point(aes(x = area_above_curve, y = value)) +
  xlab("Area Above Curve") +
  ylab("Copy Number") +
  annotate(geom = "point",
           x = cnv_aac_subset[stripped_cell_line_name == cell_line_name]$area_above_curve,
           y = cnv_aac_subset[stripped_cell_line_name == cell_line_name]$value,
           colour = "orange", size = 1) + 
  annotate(
    geom = "curve",
    x = 0.6, y = 1.35,
    xend = cnv_aac_subset[stripped_cell_line_name == cell_line_name]$area_above_curve - 0.005,
    yend = cnv_aac_subset[stripped_cell_line_name == cell_line_name]$value + 0.05, 
    curvature = -.3, arrow = arrow(length = unit(2, "mm"))
  ) +
  annotate(geom = "text", x = 0.6, y = 1.4, label = "EFM192A", size = 6, 
  ) +
  theme(text = element_text(size = 14, face = "bold"))



p_exp_aac + p_cnv_aac 


ggsave("Plots/Interpretation/BTK_EXP_CNV_vs_AAC.pdf")


### Initial analysis ====
regorafenib_eol1_attrs <- get_top_attrs(mut_prot_integ, compound_name = "Regorafenib",
                                         cell_line_name = "EOL1")

regorafenib_eol1_attrs[variable %like% "VEGF"]  # very small value...
regorafenib_eol1_attrs[variable %like% "TIE2"]  # very small value...
regorafenib_eol1_attrs[variable %like% "PDFGFR"]  # very small value...
regorafenib_eol1_attrs[variable %like% "FGFR"]  # very small value...
regorafenib_eol1_attrs[variable %like% "KIT"]  # very small value...
regorafenib_eol1_attrs[variable %like% "RET"]  # very small value...
regorafenib_eol1_attrs[variable %like% "RAF"]  # very small value...

# Top positive attributed variables:
head(regorafenib_eol1_attrs)

# Top 3 positive attributions:
# (all in mutational data)
depmap_mut <- fread("Data/DepMap/21Q2/CCLE_mutations.csv")
# n_TAS2R20
# Paper: The Role of Bitter Taste Receptors in Cancer: A Systematic Review (2021)
# the agonist-related activation and overexpression of TAS2Rs per se induced various anti-cancer effects, leading to the
# hypothesis that TAS2Rs impact carcinogenesis and could serve as a target in cancer therapy by interfering with typical
# capabilities of cancerous cells, known as the hallmarks of cancer
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "TAS2R20")
get_expression_in_cell_line(data_type = "PROT", cell_line_name = "EOL1", variable_name = "TAS2R20")
get_expression_in_cell_line(data_type = "CNV", cell_line_name = "EOL1", variable_name = "TAS2R20")

sum(temp$TAS2R20)  # 1 only
# ACH-000198 cell line is EOL1
depmap_mut[DepMap_ID == "ACH-000198" & Hugo_Symbol %like% "TAS2R20"]

# n_OSBP2
# Paper: hHLM/OSBP2 is expressed in chronic myeloid leukemia (2003)
# Oxysterols are oxygenated derivatives of cholesterol that have been shown to influence a wide variety of cellular
# processes including sterol metabolism, lipid trafficking, apoptosis and more recently, cell differentiation.
# The oxysterol binding proteins (OSBPs) comprise a large conserved family of proteins in eukaryotes with high affinity 
# for oxysterols, but their precise function has not been defined yet. One member of this family in humans, HLM/OSBP2 
# protein, has recently been reported as a potential marker for solid tumor dissemination and worse prognosis in these cases.
# In this study we focused on the evaluation of HLM/OSBP2 expression in malignant cell lines from different origins
# (blood and solid tumors) and we also evaluated its expression in chronic myeloid leukemia patients, correlating the
# molecular findings with clinical outcome. Our results showed that HLM/OSBP2 was expressed in 80% of the analysed CML 
# patients, suggesting that this protein could constitute a helpful tool for disease monitoring and reinforces recent 
# findings that HLM/OSBP2 protein could be involved in the maintenance of the undifferentiated state necessary for
# leukemogenesis

temp <- get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "OSBP2")
sum(temp$OSBP2)  # only 1
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "OSBP2")
get_expression_in_cell_line(data_type = "PROT", cell_line_name = "EOL1", variable_name = "OSBP2")

depmap_mut[DepMap_ID == "ACH-000198" & Hugo_Symbol %like% "OSBP2"]


# n_SLC1A6
# Papers: Investigating the microRNA-mRNA regulatory network in acute myeloid leukemia (2017)
# SLC1A3 contributes to L-asparaginase resistance in solid tumors (2019)
# https://www.genecards.org/cgi-bin/carddisp.pl?gene=SLC1A6

get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "SLC1A6")
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EOL1", variable_name = "SLC1A3")
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "SLC1A6")
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EOL1", variable_name = "SLC1A3")

# Top 2 negative attributions:
tail(cur_integ_melt)


## Regorafenib FlexTable ====
require(flextable)
require(magrittr)
require(scales)
require(officer)

regorafenib_final_data <- final_data[cpd_name == "Regorafenib" &
                                       primary_disease == "Leukemia"]
regorafenib_final_data <- regorafenib_final_data[, head(.SD, 1), by = "cell_name"]

setcolorder(regorafenib_final_data, c(
  "cell_name", "lineage_subtype",
  "data_types", "target", "pred", "RMSE"))

regorafenib_final_data[, data_types := gsub("_", "+", data_types)]
regorafenib_final_data$cpd_name <- NULL
regorafenib_final_data$primary_disease <- NULL
regorafenib_final_data$assigned_disease <- NULL
regorafenib_final_data$lineage <- NULL
regorafenib_final_data$highest_drug_match_disease_aac <- NULL
regorafenib_final_data$split_method <- NULL
regorafenib_final_data$lineage <- NULL
# ibrutinib_final_data$lineage_subtype <- NULL

# "Data Type(s)", "Split Method", "True AAC", "Prediction", "RMSE Loss"
colnames(regorafenib_final_data) <- c(
  "Cell Line", "Lineage Subtype",
  "Data Type(s)", "True AAC", "Prediction", "MAE Loss")
regorafenib_final_data <- unique(regorafenib_final_data)

# regorafenib_final_data$Lineage <- tools::toTitleCase(gsub("_", " ", regorafenib_final_data$Lineage))
regorafenib_final_data$`Lineage Subtype` <- tools::toTitleCase(gsub("_", " ", regorafenib_final_data$`Lineage Subtype`))
ft <- flextable(regorafenib_final_data)
final_ft <- ft %>%
  # merge_v(j = c("Cancer", "Prescribed Drug(s)", "Cell Line Primary Disease")) %>%
  merge_v() %>%
  border_inner(border = fp_border(color="gray", width = 1)) %>%
  border_outer(part="all", border = fp_border(color="gray", width = 2)) %>%
  align(align = "center", part = "all")

final_ft <- autofit(final_ft)

dir.create("Plots/Drug_Tables/")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/Drug_Tables/Regorafenib_Table.docx")

# Axitinib ----
final_data[cpd_name == "Axitinib"]
# Assigned for Kidney Cancer, Good for Blood (AML and CML) and Lung (NSCLC)
# "Axitinib effectively inhibits BCR-ABL1(T315I) with a distinct binding conformation" (March 2015) for Leukemia
# "Axitinib for the treatment of advanced non-small-cell lung cancer" (June 2013) for NSCLC Lung Cancer


# ==== Axitinib and EOL1 (AML)
prot_mirna_integ <- get_all_interpret(data_types = "gnndrug_prot_mirna", split_type = "CELL_LINE")
axitinib_eol1_attrs <- get_top_attrs(prot_mirna_integ, compound_name = "Axitinib", cell_line_name = "EOL1")

# Top positive attributed variables:
head(axitinib_eol1_attrs)

# hsv2-miR-H2 (viral miRNA)
get_expression_in_cell_line(data_type = "MIRNA", cell_line_name = "EOL1", variable_name = "hsv2-miR-H2")
# Expression of hsv2-miR-H2 is in ~99 percentile, high compared to other cell lines
# This cell line may be infected with some virus (e.g. herpes) or that this miRNA has oncogenic properties in AML (EOL1)
# It's show to have oncogenic activity in prostate cancer
# Paper: Increased Expression of Herpes Virus-Encoded hsv1-miR-H18 and hsv2-miR-H9-5p in Cancer-Containing Prostate Tissue
# Compared to That in Benign Prostate Hyperplasia Tissue (2016)
# https://www.cancer.org/cancer/cancer-causes/infectious-agents/infections-that-can-lead-to-cancer/viruses.html

# miR-1247
# Paper: Epigenetically altered miR-1247 functions as a tumor suppressor in pancreatic cancer

# Top negative attributed variables
tail(axitinib_eol1_attrs)
# hsa-miR-124
get_expression_in_cell_line(data_type = "MIRNA", cell_line_name = "EOL1", variable_name = "hsa-miR-124")
# 99 percentile, is a tumor suppressor!
# Paper: Methylation-mediated silencing and tumour suppressive function of hsa-miR-124 in cervical cancer (2010)

# Belinostat ----
final_data[cpd_name == "Belinostat"]
# Assigned for Lymphoma, Good for Neuroblastoma (pediatric) and Leukemia (AML)
# Selective Inhibition of HDAC Class I Sensitizes Leukemia and Neuroblastoma Cells to Anticancer Drugs (2021)

exp_rppa_integ <- get_all_interpret(data_types = "gnndrug_exp_rppa", split_type = "CELL_LINE")
belinostat_imr32_attrs <- get_top_attrs(exp_rppa_integ, compound_name = "Belinostat", cell_line_name = "IMR32")

# Top positive attributed variables:
head(belinostat_imr32_attrs)
# YAP_pS127, YAP Protein (only when phosphorylated at serine 127)
# Potential target for cancer treatment: https://www.genecards.org/cgi-bin/carddisp.pl?gene=YAP1
# Paper: Histone Acetylation-Mediated Regulation of the Hippo Pathway (2013)
# YAP is involved in the Hippo pathway, which is related to angiogenesis
# Paper: YAP and the Hippo pathway in pediatric cancer (2017)
# IMR32 was sampled from a pediatric tumor sample 

# HER3 / ERBB
# Paper: Signaling of ERBB Receptor Tyrosine Kinases Promotes Neuroblastoma Growth in vitro and in vivo

# PRKCD (PKC delta)
# Interacts with ERBB2

# Top negative attributed variables
tail(belinostat_imr32_attrs)
# Raptor (RPTOR), involved in the mTOR pathway
get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "IMR32", variable_name = "Raptor")
# 100 percentile, highly express, so is it a tumor supporessor?
# Paper: mTOR Interacts with Raptor to Form a Nutrient-Sensitive Complex that Signals to the Cell Growth Machinery (2002)
# "...The association of raptor with mTOR also negatively regulates the mTOR kinase activity..."

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "IMR32", variable_name = "GATA3")
# 96th percentile, highly expressed, is it a tumor suppressor?
# Paper: GATA-3 expression in breast cancer has a strong association with estrogen receptor but lacks independent prognostic value (2008)
# ...In univariate analysis, the presence of GATA-3 is a marker of good prognosis and predicted for superior breast
# cancer-specific survival, relapse-free survival, and overall survival...
# Paper: GATA3 is a reliable marker for neuroblastoma in limited samples, including FNA Cell Blocks, core biopsies,
# and touch imprints (2017)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "IMR32", variable_name = "Bak_Caution")
# 94th percentile, Highly expressed
# https://www.genecards.org/cgi-bin/carddisp.pl?gene=BAK1
# Is pro-apoptotic, and also interacts with p53


# Bosutinib ----
final_data[cpd_name == "Bosutinib"]
# Assigned for Leukemia, Good for Lung cancer (NSCLC)
# Bosutinib inhibits migration and invasion via ACK1 in KRAS mutant non-small cell lung cancer (2014)
# MUT_CNV has lowest RMSE

mut_cnv_integ <- get_all_interpret(data_types = "gnndrug_mut_cnv", split_type = "CELL_LINE")
bosutinib_pc14_attrs <- get_top_attrs(mut_cnv_integ, compound_name = "Bosutinib", cell_line_name = "PC14")

# Top positive attributed variables:
head(bosutinib_pc14_attrs)

# OR2T34
get_expression_in_cell_line(data_type = "CNV", cell_line_name = "PC14", variable_name = "OR2T34")
# ~ 100th percentile, is it oncogenic?
# Genetic Features of Lung Adenocarcinoma with Ground-Glass Opacity: What Causes the Invasiveness of Lung Adenocarcinoma? (2020)
# ... Among the mutant genes commonly expressed in GGO and non-GGO LUAD, the top 10 most significant genes were OR2T34...
# ...Among them, OR2T34 was the most frequently-appearing gene...

get_expression_in_cell_line(data_type = "CNV", cell_line_name = "PC14", variable_name = "FAM25E")
# ~ 94th percentile, is it oncogenic?


# Top negative attributed variables
tail(bosutinib_pc14_attrs)

get_expression_in_cell_line(data_type = "CNV", cell_line_name = "PC14", variable_name = "KDM5A")
# ~ 100th percentile, is it a tumor suppressor?
# https://www.genecards.org/cgi-bin/carddisp.pl?gene=KDM5A
# Implicated in the transcriptional regulation of Hox genes and cytokines, may play a role in tumor progression
# "Seems to act as a transcriptional corepressor for some genes such as MT1F and to favor the proliferation of cancer cells"


# Cabozantinib ----
final_data[cpd_name == "Cabozantinib"]
# Assigned for Thyroid Cancer, Good for Leukemia (AML)
# Cabozantinib is selectively cytotoxic in acute myeloid leukemia cells with FLT3-internal tandem duplication (FLT3-ITD) (2016)

cnv_rppa_integ <- get_all_interpret(data_types = "gnndrug_cnv_rppa", split_type = "CELL_LINE")
cabozantinib_molm13_attrs <- get_top_attrs(cnv_rppa_integ, compound_name = "Cabozantinib", cell_line_name = "MOLM13")

# Top positive attributed variables:
head(cabozantinib_molm13_attrs)

# n_4E-BP1_pT37_T46 antibody -> EIF4EBP1
# Paper: 4E-BP1, a multifactor regulated multifunctional protein (2016)
# ...It is likely that ERK acts directly on 4E-BP1 and indirectly via TSC2/mTOR following ionizing radiation (IR)
# and stimulates protein synthesis via ATM-dependent ERK phosphorylation...
# Paper: Eukaryotic initiation factor 4E-binding protein 1 (4E-BP1): a master regulator of mRNA translation involved in tumorigenesis (2016)
# Paper: Bcr-Abl Kinase Modulates the Translation Regulators Ribosomal Protein S6 and 4E-BP1 in Chronic Myelogenous Leukemia Cells via the Mammalian Target
# of Rapamycin

# BCL-2, eIF4E, FoxM1

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "4E-BP1_pT37_T46")
# 99th percentile, is it an oncogene?


# Top negative attributed variables
tail(cabozantinib_molm13_attrs)
# B-Raf
get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")
# 2nd percentile, underexpressed...
# Perhaps this underexpression is strange to the model...
get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "eEF2K")
# 6th percentile, underexpressed...
# Perhaps this underexpression is strange to the model...
get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "PCNA_Caution")
# 99th percentile, Overexpressed...


# Crizotinib ----
final_data[cpd_name == "Crizotinib"]
# Assigned for Lung Cancer, Good for Leukemia (CML)
# Crizotinib acts as ABL1 inhibitor combining ATP-binding with allosteric inhibition and is active against
# native BCR-ABL1 and its resistance and compound mutants BCR-ABL1T315I and BCR-ABL1T315I-E255K (2021)

cnv_exp_integ <- get_all_interpret(data_types = "gnndrug_cnv_exp", split_type = "CELL_LINE")
crizotinib_jk1_attrs <- get_top_attrs(cnv_exp_integ, compound_name = "Crizotinib", cell_line_name = "JK1")


# Top positive attributed variables:
head(crizotinib_jk1_attrs)

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "JK1", variable_name = "ACSBG1")
# ~ 100th percentile, overexpressed
get_expression_in_cell_line(data_type = "CNV", cell_line_name = "JK1", variable_name = "ACSBG1")
# ~ 37th percentile
# Weak link Global Identification of EVI1 Target Genes in Acute Myeloid Leukemia
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3694976/

# Top negative attributed variables
tail(crizotinib_jk1_attrs)

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "JK1", variable_name = "HBM")
# 100th percentile, overexpressed, the model doesn't expect this for CML?
# https://genevisible.com/cancers/HS/Gene%20Symbol/HBM
get_expression_in_cell_line(data_type = "CNV", cell_line_name = "JK1", variable_name = "HBM")
# ~ 47th percentile

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "JK1", variable_name = "OR51V1")
# 100th percentile
# Olfactory receptor,  
# Unique Polymorphisms at BCL11A, HBS1L-MYB and HBB Loci Associated with HbF in Kuwaiti Patients with Sickle Cell Disease (2021)

# Dabrafenib ----
final_data[cpd_name == "Dabrafenib"]
ctrp[cpd_name %like% "vemurafenib"]  # not in the dataset
gdsc2[cpd_name %like% "vemurafenib"]  # not in the dataset

# Assigned for Lung Cancer, Good for breast cancer (BRAF V600E) and Skin Cancer (melanoma)
# Dabrafenib binds and inhibits mutated BRAF
# It had been already approved for melanoma since 2013, but use in breast cancer is novel

# The model predicts the effect on DU4475 well, but doesn't consider BRAF in its decisions...

# DU4475 cell line page: https://web.expasy.org/cellosaurus/CVCL_1183
# Five-Year Outcomes with Dabrafenib plus Trametinib in Metastatic Melanoma (2019)

mut_exp_integ <- get_all_interpret(data_types = "gnndrug_mut_exp", split_type = "CELL_LINE")
dabrafenib_du4475_attrs <- get_top_attrs(mut_exp_integ, compound_name = "Dabrafenib", cell_line_name = "DU4475")

# Top positive attributed variables:
head(dabrafenib_du4475_attrs)
# OR4F21
# Paper: Olfactory Receptors as Biomarkers in Human Breast Carcinoma Tissues (2018)
# ...Furthermore, we observed the expression of ORs in brain tissue and brain tumor tissue (Figure ​(Figure5C).5C).
# It was striking that there were only three ORs (OR4F21, OR1F1, and OR2B6) expressed in cancerous tissues and cell lines,
# whereas several ORs were expressed in three different healthy brain tissues...
# Paper: Genome-wide copy number analysis of circulating tumor cells in breast cancer patients with liver metastasis (2020)
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "DU4475", variable_name = "OR4F21")
# ~ 99th percentile, overexpressed

# DNAJC5G
# Paper: Identification of novel methylation markers in HPV-associated oropharyngeal cancer: genome-wide discovery,
# tissue verification and validation testing in ctDNA (2020)
# ...Further study on ctDNA using Q-MSP in HPV-associated OPC showed that three genes (CALML5, DNAJC5G, and LY6D) had a
# high predictive ability as emerging biomarkers for a validation set, each capable of discriminating between the plasma
# of the patients from healthy individuals...
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "DU4475", variable_name = "DNAJC5G")
# ~100th percentile, overexpressed

get_expression_in_cell_line(data_type = "MUT", cell_line_name = "DU4475", variable_name = "BRAF")
dabrafenib_du4475_attrs[variable %like% "BRAF"]  # Very low value: -2.79308e-05
 
# Top negative attributed variables
tail(dabrafenib_du4475_attrs)

# C16orf82 (TNT protein)
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "DU4475", variable_name = "C16orf82")
# 100th percentile, highly expressed, does it act as a tumor suppressor?

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "DU4475", variable_name = "MRGPRG")
# 100th percentile, highly expressed, does it act as a tumor suppressor? 
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "DU4475", variable_name = "ADAD2")
# 100th percentile, highly expressed, does it act as a tumor suppressor? 
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "DU4475", variable_name = "CA6")
# 100th percentile, highly expressed, does it act as a tumor suppressor? 

# Ibrutinib ----
final_data[cpd_name == "Ibrutinib"]
# Assigned for Leukemia, Good for Breast Cancer, Lung Cancer (NSCLC), Colon/Colorectal Cancer, Gastric Cancer,
# and Lymphoma


## TE617T (Sarcoma, soft tissue, rhabdomyosarcoma) ====
# Paper: Ibrutinib inhibition of ERBB4 reduces cell growth in a WNT5A-dependent manner (2018), mentions Ewing’s sarcoma and
# triple negative breast cancer


cnv_exp_integ <- get_all_interpret(data_types = "gnndrug_cnv_exp", split_type = "CELL_LINE")
ibrutinib_te617t_attrs <- get_top_attrs(cnv_exp_integ, compound_name = "Ibrutinib", cell_line_name = "TE617T")
ibrutinib_te617t_attrs[variable %like% "BTK"]  # very small value...

## EFM192A, AU565, SKBR3, ZR7530 and HCC1419 Breast Cancer cell lines ----
# Ibrutinib treatment inhibits breast cancer progression and metastasis by inducing conversion of myeloid-derived
# suppressor cells to dendritic cells (2020)

cnv_exp_integ <- get_all_interpret(data_types = "gnndrug_cnv_exp", split_type = "CELL_LINE")
ibrutinib_EFM192A_attrs <- get_top_attrs(cnv_exp_integ, compound_name = "Ibrutinib", cell_line_name = "EFM192A")

ibrutinib_EFM192A_attrs[variable %like% "BTK"]  # very small value...
ibrutinib_EFM192A_attrs[variable %like% "ITK"]  # very small value...
ibrutinib_EFM192A_attrs[variable %like% "EGFR"]  # very small value...
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "BTK")
# Expression of BTK is in the 46th percentile among all cell lines, and 47th in breast cancer
get_expression_in_cell_line(data_type = "CNV", cell_line_name = "EFM192A", variable_name = "BTK")
# CNV of BTK is in the 93rd percentile (among all cell lines)

# Top positive attributed variables:
head(ibrutinib_EFM192A_attrs, n = 10)
quantile(ibrutinib_EFM192A_attrs$value)


# MAGEA8, CRLF3, FAM91A1, DERL1, THAP3, PSMA1, TOB1-AS1, FAM90A26, CDC27, DNTTIP1
sum(ibrutinib_EFM192A_attrs[value >= 0]$value)
sum(ibrutinib_EFM192A_attrs[value < 0]$value)

# MAGEA8
# Paper: Prognostic roles of MAGE family members in breast cancer based on KM‑Plotter Data (2019)
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "MAGEA8")
# 98th percentile in all cell lines
# 97th percentile in breast cancer cell lines

# CRLF3
# Paper: Comprehensive molecular biomarker identification in breast cancer brain metastases (2017)
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "CRLF3")
# 2nd percentile in all cell lines
# 18th percentile in breast cancer cell lines
# Seems to be downregulated

# DERL1
# Paper: Derlin-1 functions as a growth promoter in breast cancer (2020)
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "DERL1")
# 100th percentile in all cell lines 

# FAM91A1
# Chromatin interactome mapping at 139 independent breast cancer risk signals (2020)
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "FAM91A1")
# 99.9th percentile in all cell lines
# 100th percentile in breast cancer

# Top negative attributed variables
tail(ibrutinib_EFM192A_attrs)

# ODF4
# Upregulation of RHOXF2 and ODF4 Expression in Breast Cancer Tissues (2015)
# ... The expression of both genes was correlated with HER2/neu overexpression...

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "ODF4")
# 100th percentile in all cell lines

get_expression_in_cell_line(data_type = "CNV", cell_line_name = "EFM192A", variable_name = "ODF4")
# 16th percentile

# ACTA1
# The remodelling of actin composition as a hallmark of cancer (2021)
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "ACTA1")
# 100th percentile

# LYZL2
# LYZL2 expression associates with survival in triple negative breast cancer (preprint)
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "LYZL2")
# 100th percentile

### EFM192A FlexTable ----
require(flextable)
require(magrittr)
require(scales)
require(officer)

ibrutinib_final_data <- final_data[cpd_name == "Ibrutinib" &
                                     primary_disease == "Breast Cancer"]
cnv_exp_sub <- ibrutinib_final_data[data_types == "CNV_EXP"]
ibrutinib_final_data <- ibrutinib_final_data[, head(.SD, 1), by = "cell_name"]

setcolorder(ibrutinib_final_data, c(
                          "cell_name", "lineage_subtype",
                          "data_types", "target", "pred", "RMSE"))

ibrutinib_final_data[, data_types := gsub("_", "+", data_types)]
ibrutinib_final_data$cpd_name <- NULL
ibrutinib_final_data$primary_disease <- NULL
ibrutinib_final_data$assigned_disease <- NULL
ibrutinib_final_data$lineage <- NULL
ibrutinib_final_data$highest_drug_match_disease_aac <- NULL
ibrutinib_final_data$split_method <- NULL
ibrutinib_final_data$lineage <- NULL
# ibrutinib_final_data$lineage_subtype <- NULL

# "Data Type(s)", "Split Method", "True AAC", "Prediction", "RMSE Loss"
colnames(ibrutinib_final_data) <- c(
                          "Cell Line", "Lineage Subtype",
                          "Data Type(s)", "True AAC", "Prediction", "MAE Loss")
ibrutinib_final_data <- unique(ibrutinib_final_data)

ibrutinib_final_data$`Lineage Subtype` <- tools::toTitleCase(gsub("_", " ", ibrutinib_final_data$`Lineage Subtype`))
ft <- flextable(ibrutinib_final_data)

final_ft <- ft %>%
  # merge_v(j = c("Cancer", "Prescribed Drug(s)", "Cell Line Primary Disease")) %>%
  merge_v() %>%
  border_inner(border = fp_border(color="gray", width = 1)) %>%
  border_outer(part="all", border = fp_border(color="gray", width = 2)) %>%
  align(align = "center", part = "all")

final_ft <- autofit(final_ft)

dir.create("Plots/Drug_Tables/")
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/Drug_Tables/Ibrutinib_Table.docx")


### EFM192A GSEA ----
require(clusterProfiler)
require(pathview)
organism = "org.Hs.eg.db"
# BiocManager::install(organism, character.only = TRUE)
library(organism, character.only = TRUE)
keytypes(get(organism))
org.Hs.eg.db

# Get top 5% of attributions from EXP in each of positive and negative attributions

get_top_and_bottom_attrs <- function(all_attrs, omic_grep) {
  # Get attributes from specified omic data type
  cur_attrs <- all_attrs[variable %like% omic_grep]
  # Separate positive and negative attributes
  cur_attr_pos <- cur_attrs[value >= 0]
  cur_attr_neg <- cur_attrs[value < 0]
  # Get top 5 and bottom 5 percentile attributes
  attr_top_5 <- cur_attr_pos[value > quantile(cur_attr_pos$value, 0.95)]
  attr_bot_5 <- cur_attr_neg[value < quantile(cur_attr_neg$value, 0.05)]
  
  return(list(attr_top_5 = attr_top_5,
              attr_bot_5 = attr_bot_5))
}

exp_top_bot_attrs <- get_top_and_bottom_attrs(all_attrs = ibrutinib_EFM192A_attrs, omic_grep = "exp_.+")
top_exp_variables <- gsub(pattern = "exp_", "", exp_top_bot_attrs[[1]]$variable)
top_exp_variables <- setNames(exp_top_bot_attrs[[1]]$value, top_exp_variables)

cnv_top_bot_attrs <- get_top_and_bottom_attrs(all_attrs = ibrutinib_EFM192A_attrs, omic_grep = "cnv_.+")
top_cnv_variables <- gsub(pattern = "cnv_", "", cnv_top_bot_attrs[[1]]$variable)
top_cnv_variables <- setNames(cnv_top_bot_attrs[[1]]$value, top_cnv_variables)

cnv_exp_top_variables <- c(top_exp_variables, top_cnv_variables)
cnv_exp_top_variables <- sort(cnv_exp_top_variables, decreasing = T)
top_gse_cnv_exp <- gseGO(
  geneList = cnv_exp_top_variables,
  ont = "ALL",
  keyType = "SYMBOL",
  # nPerm = 10000,
  minGSSize = 3,
  maxGSSize = 800,
  pvalueCutoff = 0.05,
  verbose = TRUE,
  OrgDb = get(organism),
  scoreType = "pos",
  pAdjustMethod = "BH"
)
p_top_cnv_exp <- ridgeplot(top_gse_cnv_exp) + labs(x = "enrichment distribution") +
  ggtitle("Top 5% EXP Attributions GSE",
          subtitle = "Cell-line EFM192A (Breast Adenocarcinoma) + Ibrutinib\nTarget: 0.66, Predicted: 0.51")

exp_attrs <- ibrutinib_EFM192A_attrs[variable %like% "exp_.+"]
all_exp_variables <- gsub(pattern = "exp_", "", exp_attrs$variable)
all_exp_variables <- setNames(exp_attrs$value, all_exp_variables)
# 
# all_gse_exp <- gseGO(
#   geneList = all_exp_variables,
#   ont = "ALL",
#   keyType = "SYMBOL",
#   nPerm = 10000,
#   minGSSize = 3,
#   maxGSSize = 800,
#   pvalueCutoff = 0.05,
#   verbose = TRUE,
#   OrgDb = get(organism),
#   # scoreType = "pos",
#   pAdjustMethod = "BH", 
# )
# p_all_exp <- ridgeplot(all_gse_exp) + labs(x = "enrichment distribution") +
#   ggtitle("Top 5% EXP Attributions GSE",
#           subtitle = "Cell-line EFM192A (Breast Adenocarcinoma) + Ibrutinib\nTarget: 0.66, Predicted: 0.51")
all_gse_exp <- enrichDAVID(
  gene = all_exp_variables,
  idType = "SYMBOL",
  universe = 
  ont = "ALL",
  nPerm = 10000,
  minGSSize = 3,
  maxGSSize = 800,
  pvalueCutoff = 0.05,
  verbose = TRUE,
  # OrgDb = get(organism),
  # scoreType = "pos",
  pAdjustMethod = "BH", 
)
p_all_exp <- ridgeplot(all_gse_exp) + labs(x = "enrichment distribution") +
  ggtitle("Top 5% EXP Attributions GSE",
          subtitle = "Cell-line EFM192A (Breast Adenocarcinoma) + Ibrutinib\nTarget: 0.66, Predicted: 0.51")


### BTK in EFM192A AAC vs EXP/CNV ====
MUT <- fread("Data/DRP_Training_Data/DepMap_21Q2_Mutations_by_Cell.csv")
CNV <- fread("Data/DRP_Training_Data/DepMap_21Q2_CopyNumber.csv")
EXP <- fread("Data/DRP_Training_Data/DepMap_21Q2_Expression.csv")

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "BTK")
# Expression of BTK is in the 46th percentile (among all cell lines)
get_expression_in_cell_line(data_type = "CNV", cell_line_name = "EFM192A", variable_name = "BTK")
# Expression of BTK is in the 46th percentile (among all cell lines)
get_expression_in_cell_line(data_type = "MUT", cell_line_name = "EFM192A", variable_name = "BTK")

drug_name <- "Ibrutinib"
gene_name <- "BTK"
cell_line_name <- "EFM192A"
# Find expression of BTK in all cell lines that ibrutinib was tested in, plot it against AAC 
cur_cell_aac <- unique(ctrp[cpd_name == drug_name][, c("ccl_name", "area_above_curve")])
cur_exp_subset <- EXP[stripped_cell_line_name %in% unique(cur_cell_aac$ccl_name)][, c("stripped_cell_line_name",
                                                                                      gene_name), with = F]
exp_aac_subset <- merge(cur_exp_subset, cur_cell_aac, by.x = "stripped_cell_line_name", by.y = "ccl_name")
colnames(exp_aac_subset)[2] <- "value"

# Find copy number of BTK in all cell lines that ibrutinib was tested in, plot it against AAC 
cur_cnv_subset <- CNV[stripped_cell_line_name %in% unique(cur_cell_aac$ccl_name)][, c("stripped_cell_line_name",
                                                                                      gene_name), with = F]
cnv_aac_subset <- merge(cur_cnv_subset, cur_cell_aac, by.x = "stripped_cell_line_name", by.y = "ccl_name")
colnames(cnv_aac_subset)[2] <- "value"

require(ggplot2)
require(patchwork)
p_exp_aac <- ggplot(data = exp_aac_subset) +
  geom_point(aes(x = area_above_curve, y = value)) +
  xlab("Area Above Curve") +
  ylab("Gene Expression") +
  annotate(geom = "point",
           x = exp_aac_subset[stripped_cell_line_name == cell_line_name]$area_above_curve,
           y = exp_aac_subset[stripped_cell_line_name == cell_line_name]$value,
           colour = "orange", size = 1) + 
  annotate(
    geom = "curve",
    x = 0.6, y = 2,
    xend = exp_aac_subset[stripped_cell_line_name == cell_line_name]$area_above_curve - 0.005,
    yend = exp_aac_subset[stripped_cell_line_name == cell_line_name]$value + 0.2, 
    curvature = -.3, arrow = arrow(length = unit(2, "mm"))
  ) +
  annotate(geom = "text", x = 0.6, y = 2.3, label = "EFM192A", size = 6,
  ) +
  theme(text = element_text(size = 14, face = "bold"))

p_cnv_aac <- ggplot(data = cnv_aac_subset) +
  geom_point(aes(x = area_above_curve, y = value)) +
  xlab("Area Above Curve") +
  ylab("Copy Number") +
  annotate(geom = "point",
           x = cnv_aac_subset[stripped_cell_line_name == cell_line_name]$area_above_curve,
           y = cnv_aac_subset[stripped_cell_line_name == cell_line_name]$value,
           colour = "orange", size = 1) + 
  annotate(
    geom = "curve",
    x = 0.6, y = 1.35,
    xend = cnv_aac_subset[stripped_cell_line_name == cell_line_name]$area_above_curve - 0.005,
    yend = cnv_aac_subset[stripped_cell_line_name == cell_line_name]$value + 0.05, 
    curvature = -.3, arrow = arrow(length = unit(2, "mm"))
  ) +
  annotate(geom = "text", x = 0.6, y = 1.4, label = "EFM192A", size = 6, 
  ) +
  theme(text = element_text(size = 14, face = "bold"))



p_exp_aac + p_cnv_aac 


ggsave("Plots/Interpretation/BTK_EXP_CNV_vs_AAC.pdf")

# Top positive attributed variables:
head(ibrutinib_te617t_attrs)

# FGF10 
# Paper: FGF10/FGFR2 signal induces cell migration and invasion in pancreatic cancer (2008)
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "TE617T", variable_name = "FGF10")
# ~ 99th percentile
get_expression_in_cell_line(data_type = "CNV", cell_line_name = "TE617T", variable_name = "FGF10")
# 24th percentile

# UTS2B, potent vasoconstrictor...
# Paper: Protein expression of urotensin II, urotensin-related peptide and their receptor in the lungs of patients 
# with lymphangioleiomyomatosis (2010)
# ...Urotensin II (UII) and urotensin-related peptide (URP) are vasoactive neuropeptides with wide ranges of action in
# the normal mammalian lung, including the control of smooth muscle cell proliferation...

# Overexpressed in a bunch of cancers: https://www.proteinatlas.org/ENSG00000188958-UTS2B/pathology
# Paper: Integrated Genomic Analysis of Hu€rthle Cell Cancer Reveals Oncogenic Drivers, Recurrent Mitochondrial Mutations,
# and Unique Chromosomal Landscapes (2018) ...overexpressed in thyroid cancer...

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "TE617T", variable_name = "UTS2B")
# ~ 99th percentile

# BMP3
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "TE617T", variable_name = "BMP3")
# ~ 100th percentile
# Paper: Bone morphogenic protein 3 inactivation is an early and frequent event in colorectal cancer development (2008)
# Overexpressed in few cancer types: https://www.proteinatlas.org/ENSG00000152785-BMP3/pathology

# Top negative attributed variables
tail(ibrutinib_te617t_attrs)

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "TE617T", variable_name = "SYPL2")
# 100th percentile, overexpressed 


### MAGEA8, DERL1, FAM91A1  in EFM192A AAC vs EXP/CNV ====
# MUT <- fread("Data/DRP_Training_Data/DepMap_21Q2_Mutations_by_Cell.csv")
# CNV <- fread("Data/DRP_Training_Data/DepMap_21Q2_CopyNumber.csv")
EXP <- fread("Data/DRP_Training_Data/DepMap_21Q2_Expression.csv")

cnv_exp_integ <- get_all_interpret(data_types = "gnndrug_cnv_exp", split_type = "CELL_LINE")
ibrutinib_EFM192A_attrs <- get_top_attrs(cnv_exp_integ, compound_name = "Ibrutinib", cell_line_name = "EFM192A")
head(ibrutinib_EFM192A_attrs, n = 10)
# MAGEA8, CRLF3, FAM91A1, DERL1, THAP3, PSMA1, TOB1-AS1 (CNV), FAM90A26, CDC27, DNTTIP1

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "MAGEA8")
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "DERL1")
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "FAM91A1")

# Find expression of each gene in all cell lines that ibrutinib was tested in, plot it against AAC 
drug_name <- "Ibrutinib"
cell_line_name <- "EFM192A"
cur_cell_aac <- unique(ctrp[cpd_name == drug_name][, c("ccl_name", "area_above_curve", "primary_disease")])

# MAGEA8
magea8_exp_subset <- EXP[stripped_cell_line_name %in% unique(cur_cell_aac$ccl_name)][, c("stripped_cell_line_name",
                                                                                      "MAGEA8"), with = F]
magea8_exp_aac_subset <- merge(magea8_exp_subset, cur_cell_aac, by.x = "stripped_cell_line_name", by.y = "ccl_name")
colnames(magea8_exp_aac_subset)[2] <- "value"

# CRLF3
crlf3_exp_subset <- EXP[stripped_cell_line_name %in% unique(cur_cell_aac$ccl_name)][, c("stripped_cell_line_name",
                                                                                         "CRLF3"), with = F]
crlf3_exp_aac_subset <- merge(crlf3_exp_subset, cur_cell_aac, by.x = "stripped_cell_line_name", by.y = "ccl_name")
colnames(crlf3_exp_aac_subset)[2] <- "value"

#"DERL1"
derl1_exp_subset <- EXP[stripped_cell_line_name %in% unique(cur_cell_aac$ccl_name)][, c("stripped_cell_line_name",
                                                                                      "DERL1"), with = F]
derl1_exp_aac_subset <- merge(derl1_exp_subset, cur_cell_aac, by.x = "stripped_cell_line_name", by.y = "ccl_name")
colnames(derl1_exp_aac_subset)[2] <- "value"

#"FAM91A1"
fam91a1_exp_subset <- EXP[stripped_cell_line_name %in% unique(cur_cell_aac$ccl_name)][, c("stripped_cell_line_name",
                                                                                      "FAM91A1"), with = F]
fam91a1_exp_aac_subset <- merge(fam91a1_exp_subset, cur_cell_aac, by.x = "stripped_cell_line_name", by.y = "ccl_name")
colnames(fam91a1_exp_aac_subset)[2] <- "value"

# # Find copy number of BTK in all cell lines that ibrutinib was tested in, plot it against AAC 
# cur_cnv_subset <- CNV[stripped_cell_line_name %in% unique(cur_cell_aac$ccl_name)][, c("stripped_cell_line_name",
#                                                                                       gene_name), with = F]
# cnv_aac_subset <- merge(cur_cnv_subset, cur_cell_aac, by.x = "stripped_cell_line_name", by.y = "ccl_name")
# colnames(cnv_aac_subset)[2] <- "value"


plot_relative_expression <- function(data, xlab, ylab, annotate_label, cell_line_name,
                                     curvature=-0.3,
                                     curve_x_plus = 0.1, curve_y_plus = 0.1,
                                     curve_xend_plus = 0.01, curve_yend_plus = 0,
                                     text_x_plus = 0.1, text_y_plus = 0.1,
                                     xlim_min = 0, xlim_max = 1) {
  p_exp_aac <- ggplot(data = data, aes(x= area_above_curve, y = value)) +
    geom_point(aes(x = area_above_curve, y = value, color = primary_disease)) +
    xlab(xlab) +
    ylab(ylab) +
    annotate(geom = "point",
             x = data[stripped_cell_line_name == cell_line_name]$area_above_curve,
             y = data[stripped_cell_line_name == cell_line_name]$value,
             colour = "orange", size = 1) + 
    annotate(
      geom = "curve",
      x = data[stripped_cell_line_name == cell_line_name]$area_above_curve + curve_x_plus,
      y = data[stripped_cell_line_name == cell_line_name]$value + curve_y_plus,
      xend = data[stripped_cell_line_name == cell_line_name]$area_above_curve + curve_xend_plus,
      yend = data[stripped_cell_line_name == cell_line_name]$value + curve_yend_plus, 
      curvature = curvature, arrow = arrow(length = unit(2, "mm"))
    ) +
    annotate(geom = "text",
             x = data[stripped_cell_line_name == cell_line_name]$area_above_curve + text_x_plus,
             y = data[stripped_cell_line_name == cell_line_name]$value + text_y_plus,
             label = cell_line_name, size = 4,
    ) +
    theme(text = element_text(size = 14, face = "bold"), legend.position = "top") +
    scale_color_discrete(name = "Primary Disease") +
    scale_x_continuous(breaks = seq(xlim_min, xlim_max, by = 0.1)) +
    ylim(0, 10) +
    # xlim(xlim_min, xlim_max) +
    geom_smooth(aes(color = primary_disease),
                data = subset(data, primary_disease == "Breast Cancer"),
                method='lm')
  
  return(p_exp_aac)
}

require(ggplot2)
require(patchwork)
magea8_exp_aac_subset[!(primary_disease %in% c("Breast Cancer", "Leukemia", "Lymphoma")), primary_disease := "Other"]
magea8_exp_aac_subset <- magea8_exp_aac_subset[primary_disease != "Other"]
p_magea8_exp_aac <- plot_relative_expression(data = magea8_exp_aac_subset,
                         xlab = "Area Above Curve", ylab = "MAGEA8 Expression",
                         cell_line_name = "EFM192A")


# p_magea8_exp_aac <- ggplot(data = magea8_exp_aac_subset, aes(x= area_above_curve, y = value)) +
#   geom_point(aes(x = area_above_curve, y = value, color = primary_disease)) +
#   xlab("Area Above Curve") +
#   ylab("MAGEA8 Expression") +
#   annotate(geom = "point",
#            x = magea8_exp_aac_subset[stripped_cell_line_name == cell_line_name]$area_above_curve,
#            y = magea8_exp_aac_subset[stripped_cell_line_name == cell_line_name]$value,
#            colour = "orange", size = 1) + 
#   annotate(
#     geom = "curve",
#     x = 0.6, y = 4.5,
#     xend = magea8_exp_aac_subset[stripped_cell_line_name == cell_line_name]$area_above_curve - 0.01,
#     yend = magea8_exp_aac_subset[stripped_cell_line_name == cell_line_name]$value, 
#     curvature = -.3, arrow = arrow(length = unit(2, "mm"))
#   ) +
#   annotate(geom = "text", x = 0.6, y = 4.4, label = "EFM192A", size = 4,
#   ) +
#   theme(text = element_text(size = 14, face = "bold"), legend.position = "top") +
#   scale_color_discrete(name = "Primary Disease") +
#   ylim(0, 10) +
#   xlim(0, 1) +
#   geom_smooth(aes(color = primary_disease),
#               data = subset(magea8_exp_aac_subset, primary_disease == "Breast Cancer"),
#               method='lm')
  


derl1_exp_aac_subset[!(primary_disease %in% c("Breast Cancer", "Leukemia", "Lymphoma")), primary_disease := "Other"]
derl1_exp_aac_subset <- derl1_exp_aac_subset[primary_disease != "Other"]

p_derl1_exp_aac <- plot_relative_expression(data = derl1_exp_aac_subset,
                                             xlab = "Area Above Curve", ylab = "DERL1 Expression",
                                             cell_line_name = "EFM192A",
                                            curvature = -0.3,
                                            curve_x_plus = -0.02, curve_y_plus = 0.5,
                                            curve_xend_plus = -0.001, curve_yend_plus = 0.1,
                                            text_x_plus = -0.075, text_y_plus = 0.5,
                                            xlim_max = 0.75)

library(dplyr)
fitted_models <- derl1_exp_aac_subset %>% group_by(primary_disease) %>% do(model = lm(value ~ area_above_curve, data = .))
fitted_models$model
summary(fitted_models$model[[1]])  # 0.3505 Adjusted R-squared, breast cancer
summary(fitted_models$model[[2]])  # 0.0124 leukemia
summary(fitted_models$model[[3]])  # -0.0267 lymphoma


fam91a1_exp_aac_subset[!(primary_disease %in% c("Breast Cancer", "Leukemia", "Lymphoma")), primary_disease := "Other"]
fam91a1_exp_aac_subset <- fam91a1_exp_aac_subset[primary_disease != "Other"]
p_fam91a1_exp_aac <- plot_relative_expression(data = fam91a1_exp_aac_subset,
                                            xlab = "Area Above Curve", ylab = "FAM91A1 Expression",
                                            cell_line_name = "EFM192A",
                                            curvature = 0.3,
                                            curve_x_plus = -0.05, curve_y_plus = 0.2,
                                            curve_xend_plus = -0.005, curve_yend_plus = 0.01,
                                            text_x_plus = -0.05, text_y_plus = 0.5,
                                            xlim_max = 0.75)

p_derl1_exp_aac + p_fam91a1_exp_aac +
  plot_layout(guides = "collect") & theme(legend.position = "top")
ggsave("Plots/Interpretation/DERL1_FAM91A1_EXP_vs_AAC.pdf")

library(dplyr)
fitted_models <- fam91a1_exp_aac_subset %>% group_by(primary_disease) %>% do(model = lm(value ~ area_above_curve, data = .))
fitted_models$model
summary(fitted_models$model[[1]])  # 0.1336 Adjusted R-squared
summary(fitted_models$model[[2]])  # -0.02229
summary(fitted_models$model[[3]])  # 0.005535


p_magea8_exp_aac + p_derl1_exp_aac + p_fam91a1_exp_aac +
  plot_layout(guides = "collect") & theme(legend.position = "top")


ggsave("Plots/Interpretation/MAGEA8_DERL1_FAM91A1_EXP_vs_AAC.pdf",
       width = 16, height = 7)



top_genes <- c("MAGEA8", "CRLF3", "FAM91A1", "DERL1", "THAP3", "PSMA1", "FAM90A26", "CDC27", "DNTTIP1")
top_cnv <- "TOB1-AS1"
all_plots <- vector(mode = "list", length = length(top_genes))
# MAGEA8, CRLF3, FAM91A1, DERL1, THAP3, PSMA1, TOB1-AS1 (CNV), FAM90A26, CDC27, DNTTIP1
dir.create("Plots/Interpretation/EFM192A/")
for (i in 1:length(top_genes)) {
  cur_gene <- top_genes[i]
  exp_subset <- EXP[stripped_cell_line_name %in% unique(cur_cell_aac$ccl_name)][, c("stripped_cell_line_name",
                                                                                            cur_gene), with = F]
  cur_exp_aac_subset <- merge(exp_subset, cur_cell_aac, by.x = "stripped_cell_line_name", by.y = "ccl_name")
  colnames(cur_exp_aac_subset)[2] <- "value"
  
  cur_exp_aac_subset <- cur_exp_aac_subset[primary_disease %in% c("Breast Cancer", "Leukemia", "Lymphoma")]
  # cur_exp_aac_subset <- cur_exp_aac_subset[primary_disease != "Other"]
  p_exp_aac <- plot_relative_expression(data = cur_exp_aac_subset,
                                                xlab = "Area Above Curve", ylab = paste(cur_gene, "Expression"),
                                                cell_line_name = "EFM192A")
  all_plots[[i]] <- p_exp_aac
  ggsave(filename = paste0("Plots/Interpretation/EFM192A/", cur_gene, "_EXP_vs_AAC.pdf"),
         plot = p_exp_aac)
}


all_plots[[5]]

# SVM and Biomarker Proof ====
# Use top genes as features for an SVM, classifying responsive (AAC >= 0.5) and non-responsive
# (AAC < 0.5) cell lines to ibrutinib
require(data.table)
EXP <- fread("Data/DRP_Training_Data/DepMap_21Q2_Expression.csv")
ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")

cnv_exp_integ <- get_all_interpret(data_types = "gnndrug_cnv_exp", split_type = "CELL_LINE")
ibrutinib_EFM192A_attrs <- get_top_attrs(cnv_exp_integ, compound_name = "Ibrutinib", cell_line_name = "EFM192A")

# Take top 100 exp features
head(ibrutinib_EFM192A_attrs, n = 100)
top_100 <- head(ibrutinib_EFM192A_attrs[variable %like% "exp_"], n = 100)$variable
top_100 <- gsub("exp_", "", top_100)
top_genes <- c("MAGEA8", "CRLF3", "FAM91A1", "DERL1", "THAP3", "PSMA1", "FAM90A26", "CDC27", "DNTTIP1")


for (gene in top_genes) {
  print(get_expression_in_cell_line(data_type = "EXP", cell_line_name = "EFM192A", variable_name = "MAGEA8"))
}


# Subset gene expression data to only the 9 genes
exp_sub <- EXP[, c("stripped_cell_line_name", top_genes), with = F]
# Add AAC information
ctrp_sub <- ctrp[cpd_name == "Ibrutinib", c("ccl_name", "primary_disease", "area_above_curve")]
cur_data <- merge(exp_sub, ctrp_sub, by.x = "stripped_cell_line_name", by.y = "ccl_name")
cur_data[, responsive := ifelse(area_above_curve >= 0.5, 1, 0)]


# require(devtools)
# install_version("RSofia", version = "1.1", repos = "http://cran.us.r-project.org")
# require("RSofia")
require(caTools)
require(e1071)

set.seed(42) 
sum(cur_data$responsive)
cur_data$responsive <- as.factor(cur_data$responsive)

sample = sample.split(cur_data$responsive, SplitRatio = .5)  # stratifies data too
train <- cur_data[(sample)]
test <- cur_data[(!sample)]
# train = subset(cur_data, sample == TRUE)
# test  = subset(cur_data, sample == FALSE)

colnames(train[, c(top_genes, "responsive"), with = F])
classifier = svm(formula = responsive ~ .,
                 data = train[, c(top_genes, "responsive"), with = F],
                 type = 'C-classification',
                 kernel = 'linear',
                 scale = F
                 )

summary(classifier)
print(classifier)

pred <- predict(classifier, test[, 2:101])
table(pred, test$responsive)  # terrible...


# Linear regression model
cnv_exp_integ <- get_all_interpret(data_types = "gnndrug_cnv_exp", split_type = "CELL_LINE")
ibrutinib_EFM192A_attrs <- get_top_attrs(cnv_exp_integ, compound_name = "Ibrutinib", cell_line_name = "EFM192A")

# Take top 100 exp features
head(ibrutinib_EFM192A_attrs, n = 100)
top_100 <- head(ibrutinib_EFM192A_attrs[variable %like% "exp_"], n = 100)$variable
top_100 <- gsub("exp_", "", top_100)
top_genes <- c("MAGEA8", "CRLF3", "FAM91A1", "DERL1", "THAP3", "PSMA1", "FAM90A26", "CDC27", "DNTTIP1")

regressor <- lm(area_above_curve ~ ., data = train[, c(top_genes, "area_above_curve"), with = F])
summary(regressor)

regressor <- lm(area_above_curve ~ ., data = train[, c(top_100, "area_above_curve"), with = F])
summary(regressor)


exp_mirna_integ <- get_all_interpret(data_types = "gnndrug_exp_mirna", split_type = "CELL_LINE")
ibrutinib_EFM192A_attrs_exp_mirna <- get_top_attrs(exp_mirna_integ, compound_name = "Ibrutinib", cell_line_name = "EFM192A")

# Take top 100 mirna features
head(ibrutinib_EFM192A_attrs_exp_mirna, n = 100)
top_100 <- head(ibrutinib_EFM192A_attrs_exp_mirna[variable %like% "-miR-"], n = 100)$variable
top_100 <- gsub("n_", "", top_100)
# top_genes <- c("MAGEA8", "CRLF3", "FAM91A1", "DERL1", "THAP3", "PSMA1", "FAM90A26", "CDC27", "DNTTIP1")
# Subset gene expression data to only the 9 genes
MIRNA <- fread(path_dict[["MIRNA"]])
mirna_sub <- MIRNA[, c("stripped_cell_line_name", top_100), with = F]
# Add AAC information
ctrp_sub <- ctrp[cpd_name == "Ibrutinib", c("ccl_name", "primary_disease", "area_above_curve")]
cur_data <- merge(mirna_sub, ctrp_sub, by.x = "stripped_cell_line_name", by.y = "ccl_name")
cur_data[, responsive := ifelse(area_above_curve >= 0.5, 1, 0)]

sample = sample.split(cur_data$responsive, SplitRatio = .5)  # stratifies data too
train <- cur_data[(sample)]
test <- cur_data[(!sample)]

regressor <- lm(area_above_curve ~ .,
                data = cur_data[, c(top_100, "area_above_curve"), with = F], )
summary(regressor)  # Adjusted R-squared:  0.0747

# Take top 100 mirna features
metab_rppa_integ <- get_all_interpret(data_types = "gnndrug_metab_rppa", split_type = "CELL_LINE")
ibrutinib_EFM192A_attrs_metab_rppa <- get_top_attrs(metab_rppa_integ, compound_name = "Ibrutinib", cell_line_name = "EFM192A")

head(ibrutinib_EFM192A_attrs_exp_mirna, n = 100)
top_100 <- head(ibrutinib_EFM192A_attrs_exp_mirna[variable %like% "-miR-"], n = 100)$variable
top_100 <- gsub("n_", "", top_100)
# top_genes <- c("MAGEA8", "CRLF3", "FAM91A1", "DERL1", "THAP3", "PSMA1", "FAM90A26", "CDC27", "DNTTIP1")
# Subset gene expression data to only the 9 genes
MIRNA <- fread(path_dict[["MIRNA"]])
mirna_sub <- MIRNA[, c("stripped_cell_line_name", top_100), with = F]
# Add AAC information
ctrp_sub <- ctrp[cpd_name == "Ibrutinib", c("ccl_name", "primary_disease", "area_above_curve")]
cur_data <- merge(mirna_sub, ctrp_sub, by.x = "stripped_cell_line_name", by.y = "ccl_name")
cur_data[, responsive := ifelse(area_above_curve >= 0.5, 1, 0)]

sample = sample.split(cur_data$responsive, SplitRatio = .5)  # stratifies data too
train <- cur_data[(sample)]
test <- cur_data[(!sample)]

regressor <- lm(area_above_curve ~ .,
                data = cur_data[, c(top_100, "area_above_curve"), with = F], )
summary(regressor)

# https://cran.r-project.org/src/contrib/Archive/RSofia/
# Top positive attributed variables:
head(ibrutinib_te617t_attrs)

# FGF10 
# Paper: FGF10/FGFR2 signal induces cell migration and invasion in pancreatic cancer (2008)
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "TE617T", variable_name = "FGF10")
# ~ 99th percentile
get_expression_in_cell_line(data_type = "CNV", cell_line_name = "TE617T", variable_name = "FGF10")
# 24th percentile

# UTS2B, potent vasoconstrictor...
# Paper: Protein expression of urotensin II, urotensin-related peptide and their receptor in the lungs of patients 
# with lymphangioleiomyomatosis (2010)
# ...Urotensin II (UII) and urotensin-related peptide (URP) are vasoactive neuropeptides with wide ranges of action in
# the normal mammalian lung, including the control of smooth muscle cell proliferation...

# Overexpressed in a bunch of cancers: https://www.proteinatlas.org/ENSG00000188958-UTS2B/pathology
# Paper: Integrated Genomic Analysis of Hu€rthle Cell Cancer Reveals Oncogenic Drivers, Recurrent Mitochondrial Mutations,
# and Unique Chromosomal Landscapes (2018) ...overexpressed in thyroid cancer...

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "TE617T", variable_name = "UTS2B")
# ~ 99th percentile

# BMP3
get_expression_in_cell_line(data_type = "EXP", cell_line_name = "TE617T", variable_name = "BMP3")
# ~ 100th percentile
# Paper: Bone morphogenic protein 3 inactivation is an early and frequent event in colorectal cancer development (2008)
# Overexpressed in few cancer types: https://www.proteinatlas.org/ENSG00000152785-BMP3/pathology

# Top negative attributed variables
tail(ibrutinib_te617t_attrs)

get_expression_in_cell_line(data_type = "EXP", cell_line_name = "TE617T", variable_name = "SYPL2")
# 100th percentile, overexpressed 



# Lapatinib ----
final_data[cpd_name == "Lapatinib"]
# Assigned for Breast Cancer, Good for Gastric Cancer (gastric adenocarcinoma)
# Reached Phase III clinical trials in combination with chemotherapy
# CNV_RPPA had lowest RMSE

mut_cnv_integ <- get_all_interpret(data_types = "gnndrug_mut_cnv", split_type = "CELL_LINE")
bosutinib_pc14_attrs <- get_top_attrs(mut_cnv_integ, compound_name = "Bosutinib", cell_line_name = "PC14")


# Top positive attributed variables:
head(bosutinib_pc14_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

# Top negative attributed variables
tail(cabozantinib_molm13_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

# ==== Sorafenib ====
final_data[cpd_name == "Sorafenib"]
# Assigned for Kidney Cancer and Thyroid Cancer, Good for Leukemia (AML)
# Sorafenib or placebo in patients with newly diagnosed acute myeloid leukaemia: long-term follow-up of the
# randomized controlled SORAML trial
# PROT_* data types are good for DRP

mut_cnv_integ <- get_all_interpret(data_types = "gnndrug_mut_cnv", split_type = "CELL_LINE")
bosutinib_pc14_attrs <- get_top_attrs(mut_cnv_integ, compound_name = "Bosutinib", cell_line_name = "PC14")


# Top positive attributed variables:
head(bosutinib_pc14_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

# Top negative attributed variables
tail(cabozantinib_molm13_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

# ==== Sunitinib ====
final_data[cpd_name == "Sunitinib"]
# Assigned for Pancreatic Cancer, Good for Leukemia (AML) and Malignant Rhabdoid Tumor (kidneys)
# A phase I/II study of sunitinib and intensive chemotherapy in patients over 60 years of age with
# acute myeloid leukaemia and activating FLT3 mutations (2015)
# PROT_* data types are good for DRP

mut_cnv_integ <- get_all_interpret(data_types = "gnndrug_mut_cnv", split_type = "CELL_LINE")
bosutinib_pc14_attrs <- get_top_attrs(mut_cnv_integ, compound_name = "Bosutinib", cell_line_name = "PC14")


# Top positive attributed variables:
head(bosutinib_pc14_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

# Top negative attributed variables
tail(cabozantinib_molm13_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

# ==== Temsirolimus ====
final_data[cpd_name == "Temsirolimus"]
print(final_data[cpd_name == "Temsirolimus"], nrows = 500)
# Assigned for Kidney Cancer, Good for Multiple Myeloma, Lymphoma (Non-Hodgkin), Leukemia (CLL),
# Breast Cancer (Breast Ductal), Ovarian Cancer (Adenocarcinoma),
# Lung Cancer (Mesothelioma), Endometrial/Uterine Cancer
# mTOR inhibitor, might explain efficacy in multiple cancers

mut_cnv_integ <- get_all_interpret(data_types = "gnndrug_mut_cnv", split_type = "CELL_LINE")
bosutinib_pc14_attrs <- get_top_attrs(mut_cnv_integ, compound_name = "Bosutinib", cell_line_name = "PC14")


# Top positive attributed variables:
head(bosutinib_pc14_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

# Top negative attributed variables
tail(cabozantinib_molm13_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

# ==== Trametinib ====
final_data[cpd_name == "Trametinib"]
# Assigned for Skin Cancer, Good for Leukemia (AML)
# Used in patients with lymphatic edema that have a specific ARAF gene variation
# Trametinib inhibits RAS-mutant MLL-rearranged acute lymphoblastic leukemia at specific niche sites
# and reduces ERK phosphorylation in vivo (2018)

mut_cnv_integ <- get_all_interpret(data_types = "gnndrug_mut_cnv", split_type = "CELL_LINE")
bosutinib_pc14_attrs <- get_top_attrs(mut_cnv_integ, compound_name = "Bosutinib", cell_line_name = "PC14")


# Top positive attributed variables:
head(bosutinib_pc14_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

# Top negative attributed variables
tail(cabozantinib_molm13_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

# ==== Vandetanib ====
final_data[cpd_name == "Vandetanib"]
# Assigned for Thyroid Cancer, Good for Lung Cancer (NSCLC), Leukemia (AML), Head and Neck Cancer (Upper aerodigestiv squamous)
# AstraZeneca tried Vandetanib for NSCLC but didn't see improved effect alongside chemotherapy

mut_cnv_integ <- get_all_interpret(data_types = "gnndrug_mut_cnv", split_type = "CELL_LINE")
bosutinib_pc14_attrs <- get_top_attrs(mut_cnv_integ, compound_name = "Bosutinib", cell_line_name = "PC14")


# Top positive attributed variables:
head(bosutinib_pc14_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

# Top negative attributed variables
tail(cabozantinib_molm13_attrs)

get_expression_in_cell_line(data_type = "RPPA", cell_line_name = "MOLM13", variable_name = "B-Raf_Caution")

