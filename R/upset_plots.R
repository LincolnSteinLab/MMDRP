# upset_plots.R

# install.packages("UpSetR")
require(UpSetR)
require(data.table)
require(stringr)
require(ggplot2)

line_info <- fread("Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv")
ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")
# gdsc1 <- fread("Data/DRP_Training_Data/GDSC1_AAC_SMILES.txt")
gdsc2 <- fread("Data/DRP_Training_Data/GDSC2_AAC_SMILES.txt")
exp <- fread("Data/DRP_Training_Data/DepMap_21Q2_Expression.csv")
mut <- fread("Data/DRP_Training_Data/DepMap_21Q2_Mutations_by_Cell.csv")
cnv <- fread("Data/DRP_Training_Data/DepMap_21Q2_CopyNumber.csv")
prot <- fread("Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv")

mirna <- fread("Data/DRP_Training_Data/DepMap_2019_miRNA.csv")
hist <- fread("Data/DRP_Training_Data/DepMap_2019_ChromatinProfiling.csv")
metab <- fread("Data/DRP_Training_Data/DepMap_2019_Metabolomics.csv")
rppa <- fread("Data/DRP_Training_Data/DepMap_2019_RPPA.csv")


# ctrp$ccl_name = str_replace(toupper(ctrp$ccl_name), "-", "")
# 
# exp_ccl_names = exp$stripped_cell_line_name
# exp_ccl_names = str_replace(toupper(exp_ccl_names), "-", "")
# 
# mut_ccl_names = mut$stripped_cell_line_name
# mut_ccl_names = str_replace(toupper(mut_ccl_names), "-", "")
# 
# cnv_ccl_names = cnv$stripped_cell_line_name
# cnv_ccl_names = str_replace(toupper(cnv_ccl_names), "-", "")
mut$stripped_cell_line_name = str_replace(toupper(mut$stripped_cell_line_name), "-", "")
cnv$stripped_cell_line_name = str_replace(toupper(cnv$stripped_cell_line_name), "-", "")
exp$stripped_cell_line_name = str_replace(toupper(exp$stripped_cell_line_name), "-", "")
prot$stripped_cell_line_name = str_replace(toupper(prot$stripped_cell_line_name), "-", "")

mirna$stripped_cell_line_name = str_replace(toupper(mirna$stripped_cell_line_name), "-", "")
hist$stripped_cell_line_name = str_replace(toupper(hist$stripped_cell_line_name), "-", "")
metab$stripped_cell_line_name = str_replace(toupper(metab$stripped_cell_line_name), "-", "")
rppa$stripped_cell_line_name = str_replace(toupper(rppa$stripped_cell_line_name), "-", "")

ctrp$ccl_name = str_replace(toupper(ctrp$ccl_name), "-", "")
gdsc2$ccl_name = str_replace(toupper(gdsc2$ccl_name), "-", "")

mut_line_info <- line_info[stripped_cell_line_name %in% unique(mut$stripped_cell_line_name)]  
cnv_line_info <- line_info[stripped_cell_line_name %in% unique(cnv$stripped_cell_line_name)]  
exp_line_info <- line_info[stripped_cell_line_name %in% unique(exp$stripped_cell_line_name)]  
prot_line_info <- line_info[stripped_cell_line_name %in% unique(prot$stripped_cell_line_name)]

mirna_line_info <- line_info[stripped_cell_line_name %in% unique(mirna$stripped_cell_line_name)]  
hist_line_info <- line_info[stripped_cell_line_name %in% unique(hist$stripped_cell_line_name)]  
metab_line_info <- line_info[stripped_cell_line_name %in% unique(metab$stripped_cell_line_name)]  
rppa_line_info <- line_info[stripped_cell_line_name %in% unique(rppa$stripped_cell_line_name)]

ctrp_line_info <- line_info[stripped_cell_line_name %in% unique(ctrp$ccl_name)]
gdsc2_line_info <- line_info[stripped_cell_line_name %in% unique(gdsc2$ccl_name)]

mut_line_info <- mut_line_info[, c("stripped_cell_line_name", "primary_disease")]
mut_line_info$data_type <- "Mutational"
cnv_line_info <- cnv_line_info[, c("stripped_cell_line_name", "primary_disease")]
cnv_line_info$data_type <- "Copy Number"
exp_line_info <- exp_line_info[, c("stripped_cell_line_name", "primary_disease")]
exp_line_info$data_type <- "Gene Expression"
prot_line_info <- prot_line_info[, c("stripped_cell_line_name", "primary_disease")]
prot_line_info$data_type <- "Protein Quantification"

mirna_line_info <- mirna_line_info[, c("stripped_cell_line_name", "primary_disease")]
mirna_line_info$data_type <- "microRNA Expression"
hist_line_info <- hist_line_info[, c("stripped_cell_line_name", "primary_disease")]
hist_line_info$data_type <- "Histone Modification"
metab_line_info <- metab_line_info[, c("stripped_cell_line_name", "primary_disease")]
metab_line_info$data_type <- "Metabolomics"
rppa_line_info <- rppa_line_info[, c("stripped_cell_line_name", "primary_disease")]
rppa_line_info$data_type <- "RPPA"

ctrp_line_info <- ctrp_line_info[, c("stripped_cell_line_name", "primary_disease")]
ctrp_line_info$data_type <- "Dose-Response"

gdsc2_line_info <- gdsc2_line_info[, c("stripped_cell_line_name", "primary_disease")]
gdsc2_line_info$data_type <- "Dose-Response"

list_input <- list(
  Mutational = mut_line_info$stripped_cell_line_name,
  `Copy Number` = cnv_line_info$stripped_cell_line_name,
  `Gene Expression` = exp_line_info$stripped_cell_line_name,
  `Protein Quantification` = prot_line_info$stripped_cell_line_name,
  
  `microRNA Expression` = mirna_line_info$stripped_cell_line_name,
  `Histone Modification` = hist_line_info$stripped_cell_line_name,
  `Metabolomics` = prot_line_info$stripped_cell_line_name,
  `RPPA` = prot_line_info$stripped_cell_line_name,
  
  `CTRPv2 Dose-Response` = unique(ctrp$ccl_name),
  `GDSC2 Dose-Response` = unique(gdsc2$ccl_name)
)

p <- upset(fromList(list_input),
      sets = c("CTRPv2 Dose-Response", "GDSC2 Dose-Response",
               "Mutational","Copy Number", "Gene Expression", "Protein Quantification",
               "microRNA Expression", "Histone Modification", "Metabolomics", "RPPA"
               ),
      keep.order = T,
      # sets = c("Dose-Response"),
      mainbar.y.label = "Data Intersection Size",
      sets.x.label = "Cell Lines per Data Type",
      # group.by = "sets",
      order.by = "freq")
p

pdf(file="Plots/Dataset_Exploration/UpSetR_Overlap_Plot_CTRPv2.pdf", width = 10, height = 8)
p
dev.off()

# ggsave(filename = "Plots/Dataset_Exploration/UpSetR_Overlap_Plot_CTRPv2.pdf")

