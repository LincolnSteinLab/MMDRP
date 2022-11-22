# upset_plots.R

# install.packages("UpSetR")
# install.packages("ggupset")
require(UpSetR)
# require(ggupset)
require(data.table)
require(stringr)
require(ggplot2)
require(patchwork)

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


dim(rppa)
dim(metab)
dim(hist)
dim(mirna)
dim(prot)
dim(exp)
dim(cnv)
dim(mut)
uniqueN(gdsc2$ccl_name)
uniqueN(ctrp$ccl_name)
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
cnv_line_info$data_type <- "Copy Number Variation"
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
rppa_line_info$data_type <- "Reverse-Phase Protein Array"

ctrp_line_info <- ctrp_line_info[, c("stripped_cell_line_name", "primary_disease")]
ctrp_line_info$data_type <- "Dose-Response"

gdsc2_line_info <- gdsc2_line_info[, c("stripped_cell_line_name", "primary_disease")]
gdsc2_line_info$data_type <- "Dose-Response"

list_input <- list(
  Mutational = mut_line_info$stripped_cell_line_name,
  `Copy Number Variation` = cnv_line_info$stripped_cell_line_name,
  `Gene Expression` = exp_line_info$stripped_cell_line_name,
  `Protein Quantification` = prot_line_info$stripped_cell_line_name,
  
  `microRNA Expression` = mirna_line_info$stripped_cell_line_name,
  `Histone Modification` = hist_line_info$stripped_cell_line_name,
  `Metabolomics` = metab_line_info$stripped_cell_line_name,
  `Reverse-Phase Protein Array` = rppa_line_info$stripped_cell_line_name,
  
  `CTRPv2 Dose-Response` = unique(ctrp$ccl_name),
  `GDSC2 Dose-Response` = unique(gdsc2$ccl_name)
)

make_all_combinations <- function(set){
  unlist(lapply(seq_along(set), function(size){
    apply(combn(set, size), 2, paste0, collapse="-")
  }))
}



p <- upset(fromList(list_input),
      sets = c("CTRPv2 Dose-Response", "GDSC2 Dose-Response",
               "Mutational","Copy Number Variation", "Gene Expression", "Protein Quantification",
               "microRNA Expression", "Histone Modification", "Metabolomics", "Reverse-Phase Protein Array"
               ),
      keep.order = T,
      # sets = c("Dose-Response"),
      mainbar.y.label = "Data Intersection Size",
      sets.x.label = "Cell Lines per Data Type",
      # group.by = "sets",
      order.by = "freq",
      text.scale = c(1.3, 1.3, 1, 1, 1, 0.75))
p

pdf(file="Plots/Dataset_Exploration/UpSetR_Overlap_Plot_CTRPv2.pdf", width = 10, height = 8)
p
dev.off()

# ggsave(filename = "Plots/Dataset_Exploration/UpSetR_Overlap_Plot_CTRPv2.pdf")

# all_cells <- rbindlist(list(
#   mut_line_info,
#   cnv_line_info,
#   exp_line_info,
#   prot_line_info,
#   mirna_line_info,
#   hist_line_info,
#   metab_line_info,
#   rppa_line_info)
# )
# all_cells <- all_cells[, -2]
# 
# # install.packages("tidyverse")
# require(tidyverse)
# all_cells <- tidyr::as_tibble(all_cells[, 1:2])
# simple_groups_df <- all_cells %>%
#   group_by(stripped_cell_line_name) %>%
#   summarize(groups = list(data_type))
# 
# extended_groups_df <- simple_groups_df %>%
#   mutate(groups = lapply(groups, make_all_combinations)) %>%
#   unnest()
# 
# unique(extended_groups_df)
# ggplot(extended_groups_df, aes(x=groups)) +
#   geom_bar() +
#   axis_combmatrix(sep = "-", )
# 
# all_cells[, extended_groups := lapply(data_type, make_all_combinations)]
# lapply(all_cells, make_all_combinations)

# ==== Bimodal Intersections Counts ====
require(ggplot2)
require(data.table)
require(flextable)
require(magrittr)
require(scales)
require(officer)

ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")
mut_line_info$data_type <- "MUT"
cnv_line_info$data_type <- "CNV"
exp_line_info$data_type <- "EXP"
prot_line_info$data_type <- "PROT"

mirna_line_info$data_type <- "MIRNA"
hist_line_info$data_type <- "HIST"
metab_line_info$data_type <- "METAB"
rppa_line_info$data_type <- "RPPA"

ctrp_line_info$data_type <- "CTRP"

all_cells <- rbindlist(list(mut_line_info, cnv_line_info, exp_line_info, prot_line_info,
                            mirna_line_info, metab_line_info, hist_line_info, rppa_line_info))
all_cells <- unique(all_cells)

ctrp_cells <- unique(ctrp_line_info$stripped_cell_line_name)
all_omics <- data.table(
  `Data Type(s)` = c("Mutational","Copy Number", "Gene Expression", "Protein Quantification",
                     "microRNA Expression", "Metabolomics", "Histone Modification", "RPPA"),
  `Abbreviation` = c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA"),
                        `Number of Samples` = vector(mode = "integer", length = 8))

for (i in 1:nrow(all_omics)) {
  first_cells <- all_cells[data_type == all_omics[i, 2]]$stripped_cell_line_name
  # second_cells <- all_cells[data_type == all_omics[i, 2]]$stripped_cell_line_name
  # cell_overlap <- Reduce(intersect, list(first_cells, second_cells, ctrp_cells))
  ctrp_overlap <- uniqueN(ctrp[ccl_name %in% first_cells])
  all_omics[i, 3] <- ctrp_overlap
}

set_flextable_defaults(
  font.size = 10, theme_fun = theme_vanilla,
  padding = 6,
  background.color = "#EFEFEF")

colourer <- col_numeric(
  palette = c("red", "white"),
  domain = c(min(all_omics$`Number of Samples`), max(all_omics$`Number of Samples`)))


# ==== bimodal 
ft <- flextable(all_omics)
final_ft <- ft %>%
  merge_v(j = c("Data Type(s)", "Number of Samples")) %>%
  border_inner(border = fp_border(color="gray", width = 1)) %>%
  border_outer(part="all", border = fp_border(color="gray", width = 2)) %>%
  align(align = "center", j = c(2, 3), part = "header") %>%

  bg(
    bg = colourer,
    j = "Number of Samples", 
    part = "body")
  

final_ft <- autofit(final_ft)
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/Dataset_Exploration/bimodal_samples_per_data_type_combo.docx")

# ==== trimodal
all_tri_omic_combos_el <- utils::combn(c("MUT", 'CNV', 'EXP', 'PROT', 'MIRNA', 'METAB', 'HIST', 'RPPA'), 2, simplify = T)
all_tri_omic_combos_el <- t(all_tri_omic_combos_el)
all_tri_omic_combos_el <- as.data.table(all_tri_omic_combos_el)

# all_sample_counts <- vector(mode = "numeric", length = nrow(temp))
ctrp_cells <- unique(ctrp_line_info$stripped_cell_line_name)
all_tri_omic_combos_el$sample_counts <- vector(mode = "integer")
for (i in 1:nrow(all_tri_omic_combos_el)) {
  first_cells <- all_cells[data_type == all_tri_omic_combos_el[i, 1]]$stripped_cell_line_name
  second_cells <- all_cells[data_type == all_tri_omic_combos_el[i, 2]]$stripped_cell_line_name
  cell_overlap <- Reduce(intersect, list(first_cells, second_cells, ctrp_cells))
  ctrp_overlap <- uniqueN(ctrp[ccl_name %in% cell_overlap])
  all_tri_omic_combos_el[i, 3] <- ctrp_overlap
}
colnames(all_tri_omic_combos_el) <- c("Data Type 1", "Data Type 2", "Number of Samples")

colourer <- col_numeric(
  palette = c("red", "white"),
  domain = c(min(all_tri_omic_combos_el$`Number of Samples`), max(all_tri_omic_combos_el$`Number of Samples`)))

ft <- flextable(all_tri_omic_combos_el)
final_ft <- ft %>%
  merge_v(j = c("Data Type 1", "Data Type 2", "Number of Samples")) %>%
  border_inner(border = fp_border(color="gray", width = 1)) %>%
  border_outer(part="all", border = fp_border(color="gray", width = 2)) %>%
  align(align = "center", j = 1:3, part = "all") %>%
  bg(
    bg = colourer,
    j = "Number of Samples", 
    part = "body")


final_ft <- autofit(final_ft)
read_docx() %>% 
  body_add_flextable(value = final_ft) %>% 
  print(target = "Plots/Dataset_Exploration/trimodal_samples_per_data_type_combo.docx")
