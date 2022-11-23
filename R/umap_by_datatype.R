# umap_by_datatype.R

require(data.table)
require(ggplot2)
library(ggfortify)
require(umap)
dir.create("Plots/UMAPs")


depmap_info <- fread("Data/DepMap/21Q2/sample_info.csv")

# ==== Proteomics Data ====
depmap_prot <- fread("Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv")
depmap_prot[1:5, 1:5]
prot_umap <- umap(depmap_prot[,-c(1:3)])
prot_info <- merge(depmap_prot[, 1:3], depmap_info, by = "DepMap_ID", sort = F)

prot_umap_layout <- as.data.table(prot_umap$layout)
prot_umap_layout$primary_disease <- depmap_prot$primary_disease
depmap_prot[,1:3]
ggplot(prot_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = primary_disease)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Primary Disease") +
  ggtitle(label = "DepMap Proteomics by Primary Disease")
ggsave(filename = "Plots/UMAPs/DepMap_Proteomics_by_Primary_Disease.pdf")

prot_umap_layout$culture_type <- prot_info$culture_type
ggplot(prot_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = culture_type)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Suspension") +
  ggtitle(label = "DepMap Proteomics by Culture Type")
ggsave(filename = "Plots/UMAPs/DepMap_Proteomics_by_Culture_Type.pdf")

rm(list = c("depmap_prot", "prot_umap", "prot_info", "prot_umap_layout"))

# ==== Mutational Data ====
depmap_mut <- fread("Data/DRP_Training_Data/DepMap_21Q2_Mutations_by_Cell.csv")
depmap_mut[1:5, 1:5]
mut_umap <- umap(depmap_mut[,-c(1:3)])
mut_info <- merge(depmap_mut[, 1:3], depmap_info, by = "DepMap_ID", sort = F)

mut_umap_layout <- as.data.table(mut_umap$layout)
mut_umap_layout$primary_disease <- depmap_mut$primary_disease
depmap_mut[,1:3]
ggplot(mut_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = primary_disease)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Primary Disease") +
  ggtitle(label = "DepMap Binary Mutational Data by Primary Disease")
ggsave(filename = "Plots/UMAPs/DepMap_Mutational_by_Primary_Disease.pdf", height = 8, width = 16, units = "in")

mut_umap_layout$culture_type <- mut_info$culture_type
ggplot(mut_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = culture_type)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  geom_point(data=mut_umap_layout[culture_type == "Suspension"],
             aes(x=V1,y=V2), color="black", size=3) +
  labs(color = "Suspension") +
  ggtitle(label = "DepMap Binary Mutational Data by Culture Type")
ggsave(filename = "Plots/UMAPs/DepMap_Mutational_by_Culture_Type.pdf")
mut_umap_layout$lineage_sub_subtype <- mut_info$lineage_sub_subtype
ggplot(mut_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = lineage_sub_subtype)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Suspension") +
  ggtitle(label = "DepMap Binary Mutational Data by Culture Type")

rm(list = c("depmap_mut", "mut_umap", "mut_info", "mut_umap_layout"))

# ==== Transcriptomic Data ====
depmap_exp <- fread("Data/DRP_Training_Data/DepMap_21Q2_Expression.csv")
depmap_exp[1:5, 1:5]
exp_umap <- umap(depmap_exp[,-c(1:2)])
exp_info <- merge(depmap_exp[, 1:2], depmap_info, by = "stripped_cell_line_name", sort = F)

exp_umap_layout <- as.data.table(exp_umap$layout)
exp_umap_layout$primary_disease <- depmap_exp$primary_disease
# depmap_exp[,1:3]
ggplot(exp_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = primary_disease)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Primary Disease") +
  ggtitle(label = "DepMap Transcriptomic by Primary Disease")
ggsave(filename = "Plots/UMAPs/DepMap_Transcriptomics_by_Primary_Disease.pdf", height = 8, width = 16, units = "in")


exp_umap_layout$culture_type <- exp_info$culture_type
ggplot(exp_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = culture_type)) +
  geom_point(data=exp_umap_layout[culture_type == "Suspension"],
             aes(x=V1,y=V2), color="black", size=3) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Suspension Type") +
  ggtitle(label = "DepMap Transcriptomics Data by Culture Type", subtitle = "Suspended cell lines highlighted in black")
ggsave(filename = "Plots/UMAPs/DepMap_Transcriptomics_by_Culture_Type.pdf", height = 8, width = 16, units = "in")

rm(list = c("depmap_exp", "exp_umap", "exp_info", "exp_umap_layout"))
# ==== Copy Number Data ====
depmap_cnv <- fread("Data/DRP_Training_Data/DepMap_21Q2_CopyNumber.csv")
depmap_cnv[1:5, 1:5]
anyNA(depmap_cnv[,-c(1:2)])
sum(is.na(depmap_cnv[,-c(1:2)]))  # 354 NAs 
setnafill(depmap_cnv, fill = 0, cols = unique(which(is.na(depmap_cnv), arr.ind = T)[,2]))
anyNA(depmap_cnv)

cnv_umap <- umap(depmap_cnv[,-c(1:2)])
cnv_info <- merge(depmap_cnv[, 1:2], depmap_info, by = "stripped_cell_line_name", sort = F)

cnv_umap_layout <- as.data.table(cnv_umap$layout)
cnv_umap_layout$primary_disease <- depmap_cnv$primary_disease
# depmap_cnv[,1:3]
ggplot(cnv_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = primary_disease)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Primary Disease") +
  ggtitle(label = "DepMap Copy Number Variation by Primary Disease")
ggsave(filename = "Plots/UMAPs/DepMap_CopyNumber_by_Primary_Disease.pdf", height = 8, width = 16, units = "in")

cnv_umap_layout$culture_type <- cnv_info$culture_type
ggplot(cnv_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = culture_type)) +
  geom_point(data=cnv_umap_layout[culture_type == "Suspension"],
             aes(x=V1,y=V2), color="black", size=3) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Suspension Type") +
  ggtitle(label = "DepMap Copy Number Variation by Culture Type", subtitle = "Suspended cell lines highlighted in black")
ggsave(filename = "Plots/UMAPs/DepMap_CopyNumber_by_Culture_Type.pdf", height = 8, width = 16, units = "in")

rm(list = c("depmap_cnv", "cnv_umap", "cnv_info", "cnv_umap_layout"))

# ==== miRNA Data ====
depmap_mirna <- fread("Data/DRP_Training_Data/DepMap_2019_miRNA.csv")
depmap_mirna[1:5, 1:5]

mirna_umap <- umap(depmap_mirna[,-c(1:4)])
mirna_info <- merge(depmap_mirna[, 1:4], depmap_info, by = "stripped_cell_line_name", sort = F)

mirna_umap_layout <- as.data.table(mirna_umap$layout)
mirna_umap_layout$primary_disease <- depmap_mirna$primary_disease
# depmap_mirna[,1:3]
ggplot(mirna_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = primary_disease)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Primary Disease") +
  ggtitle(label = "DepMap miRNA Expression by Primary Disease")
ggsave(filename = "Plots/UMAPs/DepMap_miRNA_Expression_by_Primary_Disease.pdf", height = 8, width = 16, units = "in")

mirna_umap_layout$culture_type <- mirna_info$culture_type
ggplot(mirna_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = culture_type)) +
  geom_point(data=mirna_umap_layout[culture_type == "Suspension"],
             aes(x=V1,y=V2), color="black", size=3) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Suspension Type") +
  ggtitle(label = "DepMap miRNA Expression by Culture Type", subtitle = "Suspended cell lines highlighted in black")
ggsave(filename = "Plots/UMAPs/DepMap_miRNA_Expression_by_Culture_Type.pdf", height = 8, width = 16, units = "in")

rm(list = c("depmap_mirna", "mirna_umap", "mirna_info", "mirna_umap_layout"))

# ==== RPPA Data ====
depmap_rppa <- fread("Data/DRP_Training_Data/DepMap_2019_RPPA.csv")
depmap_rppa[1:5, 1:5]

rppa_umap <- umap(depmap_rppa[,-c(1:4)])
rppa_info <- merge(depmap_rppa[, 1:4], depmap_info, by = "stripped_cell_line_name", sort = F)

rppa_umap_layout <- as.data.table(rppa_umap$layout)
rppa_umap_layout$primary_disease <- depmap_rppa$primary_disease
# depmap_rppa[,1:3]
ggplot(rppa_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = primary_disease)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Primary Disease") +
  ggtitle(label = "DepMap RPPA Data by Primary Disease")
ggsave(filename = "Plots/UMAPs/DepMap_RPPA_by_Primary_Disease.pdf", height = 8, width = 16, units = "in")

rppa_umap_layout$culture_type <- rppa_info$culture_type
ggplot(rppa_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = culture_type)) +
  geom_point(data=rppa_umap_layout[culture_type == "Suspension"],
             aes(x=V1,y=V2), color="black", size=3) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Suspension Type") +
  ggtitle(label = "DepMap RPPA Data by Culture Type", subtitle = "Suspended cell lines highlighted in black")
ggsave(filename = "Plots/UMAPs/DepMap_RPPA_by_Culture_Type.pdf", height = 8, width = 16, units = "in")

rm(list = c("depmap_rppa", "rppa_umap", "rppa_info", "rppa_umap_layout"))

# ==== Metabolomics Data ====
depmap_metab <- fread("Data/DRP_Training_Data/DepMap_2019_Metabolomics.csv")
depmap_metab[1:5, 1:5]

metab_umap <- umap(depmap_metab[,-c(1:4)])
metab_info <- merge(depmap_metab[, 1:4], depmap_info, by = "stripped_cell_line_name", sort = F)

metab_umap_layout <- as.data.table(metab_umap$layout)
metab_umap_layout$primary_disease <- depmap_metab$primary_disease
# depmap_metab[,1:3]
ggplot(metab_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = primary_disease)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Primary Disease") +
  ggtitle(label = "DepMap Metabolomics Data by Primary Disease")
ggsave(filename = "Plots/UMAPs/DepMap_Metabolomics_by_Primary_Disease.pdf", height = 8, width = 16, units = "in")

metab_umap_layout$culture_type <- metab_info$culture_type
ggplot(metab_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = culture_type)) +
  geom_point(data=metab_umap_layout[culture_type == "Suspension"],
             aes(x=V1,y=V2), color="black", size=3) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Suspension Type") +
  ggtitle(label = "DepMap Metabolomics Data by Culture Type", subtitle = "Suspended cell lines highlighted in black")
ggsave(filename = "Plots/UMAPs/DepMap_Metabolomics_by_Culture_Type.pdf", height = 8, width = 16, units = "in")

rm(list = c("depmap_metab", "metab_umap", "metab_info", "metab_umap_layout"))

# ==== Chromatin Profiling Data ====
depmap_chrom <- fread("Data/DRP_Training_Data/DepMap_2019_ChromatinProfiling.csv")
depmap_chrom[1:5, 1:5]

chrom_umap <- umap(depmap_chrom[,-c(1:4)])
chrom_info <- merge(depmap_chrom[, 1:4], depmap_info, by = "stripped_cell_line_name", sort = F)

chrom_umap_layout <- as.data.table(chrom_umap$layout)
chrom_umap_layout$primary_disease <- depmap_chrom$primary_disease
# depmap_chrom[,1:3]
ggplot(chrom_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = primary_disease)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Primary Disease") +
  ggtitle(label = "DepMap Chromatin Profiling Data by Primary Disease")
ggsave(filename = "Plots/UMAPs/DepMap_ChromatinProfiling_by_Primary_Disease.pdf", height = 8, width = 16, units = "in")

chrom_umap_layout$culture_type <- chrom_info$culture_type
ggplot(chrom_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = culture_type)) +
  geom_point(data=chrom_umap_layout[culture_type == "Suspension"],
             aes(x=V1,y=V2), color="black", size=3) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Suspension Type") +
  ggtitle(label = "DepMap Chromatin Profiling Data by Culture Type", subtitle = "Suspended cell lines highlighted in black")
ggsave(filename = "Plots/UMAPs/DepMap_ChromatinProfiling_by_Culture_Type.pdf", height = 8, width = 16, units = "in")

rm(list = c("depmap_chrom", "chrom_umap", "chrom_info", "chrom_umap_layout"))


# ==== Chromatin Profiling Data ====
depmap_chrom <- fread("Data/DRP_Training_Data/DepMap_2019_ChromatinProfiling.csv")
depmap_chrom[1:5, 1:5]

chrom_umap <- umap(depmap_chrom[,-c(1:4)])
chrom_info <- merge(depmap_chrom[, 1:4], depmap_info, by = "stripped_cell_line_name", sort = F)

chrom_umap_layout <- as.data.table(chrom_umap$layout)
chrom_umap_layout$primary_disease <- depmap_chrom$primary_disease
# depmap_chrom[,1:3]
ggplot(chrom_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = primary_disease)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Primary Disease") +
  ggtitle(label = "DepMap Chromatin Profiling Data by Primary Disease")
ggsave(filename = "Plots/UMAPs/DepMap_ChromatinProfiling_by_Primary_Disease.pdf", height = 8, width = 16, units = "in")

chrom_umap_layout$culture_type <- chrom_info$culture_type
ggplot(chrom_umap_layout) +
  geom_point(aes(x = V1, y = V2, col = culture_type)) +
  geom_point(data=chrom_umap_layout[culture_type == "Suspension"],
             aes(x=V1,y=V2), color="black", size=3) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Suspension Type") +
  ggtitle(label = "DepMap Chromatin Profiling Data by Culture Type", subtitle = "Suspended cell lines highlighted in black")
ggsave(filename = "Plots/UMAPs/DepMap_ChromatinProfiling_by_Culture_Type.pdf", height = 8, width = 16, units = "in")

rm(list = c("depmap_chrom", "chrom_umap", "chrom_info", "chrom_umap_layout"))



# ==== GDSC Drug Data ====
require(stringr)
ctrp_train <- fread("Data/DRP_Training_Data/CTRP_AAC_MORGAN_512.csv")
ctrp_morgan <- unique(ctrp_train$morgan)
ctrp_morgan <- str_split_fixed(ctrp_morgan, "", n = 512)
storage.mode(ctrp_morgan) <- "numeric"
ctrp_morgan <- as.data.table(ctrp_morgan)

dim(ctrp_morgan)
ctrp_morgan[1:5, 1:5]

ctrp_umap <- umap(ctrp_morgan)
# chrom_info <- merge(depmap_chrom[, 1:4], depmap_info, by = "stripped_cell_line_name", sort = F)

ctrp_umap_layout <- as.data.table(ctrp_umap$layout)
# ctrp_umap_layout$primary_disease <- depmap_ctrp$primary_disease
# depmap_ctrp[,1:3]
ggplot(ctrp_umap_layout) +
  geom_point(aes(x = V1, y = V2)) +
  xlab("UMAP_1") + ylab("UMAP_2") + 
  labs(color = "Primary Disease") +
  ggtitle(label = "DepMap CTRP Drug Morgan Fingerprints")
ggsave(filename = "Plots/UMAPs/DepMap_CTRP_Morgan.pdf", height = 8, width = 16, units = "in")

# ctrp_umap_layout$culture_type <- ctrp_info$culture_type
# ggplot(ctrp_umap_layout) +
#   geom_point(aes(x = V1, y = V2, col = culture_type)) +
#   geom_point(data=ctrp_umap_layout[culture_type == "Suspension"],
#              aes(x=V1,y=V2), color="black", size=3) +
#   xlab("UMAP_1") + ylab("UMAP_2") + 
#   labs(color = "Suspension Type") +
#   ggtitle(label = "DepMap ctrpatin Profiling Data by Culture Type", subtitle = "Suspended cell lines highlighted in black")
# ggsave(filename = "Plots/UMAPs/DepMap_ctrpatinProfiling_by_Culture_Type.pdf", height = 8, width = 16, units = "in")

rm(list = c("depmap_chrom", "chrom_umap", "chrom_info", "chrom_umap_layout"))


