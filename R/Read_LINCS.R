# Read_LINCS.R
# library(devtools)
# install_github("cmap/cmapR")
library(cmapR)
# LINCSDataPortal::download_dataset()

library(data.table)

# ds_path = "Data/LINCS/GSE70138/GSE70138_Broad_LINCS_Level3_INF_mlr12k_n345976x12328_2017-03-06.gctx"
ds_path = "Data/LINCS/GSE70138/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx"
# read the column annotations as a data.frame
# col_meta_path <- "Data/LINCS/GSE70138/GSE70138_Broad_LINCS_inst_info_2017-03-06.txt"
col_meta_path <- "Data/LINCS/GSE70138/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt"
col_meta <- fread(col_meta_path)
head(unique(col_meta$pert_iname)) # Drugs and shRNAs
unique(col_meta$pert_type)


cell_meta_path <- "Data/LINCS/GSE70138/GSE70138_Broad_LINCS_cell_info_2017-04-28.txt"
cell_meta <- fread(cell_meta_path)

gene_meta_path <- "Data/LINCS/GSE70138/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt"
gene_meta <- fread(gene_meta_path)

sig_metrics <- fread("Data/LINCS/GSE70138/GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt")
sig_metrics[pert_type == "trt_xpr"]

pert_meta_path <- "Data/LINCS/GSE70138/GSE70138_Broad_LINCS_pert_info_2017-03-06.txt"
pert_meta <- fread(pert_meta_path)
unique(pert_meta$pert_type)
pert_meta[pert_type == "trt_cp"]
pert_meta[pert_type == "ctl_vehicle"]

# figure out which signatures correspond to vorinostat by searching the 'pert_iname' column
col_meta
col_meta[pert_iname == "vorinostat" & cell_id == "A375"]
idx <- which(col_meta$pert_iname=="vorinostat")

# and get the corresponding sig_ids
sig_ids <- col_meta$sig_id[idx]

# read only those columns from the GCTX file by using the 'cid' parameter
cur_cols <- read.gctx.ids(ds_path, dimension = "col")

sum(grepl(pattern = "REP", x = cur_cols))
cur_cols <- gsub(pattern = "REP\\.", replacement = "", x = cur_cols)
sum(cur_cols %in% col_meta$sig_id)
vorinostat_ds <- parse.gctx(ds_path, cid=sig_ids)

# simply typing the variable name of a GCT object will display the object structure, similar to calling 'str'
vorinostat_ds


# ==== Get all untreated cell line expression data ====
untreated_ids <- col_meta[pert_iname == "DMSO", sig_id]
untreated_ds <- parse.gctx(ds_path, cid = untreated_ids)
untreated_dt <- as.data.table(untreated_ds@mat)
dim(untreated_dt)
# Get cell IDs, maintaining column order (%in% discards order!)
col_meta[sig_id %in% c("LJP005_A375_24H:A07", "LJP005_A375_24H:A03"), pert_id]
cur_cells <- col_meta[match(colnames(untreated_dt), col_meta$sig_id), cell_id]
cell_meta[match(cur_cells, cell_meta$cell_id), primary_site]
y = cell_meta[match(cur_cells, cell_meta$cell_id), subtype]
length(unique(y)) # 17 classes/subtypes

# Transpose the gene expression matrix
untreated_dt <- transpose(untreated_dt)
dim(untreated_dt)
colnames(untreated_dt)
untreated_dt <- cbind(untreated_dt, y)

# Remove samples with no subtypes indicated
untreated_dt <- untreated_dt[y != "-666"]
dim(untreated_dt)
untreated_dt[1:5, 1:5]

# Stratify into train and test data
# install.packages("caret")
library(caret)

temp <- caret::createDataPartition(y = as.factor(untreated_dt$y), times = 1, p = 0.8)
temp$Resample1
train_data <- untreated_dt[temp$Resample1,]
test_data <- untreated_dt[-temp$Resample1]
dim(train_data)
dim(test_data)

# Save
fwrite(train_data, "Data/LINCS/GSE70138/DMSO_celllines_subtypes_TRAIN.txt", sep = "\t")
fwrite(test_data, "Data/LINCS/GSE70138/DMSO_celllines_subtypes_TEST.txt", sep = "\t")

# ==== Match (compound) treatment and cell line responses ====
# Will have before (X) and after (y) treatment responses
untreated_ids <- col_meta[pert_id == "DMSO", sig_id]
ctrl <- parse.gctx(fname = ds_path, cid = untreated_ids)
gene_ids <- ctrl@rid
ctrl <- as.data.table(ctrl@mat)

# Divide training data by cell lines
all_lines <- unique(col_meta$cell_id)
cell = all_lines[1]
for (cell in all_lines) {
  cur_ids <- col_meta[cell_id == cell & pert_id != "DMSO" & pert_type == "trt_cp", sig_id]
  cur_data <- parse.gctx(fname = ds_path, cid = cur_ids, matrix_only = T)
  cur_gene_ids <- cur_data@rid
  # all(cur_gene_ids == gene_ids)
  cur_data <- as.data.table(cur_data@mat)
  cur_perts <- col_meta[match(colnames(cur_data), col_meta$sig_id), pert_id]
  # Get the SMILES for current perturbations, in order
  cur_smiles <- pert_meta[match(cur_perts, pert_meta$pert_id), canonical_smiles]
  # Match current cell line controls, duplicate as necessary (sample with replacement)
  # TODO: How to augment the data?
  cur_ctrl <- ctrl[, col_meta[cell_id == cell & pert_id == "DMSO", sig_id], with = F]
  cur_ctrl <- cur_ctrl[, sample(x = 1:ncol(cur_ctrl), size = ncol(cur_data), replace = T), with = F]
  # dim(cur_ctrl)
  # dim(cur_data)
  # length(cur_smiles)
  
  # Save Chem X
  fwrite(as.data.table(cur_smiles), paste0(cell, "_SMILES_data.txt"))
  # Save Cell X
  fwrite(transpose(cur_ctrl), paste0(cell, "_cellline_ctrl_input_data.txt"))
  # Save Cell y
  fwrite(transpose(cur_data), paste0(cell, "_cellline_trt_label_data.txt"))
}


all <- parse.gctx(fname = ds_path, matrix_only = T)
dim(ctrl)



