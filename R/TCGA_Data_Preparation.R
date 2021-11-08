# TCGA_Data_Preparation.R

getGDCprojects()

# Download_GDC.R

# ==== Preperation ====
if (!require(SummarizedExperiment)) {
  install.packages("SummarizedExperiment")
  library(SummarizedExperiment)
}
if (!require(parallel)) {
  install.packages("parallel")
  library(parallel)
}
if (!require(data.table)) {
  install.packages("data.table")
  library(data.table)
}
# Note that on Linux, libmariadb-client-lgpl-dev must be installed
# This allows for configuration of RMySQL
if (!require(TCGAbiolinks)) {
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
  BiocManager::install("TCGAbiolinks")
  require(TCGAbiolinks)
  ??TCGAbiolinks
  
  library(TCGAbiolinks)
}

getGDCInfo()
getGDCprojects()
getProjectSummary("TCGA-ACC")
# Data Release 28, February 02, 2021
# Move all data to a single directory later on
dir.create(("Data/TCGA"))
dir.create("Data/TCGA/SummarizedExperiments")
dir.create("Data/TCGA/SummarizedExperiments/Mut")
dir.create("Data/TCGA/SummarizedExperiments/CNV")
dir.create("Data/TCGA/SummarizedExperiments/Exp")

projects <- TCGAbiolinks:::getGDCprojects()$project_id
projects <- projects[grepl('^TCGA', projects, perl=TRUE)]


# ==== Download Mut data ====
plyr::alply(sort(projects), 1, function(cur_project) {
  tryCatch(
    query <- GDCquery(
      project = cur_project,
      data.category = "Simple Nucleotide Variation",
      data.type = "Masked Somatic Mutation",
      workflow.type = 'VarScan2 Variant Aggregation and Masking'
    )
  )
  query <- GDCquery(
    project = cur_project,
    data.category = "Simple Nucleotide Variation",
    data.type = "Masked Somatic Mutation",
    workflow.type = 'VarScan2 Variant Aggregation and Masking'
  )
  
  GDCdownload(query, files.per.chunk = 10, method = "client")
  GDCprepare(query,
    save = T,
    save.filename = paste0("Data/TCGA/SummarizedExperiments/Mut/", cur_project,
                           "_VarScan2_MAF.rdata")
  )
})

# ==== Download CNV data ====
plyr::alply(sort(projects)[-(1:14)], 1, function(cur_project) {
  tryCatch(
    query <- GDCquery(
      project = cur_project,
      data.category = "Copy Number Variation",
      data.type = "Masked Copy Number Segment", 
    )
  )
  query <- GDCquery(
    project = cur_project,
    data.category = "Copy Number Variation",
    data.type = "Masked Copy Number Segment",
  )
  GDCdownload(query, files.per.chunk = 10)
  GDCprepare(query,
             save = T,
             save.filename = paste0("Data/TCGA/SummarizedExperiments/CNV/", cur_project,
                                    "_CopyNumber.rdata"))
  # cur_proj_data <- get(load(paste0("Data/TCGA/SummarizedExperiments/CNV/", cur_project, "_CopyNumber.rdata")))
  
})

# ==== Download clinical CNV data ====

dir.create("Data/TCGA/Clinical")
dir.create("Data/TCGA/Clinical/CNV")
plyr::alply(sort(projects)[-(1:14)], 1, function(cur_project) {
  tryCatch(
    query.biospecimen <- GDCquery(project = cur_project, 
                                  data.category = "Biospecimen",
                                  data.type = "Biospecimen Supplement", 
                                  data.format = "BCR Biotab")
  )
  query.biospecimen <- GDCquery(project = cur_project, 
                                data.category = "Biospecimen",
                                data.type = "Biospecimen Supplement", 
                                data.format = "BCR Biotab")
  GDCdownload(query.biospecimen, files.per.chunk = 10)
  GDCprepare(query.biospecimen,
             save = T,
             save.filename = paste0("Data/TCGA/Clinical/CNV/", cur_project,
                                    "_CopyNumber_Clinical.rdata"))
  
})

# ==== Download gene expression data ====
plyr::alply(projects, 1, function(cur_project) {
  tryCatch(
      query <- GDCquery(
          project = cur_project,
          data.category = "Transcriptome Profiling",
          data.type = "Gene Expression Quantification",
          workflow.type = "HTSeq - Counts"
      )
  )
  query <- GDCquery(
    project = cur_project,
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    workflow.type = "HTSeq - Counts"
  )
  GDCdownload(query, files.per.chunk = 10, method = "client")
  GDCprepare(
    query,
    save = T,
    save.filename = paste0("Data/TCGA/SummarizedExperiments/Exp/", cur_project, "_GeneExpression.rdata")
  )
  
})


# =====================================
# For each data type, combine all projects into a single file
require(data.table)
require(TCGAbiolinks)
require(SummarizedExperiment)
require(GeoTcgaData)
require(stringr)
require(biomaRt)
require(curl)

# Get clinical info from Gene Expression data ====
all_proj_info <- vector(mode = "list", length = length(projects))
i <- 1
for (cur_project in projects) {
  cur_proj_data <- get(load(paste0("Data/TCGA/SummarizedExperiments/Exp/", cur_project, "_GeneExpression.rdata")))
  cur_proj_assay <- assays(cur_proj_data)[[1]]
  cur_proj_info <- colData(cur_proj_data)
  all_proj_info[[i]] <- as.data.table(cur_proj_info[, c("barcode", "name")])
  i <- i + 1
}

all_proj_info <- rbindlist(all_proj_info)
# Save
fwrite(all_proj_info, "Data/TCGA/All_Sample_Info.csv")


# ==== Merge Exp Data ====
require(data.table)
require(biomaRt)
dir.create("Data/TCGA/Exp")
projects <- TCGAbiolinks:::getGDCprojects()$project_id
projects <- projects[grepl('^TCGA', projects, perl=TRUE)]
# gene_dict <- fread("Data/gene_dictionary.txt")
depmap_exp <- fread("Data/DepMap/20Q2/CCLE_expression.csv")
depmap_ensg_numbers <- str_replace(colnames(depmap_exp), ".+\\((.+)\\)", "\\1")
depmap_hgnc <- str_replace(colnames(depmap_exp), "(.+)\\s\\(.+\\)", "\\1")
depmap_exp_dict <- data.table(HGNC = depmap_hgnc, ENSG = depmap_ensg_numbers)

rm(depmap_exp)

ensembl = useMart(
  biomart = "ENSEMBL_MART_ENSEMBL",
  # host = "feb2014.archive.ensembl.org",
  path = "/biomart/martservice",
  dataset = "hsapiens_gene_ensembl"
)
atts <- listAttributes(ensembl, what = "name")
atts[grep(pattern = "gene", x = atts, ignore.case = T)]

cur_project = projects[1]

for (cur_project in projects) {
  cur_proj_data <- get(load(paste0("Data/TCGA/SummarizedExperiments/Exp/", cur_project, "_GeneExpression.rdata")))
  cur_proj_assay <- assays(cur_proj_data)[[1]]
  cur_proj_info <- colData(cur_proj_data)
  cur_proj_tpm <- GeoTcgaData::fpkmToTpm_matrix(cur_proj_assay)
  
  # log2 transform with a pseudo-count of 1
  cur_proj_tpm <- cur_proj_tpm+1
  cur_proj_tpm <- log2(cur_proj_tpm)
  
  # Convert ENSG ID's to numbers
  cur_ensg_numbers <- str_replace(rownames(cur_proj_tpm), "ENSG0*", "")
  cur_proj_tpm_dt <- as.data.table(cur_proj_tpm, keep.rownames = T)
  cur_dict <- biomaRt::getBM(attributes = c("ensembl_gene_id",
                                            "external_gene_name"),
                         filters = "ensembl_gene_id", values = cur_proj_tpm_dt$rn,
                         mart = ensembl)
  cur_dict <- as.data.table(cur_dict)
  temp <- merge(cur_proj_tpm_dt, cur_dict, by.x = "rn", by.y = "ensembl_gene_id")
  # Find overlap with HGNC symbols in DepMap 
  cur_reduced_dt <- temp[external_gene_name %in% depmap_exp_dict$HGNC]
  
  dup_genes <- cur_reduced_dt[duplicated(cur_reduced_dt$external_gene_name)]$external_gene_name
  # Duplicated genes may represent transcripts from the same gene, must consolidate by adding TPM values
  cur_reduced_dt$rn <- NULL
  cur_reduced_dt <- cur_reduced_dt[, lapply(.SD, sum), by = external_gene_name]
  final_dt <- transpose(cur_reduced_dt, keep.names = "external_gene_name")
  
  colnames(final_dt) <- unname(unlist(final_dt[1,]))
  final_dt <- final_dt[!(TSPAN6 == "TSPAN6")]
  # Add cancer type
  final_dt <- cbind(cancer_type = cur_proj_info$name, final_dt)
  colnames(final_dt)[2] <- "tcga_sample_id"
  # Sort all the other columns
  new_order <- c("tcga_sample_id", "cancer_type", sort(colnames(final_dt)[-c(1:2)]))
  setcolorder(final_dt, new_order)
  
  fwrite(final_dt, paste0("Data/TCGA/Exp/", cur_project, "_Exp.csv"))
}

# Now merge all the data.tables
exp_file_paths <- list.files("Data/TCGA/Exp/", full.names = T)
exp_dt_list <- vector(mode = "list", length = length(exp_file_paths))

for (i in 1:length(exp_file_paths)) {
  exp_dt_list[[i]] <- fread(exp_file_paths[i])
}

all_exp <- rbindlist(exp_dt_list)
all_exp[1:5, 1:5]
dim(all_exp)
fwrite(all_exp, "Data/TCGA/TCGA_All_Exp_Data.csv")
rm(exp_dt_list)
rm(all_exp)

# all_exp <- fread("Data/TCGA/TCGA_All_Exp_Data.csv")
# ==== Merge CNV Data ====
require(GenomicRanges)
# Must find the genomic ranges of genes in DepMap CNV data (ensure reference is the same), then merge using the GRanges package (map onto)
dir.create("Data/TCGA/CNV")
projects <- TCGAbiolinks:::getGDCprojects()$project_id
projects <- projects[grepl('^TCGA', projects, perl=TRUE)]
# gene_dict <- fread("Data/gene_dictionary.txt")
depmap_cnv <- fread("Data/DepMap/20Q2/CCLE_gene_copy_number.csv")
head(colnames(depmap_cnv))
depmap_ensg_numbers <- str_replace(colnames(depmap_cnv), ".+\\((.+)\\)", "\\1")
depmap_hgnc <- str_replace(colnames(depmap_cnv), "(.+)\\s\\(.+\\)", "\\1")
depmap_cnv_dict <- data.table(HGNC = depmap_hgnc, ENSG = depmap_ensg_numbers)

rm(depmap_cnv)

ensembl = useMart(
  biomart = "ENSEMBL_MART_ENSEMBL",
  # host = "feb2014.archive.ensembl.org",
  path = "/biomart/martservice",
  dataset = "hsapiens_gene_ensembl"
)

cur_translation <- biomaRt::getBM(attributes = c("ensembl_gene_id", "wikigene_name",
                                                 "external_gene_name", 'chromosome_name',
                                                 'start_position', 'end_position', 'strand'),
                           filters = "external_gene_name", values = depmap_cnv_dict$HGNC, uniqueRows = T,
                           mart = ensembl)
cur_translation <- as.data.table(cur_translation)
cur_translation <- cur_translation[!duplicated(cur_translation[,"wikigene_name"]), ]
cur_translation <- cur_translation[!duplicated(cur_translation[,"external_gene_name"]), ]

depmap_cnv_dict_1 <- merge(depmap_cnv_dict, cur_translation, by.x = "HGNC", by.y = "external_gene_name")
depmap_cnv_dict_1$wikigene_name <- NULL
# unfound <- depmap_cnv_dict$HGNC[!(depmap_cnv_dict$HGNC %in% cur_translation$wikigene_name)]
unfound <- depmap_cnv_dict$HGNC[!(depmap_cnv_dict$HGNC %in% cur_translation$external_gene_name)]
# Search for the unfound in wikigene info
unfound_translation <- biomaRt::getBM(attributes = c("ensembl_gene_id", "wikigene_name",
                                                 "external_gene_name", 'chromosome_name',
                                                 'start_position', 'end_position', 'strand'),
                                  filters = "wikigene_name", values = unfound, uniqueRows = T,
                                  mart = ensembl)
unfound_translation <- as.data.table(unfound_translation)
depmap_cnv_dict_2 <- merge(depmap_cnv_dict, unfound_translation, by.x = "HGNC", by.y = "wikigene_name")
depmap_cnv_dict_2$external_gene_name <- NULL

# Add together
# all_translation <- rbindlist(list(cur_translation, unfound_translation))
depmap_cnv_dict <- rbindlist(list(depmap_cnv_dict_1, depmap_cnv_dict_2))
depmap_cnv_dict$strand[depmap_cnv_dict$strand == -1] <- "-"
depmap_cnv_dict$strand[depmap_cnv_dict$strand == 1] <- "+"

# Convert to GRanges
depmap_cnv_granges <- makeGRangesFromDataFrame(df = depmap_cnv_dict, keep.extra.columns = T, seqnames.field = "chromosome_name",
                         start.field = "start_position", end.field = "end_position", strand.field = "strand")

atts <- listAttributes(ensembl, what = "name")
atts[grep(pattern = "gene", x = atts, ignore.case = T)]
atts[grep(pattern = "chromosome", x = atts, ignore.case = T)]

cur_project = projects[1]

rm(list = c("depmap_cnv_dict_1", "depmap_cnv_dict_2"))


for (cur_project in projects) {
  cur_proj_data <- get(load(paste0("Data/TCGA/SummarizedExperiments/CNV/", cur_project, "_CopyNumber.rdata")))
  # Convert to GRanges
  cur_proj_granges <- makeGRangesFromDataFrame(df = cur_proj_data, keep.extra.columns = T, start.field = "Start",
                                               end.field = "End", seqnames.field = "Chromosome")
  # Merge with DepMap data
  cur_proj_merge <- mergeByOverlaps(query = cur_proj_granges, subject = depmap_cnv_granges)
  cur_proj_merge <- cur_proj_merge[, c("Sample", "HGNC", "Segment_Mean")]
  cur_proj_merge <- as.data.table(cur_proj_merge)
  cur_proj_merge <- unique(cur_proj_merge)
  
  cur_wide_data <- dcast(cur_proj_merge, Sample ~ HGNC, value.var = "Segment_Mean", fun.aggregate = sum)
  
  colnames(cur_wide_data)[1] <- "tcga_sample_id"
  # Sort all the other columns
  new_order <- c("tcga_sample_id", sort(colnames(cur_wide_data)[-1]))
  setcolorder(cur_wide_data, new_order)

  fwrite(cur_wide_data, paste0("Data/TCGA/CNV/", cur_project, "_CNV.csv"))
}

# Now merge all the data.tables
cnv_file_paths <- list.files("Data/TCGA/CNV/", full.names = T)
cnv_dt_list <- vector(mode = "list", length = length(cnv_file_paths))

for (i in 1:length(cnv_file_paths)) {
  cnv_dt_list[[i]] <- fread(cnv_file_paths[i])
}

all_cnv <- rbindlist(cnv_dt_list)
all_cnv[1:5, 1:5]
fwrite(all_cnv, "Data/TCGA/TCGA_All_CNV_Data.csv")

# The first 3 ID's should be enough to indicate tumor type
require(stringr)
all_proj_info <- fread("Data/TCGA/All_Sample_Info.csv")
temp <- str_split_fixed(all_proj_info$barcode, "-", n = 7)[, 1:3]
all_proj_info$sub_id <- paste(temp[,1], temp[,2], temp[,3], sep = '-')

all_cnv <- fread("Data/TCGA/TCGA_All_CNV_Data.csv")
temp2 <- str_split_fixed(all_cnv$tcga_sample_id, "-", n = 7)[, 1:4]
sample_type <- str_split_fixed(temp2[, 4], "", n=3)[, 1:2]
sample_type <- as.integer(paste0(sample_type[,1], sample_type[,2]))
all_cnv$sub_id <- paste(temp2[,1], temp2[,2], temp2[,3], sep = '-')

all_cnv$sample_type <- ifelse(sample_type < 10, yes = "Tumor", no = "Normal")
table(all_cnv$sample_type)

# Subset to Tumor
all_cnv <- all_cnv[sample_type == "Tumor"]
# add cancer type
all_cnv <- merge(all_cnv, all_proj_info[, c("name", "sub_id")], by = "sub_id")
table(all_cnv$name)
dim(all_cnv)
# remove sub_id, sample_type
all_cnv$sub_id <- NULL
all_cnv$sample_type <- NULL
# reorder
setcolorder(all_cnv, "name")
colnames(all_cnv)[1] <- "cancer_type"
all_cnv[1:5, 1:5]

# Save
fwrite(all_cnv, "Data/TCGA/TCGA_All_CNV_Data.csv")

dim(all_cnv)

rm(cnv_dt_list)
rm(all_cnv)


# ==== Merge Mut Data ====
# dir.create("Data/TCGA/Mut")
# projects <- TCGAbiolinks:::getGDCprojects()$project_id
# projects <- projects[grepl('^TCGA', projects, perl=TRUE)]
# # gene_dict <- fread("Data/gene_dictionary.txt")
# depmap_mut <- fread("Data/DRP_Training_Data/DepMap_20Q2_CGC_Mutations_by_Cell.csv")
# depmap_ensg_numbers <- str_replace(colnames(depmap_mut), ".+\\((.+)\\)", "\\1")
# depmap_hgnc <- str_replace(colnames(depmap_mut), "(.+)\\s\\(.+\\)", "\\1")
# depmap_mut_dict <- data.table(HGNC = depmap_hgnc, ENSG = depmap_ensg_numbers)
# 
# rm(depmap_mut)
# 
# ensembl = useMart(
#   biomart = "ENSEMBL_MART_ENSEMBL",
#   # host = "feb2014.archive.ensembl.org",
#   path = "/biomart/martservice",
#   dataset = "hsapiens_gene_ensembl"
# )
# atts <- listAttributes(ensembl, what = "name")
# 
# # cur_project = projects[1]
# 
# for (cur_project in projects) {
#   cur_proj_data <- get(load(paste0("Data/TCGA/SummarizedExperiments/Mut/", cur_project, "_VarScan2_MAF.rdata")))
#   cur_sub <- cur_proj_data[, c("Hugo_Symbol", "Variant_Classification", "Variant_Type", "Tumor_Sample_Barcode")]
#   unique(cur_sub$Hugo_Symbol)
#   # Find overlap with DepMap
#   sum(depmap_mut_dict$HGNC %in% unique(cur_sub$Hugo_Symbol))
#   
#   
#   cur_proj_assay <- assays(cur_proj_data)[[1]]
#   # cur_proj_assay[1:5, 1:5]
#   # class(cur_proj_assay)
#   cur_proj_tpm <- GeoTcgaData::fpkmToTpm_matrix(cur_proj_assay)
#   # Convert ENSG ID's to numbers
#   cur_ensg_numbers <- str_replace(rownames(cur_proj_tpm), "ENSG0*", "")
#   cur_proj_tpm_dt <- as.data.table(cur_proj_tpm, keep.rownames = T)
#   # cur_proj_tpm_dt[1:5, 1:5]
#   # cur_proj_tpm_dt$rn <- str_replace(cur_proj_tpm_dt$rn, "ENSG0*", "")
#   cur_dict <- biomaRt::getBM(attributes = c("ensembl_gene_id",
#                                             "external_gene_name"),
#                              filters = "ensembl_gene_id", values = cur_proj_tpm_dt$rn,
#                              mart = ensembl)
#   cur_dict <- as.data.table(cur_dict)
#   temp <- merge(cur_proj_tpm_dt, cur_dict, by.x = "rn", by.y = "ensembl_gene_id")
#   # dim(temp)
#   # Find overlap with HGNC symbols in DepMap 
#   cur_reduced_dt <- temp[external_gene_name %in% depmap_exp_dict$HGNC]
#   # dim(cur_reduced_dt)
#   
#   # sum(duplicated(cur_reduced_dt$external_gene_name))
#   dup_genes <- cur_reduced_dt[duplicated(cur_reduced_dt$external_gene_name)]$external_gene_name
#   # Duplicated genes may represent transcripts from the same gene, must consolidate by adding TPM values
#   cur_reduced_dt$rn <- NULL
#   cur_reduced_dt <- cur_reduced_dt[, lapply(.SD, sum), by = external_gene_name]
#   final_dt <- transpose(cur_reduced_dt, keep.names = "external_gene_name")
#   colnames(final_dt) <- unname(unlist(final_dt[1,]))
#   # final_dt[1:5, 1:5]
#   final_dt <- final_dt[!(TSPAN6 == "TSPAN6")]
#   # final_dt[1:5, 1:5]
#   # dim(final_dt)
#   
#   # temp1[external_gene_name %in% dup_genes][,c("external_gene_name", "TCGA-FI-A2D5-01A-11R-A17B-07")]
#   fwrite(final_dt, paste0("Data/TCGA/Exp/", cur_project, "_Exp.csv"))
# }
# 
# # Now merge all the data.tables
# exp_file_paths <- list.files("Data/TCGA/Exp/", full.names = T)
# exp_dt_list <- vector(mode = "list", length = length(exp_file_paths))
# 
# for (i in 1:length(exp_file_paths)) {
#   exp_dt_list[[i]] <- fread(exp_file_paths[i])
# }
# 
# all_exp <- rbindlist(exp_dt_list)
# fwrite(all_exp, "Data/TCGA/TCGA_All_Exp_Data.csv")
# 


# ==== Sort and Clean Data ====

# ==== Match TCGA and DepMap Exp Data ====
require(data.table)
exp_tcga <- fread("Data/TCGA/TCGA_All_Exp_Data.csv")
exp_tcga[1:5, 1:5]
exp_depmap <- fread("Data/DRP_Training_Data/DepMap_21Q2_Expression.csv")
exp_depmap[1:5, 1:5]
dim(exp_tcga)
dim(exp_depmap)

# Find shared columns:
shared_exp <- intersect(colnames(exp_depmap), colnames(exp_tcga))
# Sort columns
shared_exp <- sort(shared_exp)
shared_exp <- c("tcga_sample_id", "cancer_type", shared_exp)
exp_tcga <- exp_tcga[, ..shared_exp]
colnames(exp_tcga)[2] <- "primary_disease"

exp_tcga[1:5, 1:5]
dim(exp_tcga)

# Save TCGA pre-training data
fwrite(exp_tcga, "Data/DRP_Training_Data/TCGA_PreTraining_Expression.csv")

# Subset DepMap data to what's available in TCGA
shared_exp <- unique(c("stripped_cell_line_name", "primary_disease", sort(intersect(colnames(exp_depmap), colnames(exp_tcga)))))
exp_depmap <- exp_depmap[, ..shared_exp]
exp_depmap[1:5, 1:5]
dim(exp_depmap)
# Save DepMap training data
fwrite(exp_depmap, "Data/DRP_Training_Data/DepMap_21Q2_Training_Expression.csv")

rm(exp_depmap)
rm(exp_tcga)
# exp_tcga <- fread("Data/DRP_Training_Data/TCGA_PreTraining_Expression.csv")
# exp_depmap <- fread("Data/DRP_Training_Data/DepMap_20Q2_Training_Expression.csv")
# exp_tcga[1:5, 1:5]
# exp_depmap[1:5, 1:5]

# ==== Match TCGA and DepMap CNV Data ====
cnv_tcga <- fread("Data/TCGA/TCGA_All_CNV_Data.csv")
cnv_tcga[1:5, 1:5]
cnv_depmap <- fread("Data/DRP_Training_Data/DepMap_21Q2_CopyNumber.csv")
cnv_depmap[1:5, 1:5]
dim(cnv_depmap)
dim(cnv_tcga)

shared_cnv <- intersect(colnames(cnv_depmap), colnames(cnv_tcga))
# Sort columns
shared_cnv <- sort(shared_cnv)
shared_cnv <- c("tcga_sample_id", "cancer_type", shared_cnv)
cnv_tcga <- cnv_tcga[, ..shared_cnv]
colnames(cnv_tcga)[2] <- "primary_disease"

cnv_tcga[1:5, 1:5]

# Must add pseudocounts to log2 ratios of TCGA data
temp <- cnv_tcga[, -c(1:2)][, lapply(.SD, function(x) {log2(2**x + 1)})]
temp[1:5, 1:5]
temp$tcga_sample_id <- cnv_tcga$tcga_sample_id
temp$primary_disease <- cnv_tcga$primary_disease
setcolorder(temp, c("tcga_sample_id", "primary_disease"))
temp[1:5, 1:5]
dim(temp)

fwrite(temp, "Data/DRP_Training_Data/TCGA_PreTraining_CopyNumber.csv")


# Subset DepMap data to what's available in TCGA
cnv_depmap[1:5, 1:5]
shared_cnv <- unique(c("stripped_cell_line_name", "primary_disease", sort(intersect(colnames(cnv_depmap), colnames(cnv_tcga)))))
cnv_depmap <- cnv_depmap[, ..shared_cnv]

cnv_depmap[1:5, 1:5]
dim(cnv_depmap)
# Save DepMap training data
fwrite(cnv_depmap, "Data/DRP_Training_Data/DepMap_21Q2_Training_CopyNumber.csv")

rm(cnv_depmap)
rm(cnv_tcga)
gc()
