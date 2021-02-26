# TCGA_Data_Preparation.R

getGDCprojects()
query <- GDCquery(project = )

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
      data.type = "Annotated Somatic Mutation"
    )
  )
  query <- GDCquery(
    project = cur_project,
    data.category = "Simple Nucleotide Variation",
    data.type = "Annotated Somatic Mutation"
  )
  
  GDCdownload(query, files.per.chunk = 10, method = "client")
  GDCprepare(query,
    save = T,
    save.filename = paste0("Data/TCGA/SummarizedExperiments/Mut/", cur_project,
                           "_microRNAexpression.rdata")
  )
})

# ==== Download CNV data ====
plyr::alply(sort(projects), 1, function(cur_project) {
  tryCatch(
    query <- GDCquery(
      project = cur_project,
      data.category = "Copy Number Variation",
      data.type = "Copy Number Segment",
      platform = "Affymetrix SNP 6.0"
    )
  )
  query <- GDCquery(
    project = cur_project,
    data.category = "Copy Number Variation",
    data.type = "Copy Number Segment",
    platform = "Affymetrix SNP 6.0"
  )
  GDCdownload(query, files.per.chunk = 10)
  GDCprepare(query,
             save = T,
             save.filename = paste0("Data/TCGA/SummarizedExperiments/CNV/", cur_project,
                                    "_CopyNumber.rdata"))
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


# We can use the rename function to move the file to a new path (note that the
# original files are deleted, use file.move() if this is not desired)
# oldPaths = list.files(pattern = ".*microrna*",
#                          full.names = T, ignore.case = T)
# newPaths = paste0("SummarizedExperiments/miRNA/", basename(oldPaths))
# file.rename(from = oldPaths, to = newPaths)
# 
# oldPaths = list.files(pattern = ".*copynumber*",
#                          full.names = T, ignore.case = T)
# newPaths = paste0("SummarizedExperiments/CNA/", basename(oldPaths))
# file.rename(from = oldPaths, to = newPaths)
# 
# oldPaths = list.files(pattern = ".*geneexpression.*",
#                          full.names = T, ignore.case = T)
# newPaths = paste0("SummarizedExperiments/mRNA/", basename(oldPaths))
# file.rename(from = oldPaths, to = newPaths)

# [END]
