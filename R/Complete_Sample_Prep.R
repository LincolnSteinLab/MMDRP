# Complete_Sample_Prep.R

# This script is intended to pair genomics, transcriptomics, proteomics and drug response data
# mainly from the DepMap resource.
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/"
dir.create(paste0(path, "Data/DRP_Training_Data"))

require(data.table)

# ==== Cell line info cleanup ====
depmap_samples <- fread(paste0(path, "Data/DepMap/21Q2/sample_info.csv"))
# Subset relevant (machine learning) columns 
depmap_samples <- depmap_samples[, c("DepMap_ID", "stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype")]

fwrite(depmap_samples, paste0(path, "Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv"))

# depmap_samples <- fread("Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv")
# ==== Expression data cleanup ====
ccle_exp <- fread(paste0(path, "Data/DepMap/21Q2/CCLE_expression.csv"))
max(ccle_exp[, -1])
min(ccle_exp[, -1])
dim(ccle_exp)
ccle_exp[1:5, 1:20]
# Change column names to only contain HGNC name: replace everything after first word with ""
colnames(ccle_exp) <- gsub(" .+", "", colnames(ccle_exp))
colnames(ccle_exp)[1] <- "DepMap_ID"
# Merge with sample info to have cell line name in addition to DepMap ID
ccle_exp <- merge(ccle_exp, depmap_samples[, c("DepMap_ID", "stripped_cell_line_name", "primary_disease")], by = "DepMap_ID")
ccle_exp[, DepMap_ID := NULL]
ccle_exp[1:5, 1:20]

# Move cell line name to the first column: just giving the column name to the function moves it to first place
setcolorder(ccle_exp, neworder = sort(colnames(ccle_exp)))
setcolorder(ccle_exp, neworder = c("stripped_cell_line_name", "primary_disease"))
ccle_exp[1:5, 1:20]

# Save
fwrite(ccle_exp, paste0(path, "Data/DRP_Training_Data/DepMap_21Q2_Expression.csv"), sep = ',')

ccle_exp <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_21Q2_Expression.csv"))

ccle_exp
# DIMENSIONS OF EXPRESSION DATA: 1375 X 19178
rm(ccle_exp)
# ==== Copy number data cleanup ====
ccle_cn <- fread(paste0(path, "Data/DepMap/21Q2/CCLE_gene_copy_number.csv"))
dim(ccle_cn)
ccle_cn[1:5, 1:10]
# Change column names to only contain HGNC name: replace everything after first word with ""
colnames(ccle_cn) <- gsub(" .+", "", colnames(ccle_cn))
colnames(ccle_cn)[1] <- "DepMap_ID"
# Merge with sample info to have cell line name in addition to DepMap ID
ccle_cn <- merge(ccle_cn, depmap_samples[, c("DepMap_ID", "stripped_cell_line_name", "primary_disease")], by = "DepMap_ID")
ccle_cn[, DepMap_ID := NULL]

setcolorder(ccle_cn, neworder = sort(colnames(ccle_cn)))
setcolorder(ccle_cn, neworder = c("stripped_cell_line_name", "primary_disease"))
ccle_cn[1:5, 1:20]
which(is.na(ccle_cn), arr.ind = T)
ccle_cn[1407, 26569]

# Replace NA with 0
setnafill(ccle_cn, fill = 0, cols = unique(which(is.na(ccle_cn), arr.ind = T)[,2]))

dim(ccle_cn)
anyNA(ccle_cn)
sum(is.na(ccle_cn))
which(is.na(ccle_cn))
# Save
fwrite(ccle_cn, paste0(path, "Data/DRP_Training_Data/DepMap_21Q2_CopyNumber.csv"), sep = ',')

# DIMENSIONS OF COPY NUMBER DATA: 1740 X 27563
rm(ccle_cn)
gc()
# ==== Proteomic data cleanup ====
ccle_prot <- fread(paste0(path, "Data/DepMap/20Q2/CCLE_protein_quant_current_normalized.csv"))
dim(ccle_prot)
ccle_prot[1:5, 1:10]
ccle_prot[1:5, 48:60]
# Subset only the Uniprot accession (since its unique unlike HGNC) and the cell line experimental data
ccle_prot <- ccle_prot[, c(6, 49:ncol(ccle_prot)), with = F]
colnames(ccle_prot) <- gsub("\\_.+", "", colnames(ccle_prot))
colnames(ccle_prot)[1] <- "Uniprot_Acc"
# Transpose the data.table to match with other data type tables
t <- transpose(ccle_prot, make.names = "Uniprot_Acc")

# Check if transpose worked as intended
as.numeric(unlist(t[1,])) == as.numeric(unlist(ccle_prot[,2]))
as.numeric(unlist(t[2,])) == as.numeric(unlist(ccle_prot[,3]))

# Add cell lines
t$stripped_cell_line_name <- colnames(ccle_prot)[-1]
# Merge with sample info to have cell line name in addition to DepMap ID
t <- merge(t, depmap_samples[, c("DepMap_ID", "stripped_cell_line_name", "primary_disease")], by = "stripped_cell_line_name")

# Move to front
setcolorder(t, neworder = c("DepMap_ID", "stripped_cell_line_name", "primary_disease"))
t[1:5, 1:10]

# Save
fwrite(t, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_ProteinQuant.csv"), sep = ',')


### Get proteins that are observed in all cell lines
# Create the same transposed table as above
# Remove all rows and columns that have any NA in them
prot_nona <- na.omit(ccle_prot)
which(is.na(prot_nona))
# Transpose the data.table to match with other data type tables
t <- transpose(prot_nona, make.names = "Uniprot_Acc")
# Check if transpose worked as intended
as.numeric(unlist(t[1,])) == as.numeric(unlist(prot_nona[,2]))
as.numeric(unlist(t[2,])) == as.numeric(unlist(prot_nona[,3]))
# Add cell lines
t$stripped_cell_line_name <- colnames(prot_nona)[-1]
# Merge with sample info to have cell line name in addition to DepMap ID
t <- merge(t, depmap_samples[, c("DepMap_ID", "stripped_cell_line_name", "primary_disease")], by = "stripped_cell_line_name")
# Move to front
setcolorder(t, neworder = c("DepMap_ID", "stripped_cell_line_name", "primary_disease"))
t[1:5, 1:10]
# Now we have ~5000 proteins that are available in all samples
dim(t)

# We have 3 duplicates
sum(duplicated(t$stripped_cell_line_name))

# Save
fwrite(t, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv"), sep = ',')
# ccle_prot <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv"))
dim(ccle_prot)
ccle_prot[1:5, 1:5]
ccle_prot[, DepMap_ID := NULL]
# fwrite(ccle_prot, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv"), sep = ',')
# DIMENSIONS OF PROTEIN QUANTITY DATA: 378 X 5155

ccle_prot <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv"))
anyDuplicated(ccle_prot$stripped_cell_line_name)

# ==== Mutation data cleanup ====
rm(list = ls(pattern = "ccle"))
require(data.table)
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/"

ccle_mut <- fread(paste0(path, "Data/DepMap/21Q2/CCLE_mutations.csv"))
table(ccle_mut$isCOSMIChotspot)
table(ccle_mut$isTCGAhotspot)
table(ccle_mut$Variant_Type)
length(unique(ccle_mut$DepMap_ID))

dim(ccle_mut)
ccle_mut[1,]
colnames(ccle_mut)
# Calculate number of mutations per cell line
temp <- ccle_mut[, c("Variant_Type", "DepMap_ID")]
temp[, nMut := .N, by = "DepMap_ID"]
temp
unique(temp$Variant_Type)
# For simplicity, extract only SNP data for now: this discards ~90,000 mutations
# ccle_mut <- ccle_mut[Variant_Type == "SNP"]
dim(ccle_mut)
t <- ccle_mut[, c("DepMap_ID", "Chromosome", "Strand", "Start_position", "End_position")]
dim(unique(t))
length(unique(ccle_mut$DepMap_ID))
# Keep relevant columns/features
# Aside: Should the sequence change be provided, or just whether the SNP is deleterious or not?
ccle_mut <- ccle_mut[, c("DepMap_ID", "Hugo_Symbol", "Chromosome", "Start_position", "End_position", "Strand",
             "Variant_Classification", "Variant_Type", "isDeleterious",
             "isTCGAhotspot", "isCOSMIChotspot", "Genome_Change", "cDNA_Change")]
dim(ccle_mut)
length(unique(ccle_mut$DepMap_ID))
table(ccle_mut$isDeleterious)
table(ccle_mut$isTCGAhotspot)
table(ccle_mut$isCOSMIChotspot)

# ==== CCLE Mut Overlap with COSMIC CGC ====
# Perhaps it's best to use the mutations in genes that COSMIC considers important, like another paper in
# the field (~500 genes)
# Or, we can use a binary vector for genes and whether they have a deleterious mutation: this will result in 
# ~20,000 parameters
length(unique(ccle_mut$Hugo_Symbol))

length(unique(ccle_mut[isCOSMIChotspot == T]$Hugo_Symbol))
length(unique(ccle_mut[isTCGAhotspot == T]$Hugo_Symbol))
length(unique(ccle_mut[isDeleterious == T]$Hugo_Symbol))

tcga_hotspot_genes <- unique(ccle_mut[isTCGAhotspot == T]$Hugo_Symbol)
# Read COSMIC Cancer Gene Census data
cgc <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/COSMIC/cancer_gene_census.csv")
dim(cgc)
cgc[1:5, 1:20]
length(unique(cgc$`Gene Symbol`))
length(unique(cgc$HGVSG))
# Get Genes in this census
cgc_genes <- unique(cgc$`Gene Symbol`)
cgc[Tier == 1]
length(unique(cgc$`Genome Location`))  # 922,732
# rm(cgc)

# Subset DepMap mutations based on the CGC genes
sum(unique(ccle_mut$Hugo_Symbol) %in% unique(cgc_genes))
ccle_mut <- ccle_mut[Hugo_Symbol %in% cgc_genes]
length(unique(ccle_mut$DepMap_ID))

sum(ccle_mut$isDeleterious)
ccle_mut[Variant_Classification == "Missense_Mutation"]
length(unique(ccle_mut[isDeleterious == T]$Hugo_Symbol))
ccle_mut[isDeleterious == T]


# TODO: Use CGC to check for overlap with CCLE cell lines, then collapse to whether each of the 700 genes for
# that cell line has a mutation listed in the CGC
length(unique(cgc$`Mutation genome position`))  # ~922,000 unique mutations
unique(ccle_mut$NCBI_Build)  # CCLE is with GRCh 37
unique(cgc$GRCh)  # CGC has GRCh 38
# We must "lift over" the mutations from 37 to 38 before checking for overlap
if (!require(liftOver)) {
    BiocManager::install("liftOver")
    require(liftOver)
    require(rtracklayer)
}
# liftOver requires a chain file to convert 37 to 38: http://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/

chain_path <- paste0(path, "Data/hg19ToHg38.over.chain")
grch_37_38_chain <- import.chain(chain_path)

# Must add "chr" to start of chromosome names
ccle_mut$Chromosome <- paste0("chr", ccle_mut$Chromosome)
# Must convert positions to GRanges
ccle_mut_gr <- makeGRangesFromDataFrame(df = ccle_mut, keep.extra.columns = T,
                         seqnames.field = "Chromosome", start.field = "Start_position",
                         end.field = "End_position", strand.field = "Strand")
length(unique(ccle_mut_gr$DepMap_ID))

# Lift over
lifted_ccle_mut <- liftOver(x = ccle_mut_gr, chain = grch_37_38_chain)
# Convert GRangesList to GRanges
lifted_ccle_mut <- unlist(lifted_ccle_mut)
# Convert back to data.table
lifted_ccle_mut <- as.data.table(lifted_ccle_mut)
# Note: Genome_Change is now out of date!
# Remove chr from seqnames
lifted_ccle_mut$seqnames <- gsub("chr", "", lifted_ccle_mut$seqnames)
# Can find the overlap of Mutation genome position in CGC with a newly created column based on CCLE positions
lifted_ccle_mut[, Mutation_Position := paste0(seqnames, ':', start, '-', end)]

ccle_mut$seqnames <- gsub("chr", "", ccle_mut$Chromosome)
ccle_mut[, Mutation_Position := paste0(seqnames, ':', as.character(Start_position), '-', as.character(End_position))]


length(unique(lifted_ccle_mut$DepMap_ID))

sum(ccle_mut$Mutation_Position %in% unique(cgc$`Genome Location`))

# Now find the overlap with CGC (which already has GRCh38)
subset <- lifted_ccle_mut[Mutation_Position %in% unique(cgc$`Genome Location`)]
table(subset$Variant_Type)
length(unique(subset$DepMap_ID))
# IMPORTANT! There is a loss of 8 cell lines (which do not have a mutation that is in
# CGC) using the Tier 1 data only

# Alternative (March 2021) ====
# Take those mutations that are COSMIC or TCGA hotspots, ignoring CGC
subset <- ccle_mut[isTCGAhotspot | isCOSMIChotspot]

### Create a vector of mutations for each cell line with the CGC genes
length(unique(subset$Hugo_Symbol))
sub_dcast <- dcast.data.table(data = subset[, c("DepMap_ID", "Hugo_Symbol")],
                 formula = DepMap_ID ~ Hugo_Symbol, fun.aggregate = length, value.var = "DepMap_ID")
dim(sub_dcast)
sub_dcast[1:5, 1:50]
sum(sub_dcast$A1BG)
sum(sub_dcast$A1CF)

depmap_samples <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv"))
sub_dcast <- merge(sub_dcast, depmap_samples[, c("DepMap_ID", "stripped_cell_line_name", "primary_disease")],
                  by = "DepMap_ID")
setcolorder(sub_dcast, c("DepMap_ID", "stripped_cell_line_name", "primary_disease"))
sub_dcast[1:5, 1:50]

# Save
fwrite(sub_dcast, paste0(path, "Data/DRP_Training_Data/DepMap_21Q2_Mutations_by_Cell.csv"), sep = ',')
dim(cgc_muts)
cgc_muts[1:5, 1:5]
typeof(cgc_muts[1,2])

temp <- fread("Data/DRP_Training_Data/DepMap_21Q2_Mutations_by_Cell.csv")
dim(temp)
temp[1:5, 1:50]

# # Attach the cell line name and primary disease
# # cgc_muts <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_CGC_Mutations_by_Cell.csv"))
# depmap_samples <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv"))
# cgc_muts <- merge(cgc_muts, depmap_samples[, c("DepMap_ID", "stripped_cell_line_name", "primary_disease")],
#                   by = "DepMap_ID")
# setcolorder(cgc_muts, neworder = c("stripped_cell_line_name", colnames(cgc_muts)[-ncol(cgc_muts)]))
# 
# # Save
# fwrite(cgc_muts, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_CGC_Mutations_by_Cell.csv"), sep = ',')
# cgc_muts <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_CGC_Mutations_by_Cell.csv"))
# cgc_muts[1:5, 1:5]
# cgc_muts[, DepMap_ID := NULL]
# DIMENSIONS OF CGC MUTATIONAL DATA: 1733 X 697



# ==== miRNA Data Cleanup ====
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/"
require(data.table)
depmap_samples <- fread("Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv")

ccle_mirna <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DepMap/Extra/CCLE_miRNA_20181103.gct")
dim(ccle_mirna)
anyNA(ccle_mirna)
ccle_mirna[1:5, 1:5]

min(ccle_mirna[, -c(1:2)], na.rm = T)
max(ccle_mirna[, -c(1:2)], na.rm = T)

ccle_mirna <- transpose(ccle_mirna, keep.names = "Name")
dim(ccle_mirna)
ccle_mirna[1:5, 1:5]
ccle_mirna$Name
sum(duplicated(unlist(ccle_mirna[2, ])))
ccle_mirna <- ccle_mirna[-1,]
ccle_mirna[1:5, 1:5]
colnames(ccle_mirna) <- unlist(ccle_mirna[1,])
ccle_mirna <- ccle_mirna[-1,]

# Clean cell line name
ccle_mirna$Description <- gsub(pattern = "\\_.+", replacement = "", ccle_mirna$Description)
colnames(ccle_mirna)[1] <- "stripped_cell_line_name"
ccle_mirna <- merge(ccle_mirna, depmap_samples[, c("stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype")],
                    by = "stripped_cell_line_name")
dim(ccle_mirna)
ccle_mirna[1:5, 1:5]
setcolorder(ccle_mirna, c("stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype"))

fwrite(ccle_mirna, paste0(path, "Data/DRP_Training_Data/DepMap_2019_miRNA.csv"), sep = ',')

rm(ccle_mirna)


# ==== Metabolomics Data Cleanup ====
depmap_samples <- fread("Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv")
ccle_metab <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DepMap/Extra/CCLE_metabolomics_20190502.csv")

dim(ccle_metab)
ccle_metab[1:5, 1:5]

min(ccle_metab[, -c(1:2)], na.rm = T)
max(ccle_metab[, -c(1:2)], na.rm = T)

anyNA(ccle_metab)
sum(is.na(ccle_metab))
which(is.na(ccle_metab), arr.ind = T)
ccle_metab[which(is.na(ccle_metab), arr.ind = T)]
ccle_metab[554, 2]  # DepMap_ID is NA

min(ccle_metab[, -c(1:2)])
ccle_metab <- merge(ccle_metab[, -1], depmap_samples[, c("DepMap_ID", "stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype")],
                    by = "DepMap_ID")

setcolorder(ccle_metab, c("stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype"))
ccle_metab$DepMap_ID <- NULL
dim(ccle_metab)

fwrite(ccle_metab, paste0(path, "Data/DRP_Training_Data/DepMap_2019_Metabolomics.csv"), sep = ',')
rm(ccle_metab)
# ==== RPPA Data Cleanup ====
ccle_rppa <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DepMap/Extra/CCLE_RPPA_20181003.csv")
dim(ccle_rppa)
ccle_rppa[1:5, 1:5]
anyNA(ccle_rppa)

min(ccle_rppa[, -c(1:2)], na.rm = T)  # has negative values
max(ccle_rppa[, -c(1:2)], na.rm = T)

ccle_rppa$V1 <- gsub(pattern = "\\_.+", replacement = "", ccle_rppa$V1)
colnames(ccle_rppa)[1] <- "stripped_cell_line_name"
ccle_rppa <- merge(ccle_rppa, depmap_samples[, c("stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype")],
                    by = "stripped_cell_line_name")
dim(ccle_rppa)
ccle_rppa[1:5, 1:10]
setcolorder(ccle_rppa, c("stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype"))

fwrite(ccle_rppa, paste0(path, "Data/DRP_Training_Data/DepMap_2019_RPPA.csv"), sep = ',')

rm(ccle_rppa)

# ==== Chromatin Profiling Data Cleanup ====
ccle_chrom <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DepMap/Extra/CCLE_GlobalChromatinProfiling_20181130.csv")
dim(ccle_chrom)
ccle_chrom[1:5, 1:10]

min(ccle_chrom[, -c(1:2)], na.rm = T)  # has negative values
max(ccle_chrom[, -c(1:2)], na.rm = T)


anyNA(ccle_chrom)
sum(is.na(ccle_chrom))  # 842 NA values

unique(which(is.na(ccle_chrom), arr.ind = T)[,2])
length(unique(which(is.na(ccle_chrom), arr.ind = T)[,2]))  # 26 columns have NAs

# Convert NA to 0
setnafill(ccle_chrom, fill = 0, cols = unique(which(is.na(ccle_chrom), arr.ind = T)[,2]))
anyNA(ccle_chrom)

ccle_chrom$CellLineName <- gsub(pattern = "\\_.+", replacement = "", ccle_chrom$CellLineName)
colnames(ccle_chrom)[1] <- "stripped_cell_line_name"
dim(ccle_chrom)
ccle_chrom <- merge(ccle_chrom, depmap_samples[, c("stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype")],
                   by = "stripped_cell_line_name")
ccle_chrom$BroadID <- NULL
dim(ccle_chrom)
ccle_chrom[1:5, 1:10]
setcolorder(ccle_chrom, c("stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype"))

fwrite(ccle_chrom, paste0(path, "Data/DRP_Training_Data/DepMap_2019_ChromatinProfiling.csv"), sep = ',')

rm(ccle_chrom)

# ==== Fusion Data Cleanup ====
ccle_fusion <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DepMap/Extra/CCLE_fusions.csv")
dim(ccle_fusion)
ccle_fusion[1:5, 1:17]
length(unique(ccle_fusion$FusionName))
length(unique(ccle_fusion$DepMap_ID))
unique(ccle_fusion$SpliceType)
quantile(ccle_fusion$FFPM)

ccle_fusion$CellLineName <- gsub(pattern = "\\_.+", replacement = "", ccle_fusion$CellLineName)
colnames(ccle_fusion)[1] <- "stripped_cell_line_name"
dim(ccle_fusion)
ccle_fusion <- merge(ccle_fusion, depmap_samples[, c("stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype")],
                    by = "stripped_cell_line_name")
ccle_fusion$BroadID <- NULL
dim(ccle_fusion)
ccle_fusion[1:5, 1:10]
setcolorder(ccle_fusion, c("stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype"))

fwrite(ccle_fusion, paste0(path, "Data/DRP_Training_Data/DepMap_2019_GeneFusion.csv"), sep = ',')

rm(ccle_fusion)

# ==== Exon Usage Ratio Data Cleanup ====
require(data.table)
setDTthreads(8)
ccle_exon <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DepMap/Extra/CCLE_RNAseq_ExonUsageRatio_20180929.gct")
dim(ccle_exon)
ccle_exon[1:10, 1:17]

transpose(ccle_exon, keep.names = "exon")
length(unique(ccle_exon$FusionName))
length(unique(ccle_exon$DepMap_ID))
unique(ccle_exon$SpliceType)
quantile(ccle_exon$FFPM)

ccle_exon$CellLineName <- gsub(pattern = "\\_.+", replacement = "", ccle_exon$CellLineName)
colnames(ccle_exon)[1] <- "stripped_cell_line_name"
dim(ccle_exon)
ccle_exon <- merge(ccle_exon, depmap_samples[, c("stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype")],
                     by = "stripped_cell_line_name")
ccle_exon$BroadID <- NULL
dim(ccle_exon)
ccle_exon[1:5, 1:10]
setcolorder(ccle_exon, c("stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype"))

# fwrite(ccle_exon, paste0(path, "Data/DRP_Training_Data/DepMap_2019_ExonUsageRatio.csv"), sep = ',')

rm(ccle_exon)

# ==== RRBS Profiling Data Cleanup ====
require(data.table)
setDTthreads(8)

# === TSS
ccle_tss <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DepMap/Extra/CCLE_RRBS_TSS1kb_20181022.txt")
dim(ccle_tss)
ccle_tss[1:5, 1:5]
length(unique(ccle_tss$cluster_id))

ccle_tss <- transpose(ccle_tss[, -2], keep.names = "cluster_id")
colnames(ccle_tss) <- unlist(ccle_tss[1,])
ccle_tss <- ccle_tss[-1,]

# === Promoter
ccle_tss <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DepMap/Extra/CCLE")
dim(ccle_tss)
ccle_tss[1:5, 1:5]
length(unique(ccle_tss$cluster_id))

ccle_tss <- transpose(ccle_tss[, -2], keep.names = "cluster_id")
colnames(ccle_tss) <- unlist(ccle_tss[1,])
ccle_tss <- ccle_tss[-1,]
# === Enhancers


# ==== GDSC Cell Line Characterization Data Cleanup ====
require(data.table)
gdsc_line_info <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/GDSC/GDSC_Line_Info.csv")
gdsc_line_info <- gdsc_line_info[-c(1,2,3, 1005), c("V1", "V2", "V9")]
colnames(gdsc_line_info) <- c("stripped_cell_line_name", "COSMIC_ID", "primary_disease")
gdsc_line_info$stripped_cell_line_name <- gsub("[^[:alnum:] ]", "", gdsc_line_info$stripped_cell_line_name)
gdsc_line_info$stripped_cell_line_name <- gsub(" ", "", gdsc_line_info$stripped_cell_line_name)
gdsc_line_info$stripped_cell_line_name <- toupper(gdsc_line_info$stripped_cell_line_name)

# Match with DepMap cell lines
depmap_line_info <- fread("Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv")
sum(gdsc_line_info$stripped_cell_line_name %in% depmap_line_info$stripped_cell_line_name) / nrow(gdsc_line_info)  # 98%
gdsc_line_info[!(gdsc_line_info$stripped_cell_line_name %in% depmap_line_info$stripped_cell_line_name)]
gdsc_line_info[stripped_cell_line_name == "EOL1CELL"] <- "EOL1"
gdsc_line_info[stripped_cell_line_name == "SR"] <- "SR786"
gdsc_line_info[stripped_cell_line_name == "U266"] <- "U266B1"
gdsc_line_info[stripped_cell_line_name == "G292"] <- "G292CLONEA141B1"
gdsc_line_info[stripped_cell_line_name == "NCISNU1"] <- "SNU1"
gdsc_line_info[stripped_cell_line_name == "NCISNU5"] <- "SNU5"
gdsc_line_info[stripped_cell_line_name == "NCISNU16"] <- "SNU16"
gdsc_line_info[stripped_cell_line_name == "7860"] <- "786O"
gdsc_line_info[stripped_cell_line_name == "U031"] <- "UO31"
gdsc_line_info[stripped_cell_line_name == "H3255"] <- "NCIH3255"
gdsc_line_info[stripped_cell_line_name == "NCIH510A"] <- "NCIH510"
gdsc_line_info[stripped_cell_line_name == "U251"] <- "U251MG"
gdsc_line_info[stripped_cell_line_name == "WM793B"] <- "WM793"
gdsc_line_info[stripped_cell_line_name == "OVCAR3"] <- "NIHOVCAR3"
gdsc_line_info[stripped_cell_line_name == "SCC90"] <- "UPCISCC090"
gdsc_line_info[COSMIC_ID == "1299064"] <- "TDOTT"
gdsc_line_info[COSMIC_ID == "930299"] <- "TT"
gdsc_line_info[COSMIC_ID == "909976"] <- "KMH2"
gdsc_line_info[COSMIC_ID == "1298167"] <- "KMHDASH2"
sum(gdsc_line_info$stripped_cell_line_name %in% depmap_line_info$stripped_cell_line_name) / nrow(gdsc_line_info)  # 100%


# ==== GDSC Expression Data Cleanup ====
gdsc_exp <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/GDSC/GDSC_Expression.txt")
gdsc_exp[1:5, 1:5]
# Delete invalid gene symbol rows
gdsc_exp <- gdsc_exp[GENE_SYMBOLS != ""]

# Rename Expression columns
colnames(gdsc_exp) <- gsub("DATA.", "", colnames(gdsc_exp))
gdsc_exp <- gdsc_exp[, !"GENE_title", with = F]
# Must transpose to have cell lines on each row
gdsc_exp <- transpose(gdsc_exp, keep.names = "GENE_SYMBOLS")
gdsc_exp[1:5, 1:5]
colnames(gdsc_exp) <- unlist(gdsc_exp[1,])
gdsc_exp <- gdsc_exp[-1,]
colnames(gdsc_exp)[1] <- "COSMIC_ID"
# gdsc_exp[COSMIC_ID == "1299064"][1:5, 1:5]
# gdsc_exp[COSMIC_ID == "930299"][1:5, 1:5]
# Merge with cell line names
gdsc_exp <- merge(gdsc_exp, gdsc_line_info, by = "COSMIC_ID")
gdsc_exp$COSMIC_ID <- NULL
setcolorder(gdsc_exp, order(colnames(gdsc_exp)))
setcolorder(gdsc_exp, c("stripped_cell_line_name", "primary_disease"))
gdsc_exp[1:5, 1:5]
# Match genes with those from DepMap Cell lines
depmap_exp <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/DepMap_21Q2_Expression.csv")
# depmap_exp <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DRP_Training_Data/DepMap_21Q2_Training_Expression.csv")
depmap_exp[1:5, 1:5]

dim(gdsc_exp)
dim(depmap_exp)

t1 <- colnames(gdsc_exp)
t2 <- colnames(depmap_exp)
sum(t1 %in% t2) / ncol(gdsc_exp)  # 91%
colnames(gdsc_exp)[!colnames(gdsc_exp) %in% colnames(depmap_exp)]

# Update gene names with biomaRt
# require(biomaRt)
# ensembl = useMart(
#   biomart = "ENSEMBL_MART_ENSEMBL",
#   # host = "feb2014.archive.ensembl.org",
#   path = "/biomart/martservice",
#   dataset = "hsapiens_gene_ensembl"
# )
# atts <- listAttributes(ensembl, what = "name")
# atts[grep(pattern = "gene", x = atts, ignore.case = T)]
# depmap_dict <- biomaRt::getBM(attributes = c("ensembl_gene_id",
#                                           "external_gene_name", "wikigene_name"),
#                            filters = "external_gene_name", values = colnames(depmap_exp),
#                            mart = ensembl)
# depmap_dict <- as.data.table(depmap_dict)
# depmap_dict <- unique(depmap_dict[, 2:3])
# 
# gdsc_dict <- biomaRt::getBM(attributes = c("ensembl_gene_id",
#                                              "external_gene_name", "wikigene_name"),
#                               filters = "external_gene_name", values = colnames(gdsc_exp),
#                               mart = ensembl)
# gdsc_dict <- as.data.table(gdsc_dict)

# Get updated gene names from: https://www.genenames.org/tools/multi-symbol-checker/

all_gene_names <- data.table(unique(c(colnames(depmap_exp), colnames(gdsc_exp)))[-(1:2)])
fwrite(all_gene_names, "GDSC_DepMap_All_Gene_Names.txt")
hgnc_check <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/hgnc-symbol-check.csv", skip = 1)

depmap_hgnc_check <- hgnc_check[Input %in% colnames(depmap_exp)[-(1:2)]]
names(depmap_exp)[match(depmap_hgnc_check$Input, names(depmap_exp))] <- depmap_hgnc_check$`Approved symbol`
gdsc_hgnc_check <- hgnc_check[Input %in% colnames(gdsc_exp)[-(1:2)]]
names(gdsc_exp)[match(gdsc_hgnc_check$Input, names(gdsc_exp))] <- gdsc_hgnc_check$`Approved symbol`


t1 <- colnames(gdsc_exp)
t2 <- colnames(depmap_exp)
sum(t1 %in% t2) / ncol(gdsc_exp)  # 97.3%
sum(t2 %in% t1) / ncol(depmap_exp)  # 88.6%
sum(t2 %in% t1)  # 16988 
sum(t1 %in% t2)  # 16948 
colnames(gdsc_exp)[!colnames(gdsc_exp) %in% colnames(depmap_exp)]

fwrite(gdsc_exp, "Data/DRP_Training_Data/GDSC_Expression.csv")

# Subset both datasets's genes based on data available to both
shared <- intersect(colnames(gdsc_exp), colnames(depmap_exp))
length(shared)
gdsc_exp_sub <- gdsc_exp[, shared, with = F]
gdsc_exp_sub[1:5, 1:5]

depmap_exp_sub <- depmap_exp[, shared, with = F]
depmap_exp_sub[1:5, 1:5]

# Save
fwrite(gdsc_exp_sub, "Data/DRP_Training_Data/GDSC_Training_Expression.csv")
fwrite(depmap_exp_sub, "Data/DRP_Training_Data/DepMap_21Q2_Training_Expression.csv")

# ==== Drug Sensitivity Data Cleanup ====
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/"
require(data.table)
require(webchem)
# BiocManager::install("ChemmineR")
require(ChemmineR)
options(chemspider_key = "N98K4aOip0VpcSc8F9GilqIIktLt0hux")
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/"
ctrp <- fread("Data/DRP_Training_Data/CTRP_AUC_SMILES.txt")
gdsc1 <- fread("Data/DRP_Training_Data/GDSC1_AUC_SMILES.txt")
gdsc2 <- fread("Data/DRP_Training_Data/GDSC2_AUC_SMILES.txt")


# Clean up duplicate with missing pubchem
cpd_info_1 <- fread(paste0(path, "Data/GDSC/GDSC1_Drug_Info.csv"))
cpd_info_1[drug_name == unique(cpd_info_1[, c("drug_name", "pubchem")])[anyDuplicated(unique(cpd_info_1[, c("drug_name", "pubchem")])$drug_name),]$drug_name]
cpd_info_1 <- cpd_info_1[drug_id != 476]
cpd_info_1 <- cpd_info_1[drug_id != 1490]
cpd_info_1 <- cpd_info_1[drug_id != 1496]
cpd_info_1 <- cpd_info_1[drug_id != 1386]
cpd_info_1 <- cpd_info_1[drug_id != 1402]
cpd_info_1 <- cpd_info_1[drug_id != 1393]
nrow(cpd_info_1[pubchem == "-"])
sum(cpd_info_1$drug_name %in% unique(ctrp$cpd_name))
# Subset for valid pubchem IDs
cpd_info_1 <- cpd_info_1[pubchem != "-"]
cpd_info_1 <- cpd_info_1[pubchem != "none"]
cpd_info_1 <- cpd_info_1[pubchem != "several"]
cpd_info_1$pubchem <- as.numeric(cpd_info_1$pubchem)

cpd_1_smiles <- webchem::pc_prop(cid = cpd_info_1$pubchem, properties = "CanonicalSMILES")
cpd_info_1 <- merge(cpd_info_1, cpd_1_smiles, by.x = "pubchem", by.y = "CID")
# Save
fwrite(cpd_info_1, "Data/GDSC/GDSC1_VALID_Drug_Info.csv")


cpd_info_2 <- fread(paste0(path, "Data/GDSC/GDSC2_Drug_Info.csv"))
cpd_info_2[drug_name == unique(cpd_info_2[, c("drug_name", "pubchem")])[anyDuplicated(unique(cpd_info_2[, c("drug_name", "pubchem")])$drug_name),]$drug_name]
cpd_info_2 <- cpd_info_2[drug_id != 1811]
cpd_info_2 <- cpd_info_2[drug_id != 1806]
cpd_info_2 <- cpd_info_2[drug_id != 1819]
cpd_info_2 <- cpd_info_2[drug_id != 1816]
cpd_info_2[pubchem == "25227436, 42602260"]$pubchem <- "25227436"
cpd_info_2[pubchem == "11719003, 58641927"]$pubchem <- "11719003"
cpd_info_2[pubchem == "66577015, 16654980"]$pubchem <- "66577015"
cpd_info_2[pubchem == "11719003, 58641927"]$pubchem <- "11719003"

nrow(cpd_info_2[pubchem == "-"])
sum(cpd_info_2$pubchem %in% cpd_info_1$pubchem) / nrow(cpd_info_2)

cpd_info_2 <- cpd_info_2[pubchem != "-"]
cpd_info_2 <- cpd_info_2[pubchem != "none"]
cpd_info_2 <- cpd_info_2[pubchem != "several"]

cpd_info_2$pubchem <- as.numeric(cpd_info_2$pubchem)

cpd_2_smiles <- webchem::pc_prop(cid = cpd_info_2$pubchem, properties = "CanonicalSMILES")
cpd_info_2 <- merge(cpd_info_2, cpd_2_smiles, by.x = "pubchem", by.y = "CID")
# Save
fwrite(cpd_info_2, "Data/GDSC/GDSC2_VALID_Drug_Info.csv")


depmap_samples <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv"))

# ==== GDSC ====
require(stringr)
gdsc1 <- fread(paste0(path, "Data/GDSC/GDSC1_Fitted_Dose_Response.csv"))
sum(unique(gdsc1$CELL_LINE_NAME) %in% depmap_samples$stripped_cell_line_name) / length(unique(gdsc1$CELL_LINE_NAME))  # 0.22
sum(toupper(unique(gdsc1$CELL_LINE_NAME)) %in% toupper(depmap_samples$stripped_cell_line_name)) / length(unique(gdsc1$CELL_LINE_NAME))  # 0.24
sum(str_remove_all(toupper(unique(gdsc1$CELL_LINE_NAME)), "-") %in% toupper(depmap_samples$stripped_cell_line_name)) / length(unique(gdsc1$CELL_LINE_NAME))  # 0.9696049

dim(gdsc1)  # 310K Combinations
colnames(gdsc1)
sum(gdsc1$AUC == 0)
min(gdsc1$AUC)
max(gdsc1$AUC)

# Count unique combinations in GDSC1
length(unique(unique(gdsc1[, c("DRUG_NAME", "CELL_LINE_NAME")])$CELL_LINE_NAME))  # 987
length(unique(unique(gdsc1[, c("DRUG_NAME", "CELL_LINE_NAME")])$DRUG_NAME))  # 345
nrow(unique(unique(gdsc1[, c("DRUG_NAME", "CELL_LINE_NAME")]))) # 292,849


gdsc1_final <- merge(unique(gdsc1[, c("DRUG_NAME", "CELL_LINE_NAME", "AUC")]), unique(cpd_info_1[, c("drug_name", "CanonicalSMILES")]), by.x = "DRUG_NAME", by.y = "drug_name")
colnames(gdsc1_final) <- c("cpd_name", "ccl_name", "area_under_curve", "cpd_smiles")
# Save
fwrite(gdsc1_final, "Data/DRP_Training_Data/GDSC1_AUC_SMILES.txt")

unique(gdsc1_pubchem$DRUG_NAME)
# gdsc1_cs_ids <- webchem::get_csid(query = unique(gdsc1$DRUG_NAME), from = "name", match = "all", verbose = T)
gdsc1_cs_ids <- webchem::cir_query(identifier = unique(gdsc1$DRUG_NAME), representation = "smiles", verbose = T, )

# Count unique combinations in GDSC2
gdsc2 <- fread(paste0(path, "Data/GDSC/GDSC2_Fitted_Dose_Response.csv"))
sum(unique(gdsc2$CELL_LINE_NAME) %in% depmap_samples$stripped_cell_line_name) / length(unique(gdsc2$CELL_LINE_NAME))  # 0.2311496
sum(toupper(unique(gdsc2$CELL_LINE_NAME)) %in% toupper(depmap_samples$stripped_cell_line_name)) / length(unique(gdsc2$CELL_LINE_NAME))  # 0.2546354
sum(str_remove_all(toupper(unique(gdsc2$CELL_LINE_NAME)), "-") %in% toupper(depmap_samples$stripped_cell_line_name)) / length(unique(gdsc2$CELL_LINE_NAME))  # 0.9678616

gdsc2_cpd_smiles <- webchem::cir_query(identifier = unique(gdsc2$DRUG_NAME), representation = "smiles", verbose = T)

dim(gdsc2)  # 135K Combinations
colnames(gdsc2)
length(unique(unique(gdsc2[, c("DRUG_NAME", "CELL_LINE_NAME")])$CELL_LINE_NAME))  # 809
length(unique(unique(gdsc2[, c("DRUG_NAME", "CELL_LINE_NAME")])$DRUG_NAME))  # 192
nrow(unique(unique(gdsc2[, c("DRUG_NAME", "CELL_LINE_NAME")]))) # 131,108

gdsc2_final <- merge(unique(gdsc2[, c("DRUG_NAME", "CELL_LINE_NAME", "AUC")]), unique(cpd_info_2[, c("drug_name", "CanonicalSMILES")]), by.x = "DRUG_NAME", by.y = "drug_name")
colnames(gdsc2_final) <- c("cpd_name", "ccl_name", "area_under_curve", "cpd_smiles")
# Save
fwrite(gdsc2_final, "Data/DRP_Training_Data/GDSC2_AUC_SMILES.txt")


# Count overlap of drugs and cell lines
sum(unique(gdsc1$DRUG_NAME) %in% unique(gdsc2$DRUG_NAME)) # Drug Overlap: 88
sum(unique(gdsc1$CELL_LINE_NAME) %in% unique(gdsc2$CELL_LINE_NAME))  # Cell Line Overlap: 808

# ==== CTRP ====
require(data.table)
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/"

# NOTE: Newer and better AUC calculation in PharmacoGx.R file!
ctrp_curves <- fread(paste0(path, "Data/CTRP/v20.data.curves_post_qc.txt"))
exper_data <- fread(paste0(path, "Data/CTRP/v20.meta.per_experiment.txt"))
cell_data <- fread(paste0(path, "Data/CTRP/v20.meta.per_cell_line.txt"))
table(cell_data$ccl_availability)

# Merge sensitivity, experimental and cell line data
temp <- merge(unique(ctrp_curves[, c("experiment_id", "master_cpd_id")]),
              unique(exper_data[, c("experiment_id", "master_ccl_id")]),
              by = "experiment_id")
ctrp <- merge(temp, cell_data[, c("master_ccl_id", "ccl_name")], by = "master_ccl_id")
sum(unique(ctrp$ccl_name) %in% depmap_samples$stripped_cell_line_name) / length(unique(ctrp$ccl_name))  # 0.9492672
sum(toupper(unique(ctrp$ccl_name)) %in% toupper(depmap_samples$stripped_cell_line_name)) / length(unique(ctrp$ccl_name))  # 0.9492672
sum(str_remove_all(toupper(unique(ctrp$ccl_name)), "-") %in% toupper(depmap_samples$stripped_cell_line_name)) / length(unique(ctrp$ccl_name))  # 0.9503946


# Add compound information
cpd_data <- fread(paste0(path, "Data/CTRP/v20.meta.per_compound.txt"))
ctrp <- merge(ctrp, cpd_data[, c("master_cpd_id", "cpd_name", "cpd_smiles")], by = "master_cpd_id")

# Add AUC curve information
ctrp_auc <- fread(paste0(path, "Data/CTRP/v20.data.curves_post_qc.txt"))

ctrp <- merge(ctrp, ctrp_auc[, c("experiment_id", "master_cpd_id", "area_under_curve")], by = c("experiment_id", "master_cpd_id"))


# Save
fwrite(ctrp, paste0(path, "Data/DRP_Training_Data/CTRP_AUC_SMILES.txt"))

# Add primary disease information. NOTE: This removes some DR data as 45 cell lines in CTRPv2 cannot be paired with DepMap!!!
line_info <- fread("Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv")
ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")

sum(unique(ctrp$ccl_name) %in% unique(line_info$stripped_cell_line_name))  # 150

line_info$other_ccl_name <- str_replace(toupper(line_info$stripped_cell_line_name), "-", "")
ctrp$other_ccl_name <- str_replace(toupper(ctrp$ccl_name), "-", "")

ctrp <- merge(ctrp, line_info[, c("other_ccl_name", "primary_disease")], by = "other_ccl_name")
ctrp$other_ccl_name <- NULL
setcolorder(ctrp, neworder = c("cpd_name", "ccl_name", "primary_disease", "area_under_curve", "cpd_smiles"))

fwrite(ctrp, "Data/DRP_Training_Data/CTRP_AUC_SMILES.txt")


# Experiment ID 
unique(ctrp[, c("master_ccl_id", "experiment_id")])
length(unique(ctrp$master_ccl_id))
length(unique(ctrp$experiment_id))
length(unique(ctrp$ccl_name))
length(unique(ctrp$master_cpd_id))

# Check overlap with GDSC 1 and 2
sum(unique(ctrp$ccl_name) %in% gdsc1$CELL_LINE_NAME)
sum(unique(ctrp$ccl_name) %in% gdsc2$CELL_LINE_NAME)

dim(ctrp)  # 395K Combinations



# ==== Chemical Data Cleanup ====
require(data.table)
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/"

chembl <- fread(paste0(path, "Data/chembl_27_chemreps.txt"))



# ==== EDA ======
require(data.table)
require(stringr)
require(ggplot2)
line_info <- fread("Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv")
ctrp <- fread("Data/DRP_Training_Data/CTRP_AUC_SMILES.txt")
gdsc2 <- fread("Data/DRP_Training_Data/GDSC2_AUC_SMILES.txt")
exp <- fread("Data/DRP_Training_Data/DepMap_20Q2_Expression.csv")
mut <- fread("Data/DRP_Training_Data/DepMap_20Q2_CGC_Mutations_by_Cell.csv")
cnv <- fread("Data/DRP_Training_Data/DepMap_20Q2_CopyNumber.csv")
prot <- fread("Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv")
pdb_table <- fread("Data/cell_annotation_table_1.1.1.csv")
pdb_sub <- pdb_table[, c("CTRPv2.cellid", "CCLE.cellid")]
pdb_sub <- pdb_sub[!is.na(CTRPv2.cellid) & !is.na(CCLE.cellid)]


exp[1:5., 1:5]

length(unique(ctrp$ccl_name))
length(unique(ctrp$cpd_name))

sum(unique(ctrp$ccl_name) %in% line_info$stripped_cell_line_name) / length(unique(ctrp$ccl_name))
ccl_names = toupper(ctrp$ccl_name)
ccl_names = unique(str_replace(ccl_names, "-", ""))
length(ccl_names)

sum(ccl_names %in% line_info$stripped_cell_line_name) / length(ccl_names)

line_info[!(stripped_cell_line_name %in% ccl_names)]
ctrp[ccl_name %like% "NIHOVCAR3"]
ctrp[ccl_name %like% "HEL"]

sum(exp$stripped_cell_line_name %in% pdb_sub$CCLE.cellid) 

# Remove hyphens and convert all to upper case
pdb_ccl_names = pdb_sub$CCLE.cellid
pdb_ccl_names = str_replace(toupper(pdb_ccl_names), "-", "")

ctrp$ccl_name = str_replace(toupper(ctrp$ccl_name), "-", "")
     
exp_ccl_names = exp$stripped_cell_line_name
exp_ccl_names = str_replace(toupper(exp_ccl_names), "-", "")

mut_ccl_names = mut$stripped_cell_line_name
mut_ccl_names = str_replace(toupper(mut_ccl_names), "-", "")

cnv_ccl_names = cnv$stripped_cell_line_name
cnv_ccl_names = str_replace(toupper(cnv_ccl_names), "-", "")

sum(exp_ccl_names %in% ccl_names) / length(unique(ccl_names))
sum(exp_ccl_names %in% pdb_ccl_names) / length(unique(pdb_ccl_names))


sum(mut_ccl_names %in% ccl_names) / length(unique(ccl_names)) * length(unique(ccl_names))
sum(mut_ccl_names %in% pdb_ccl_names) / length(unique(pdb_ccl_names)) * length(unique(pdb_ccl_names))

ctrp[ccl_name %in% mut_ccl_names[mut_ccl_names %in% ccl_names]]   ### 302K!!!!! Not 144K
ctrp[ccl_name %in% mut_ccl_names[mut_ccl_names %in% pdb_ccl_names]]   ### 302K!!!!! Not 144K

sum(cnv_ccl_names %in% ccl_names) / length(unique(ccl_names))
sum(exp_ccl_names %in% cnv_ccl_names) / length(unique(exp_ccl_names))
sum(cnv_ccl_names %in% exp_ccl_names) / length(unique(cnv_ccl_names))


dir.create(path = "Plots")
dir.create(path = "Plots/DepMap")
ggplot(data = line_info) +
  geom_bar(mapping = aes(x = primary_disease), stat = "count") +
  xlab("Primary Disease") +
  ylab("# of cell lines") + 
  ggtitle(label = "Proportion of Cancer Types in DepMap Data (overall)", subtitle = "20Q2 Version") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(filename = "Plots/DepMap/DepMap_Cell_Lines_Proportion.pdf", device = "pdf")
  
prot[, 1:5]
unique(prot$stripped_cell_line_name)

mut$stripped_cell_line_name = str_replace(toupper(mut$stripped_cell_line_name), "-", "")
cnv$stripped_cell_line_name = str_replace(toupper(cnv$stripped_cell_line_name), "-", "")
exp$stripped_cell_line_name = str_replace(toupper(exp$stripped_cell_line_name), "-", "")
prot$stripped_cell_line_name = str_replace(toupper(prot$stripped_cell_line_name), "-", "")
ctrp$ccl_name = str_replace(toupper(ctrp$ccl_name), "-", "")

mut_line_info <- line_info[stripped_cell_line_name %in% unique(mut$stripped_cell_line_name)]  
cnv_line_info <- line_info[stripped_cell_line_name %in% unique(cnv$stripped_cell_line_name)]  
exp_line_info <- line_info[stripped_cell_line_name %in% unique(exp$stripped_cell_line_name)]  
prot_line_info <- line_info[stripped_cell_line_name %in% unique(prot$stripped_cell_line_name)]
ctrp_line_info <- line_info[stripped_cell_line_name %in% unique(ctrp$ccl_name)]

mut_line_info <- mut_line_info[, c("stripped_cell_line_name", "primary_disease")]
mut_line_info$data_type <- "Mutational"

cnv_line_info <- cnv_line_info[, c("stripped_cell_line_name", "primary_disease")]
cnv_line_info$data_type <- "Copy Number"

exp_line_info <- exp_line_info[, c("stripped_cell_line_name", "primary_disease")]
exp_line_info$data_type <- "Gene Expression"

prot_line_info <- prot_line_info[, c("stripped_cell_line_name", "primary_disease")]
prot_line_info$data_type <- "Protein Quantification"

ctrp_line_info <- ctrp_line_info[, c("stripped_cell_line_name", "primary_disease")]
ctrp_line_info$data_type <- "Dose-Response"

datatype_line_info <- rbindlist(list(mut_line_info, cnv_line_info, exp_line_info, prot_line_info, ctrp_line_info))

ggplot(data = datatype_line_info) +
  geom_bar(mapping = aes(x = primary_disease, fill = data_type), stat = "count", position = "dodge") +
  xlab("Primary Disease") +
  ylab("# of cell lines") +
  labs(fill = "Data Type") +
  ggtitle(label = "Proportion of Cancer Types in DepMap Data", subtitle = "By data type, 20Q2 Version - Overlap with CTRPv2: 79%") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(filename = "Plots/DepMap/DepMap_CTRP_Cell_Lines_Proportion.pdf", device = "pdf")


BiocManager::install("VennDiagram")
require(VennDiagram)

library(RColorBrewer)
myCol <- brewer.pal(5, "Pastel2")

# NOTE: The CTRPv2 here is from before ctrp was merged with cell line info to add primary disease!
venn.diagram(x = list(mut_line_info$stripped_cell_line_name,
                      cnv_line_info$stripped_cell_line_name,
                      exp_line_info$stripped_cell_line_name,
                      prot_line_info$stripped_cell_line_name,
                      unique(ctrp$ccl_name)),
             category.names = c("Mutational", "Copy Number", "Gene Expression", "Protein Quantification", "CTRPv2 Dose-Response"),
             filename = "Plots/DepMap/DepMap_CTRP_Cell_Lines_Venn.png",
             imagetype = "png",
             output = TRUE,
             height = 3000 ,
             width = 3000 ,
             resolution = 600,
             # Circles
             lwd = 2,
             # lty = 'blank',
             fill = myCol,
             # Numbers
             cex = .6,
             fontface = "bold",
             fontfamily = "sans",
             
             # Set names
             cat.cex = 0.6,
             cat.fontface = "bold",
             cat.default.pos = "outer",
             cat.pos = c(0, 0, -130, 150, 0),
             cat.dist = c(0.2, 0.2, 0.2, 0.2, 0.2),
             cat.fontfamily = "sans",
             # rotation = 1
             
)
