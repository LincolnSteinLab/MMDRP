# Complete_Sample_Prep.R

# This script is intended to pair genomics, transcriptomics, proteomics and drug response data
# mainly from the DepMap resource.
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/"
dir.create(paste0(path, "Data/DRP_Training_Data"))

require(data.table)

# ==== Cell line info cleanup ====
depmap_samples <- fread(paste0(path, "Data/DepMap/20Q2/DepMap_Cell_Line_info-3.csv"))
# Subset relevant (machine learning) columns 
depmap_samples <- depmap_samples[, c("DepMap_ID", "stripped_cell_line_name", "primary_disease", "lineage", "lineage_subtype")]
fwrite(depmap_samples, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv"))

# depmap_samples <- fread("Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv")
# ==== Expression data cleanup ====
ccle_exp <- fread(paste0(path, "Data/DepMap/20Q2/CCLE_expression.csv"))
dim(ccle_exp)
ccle_exp[1:5, 1:20]
# Change column names to only contain HGNC name: replace everything after first word with ""
colnames(ccle_exp) <- gsub(" .+", "", colnames(ccle_exp))
colnames(ccle_exp)[1] <- "DepMap_ID"
# Merge with sample info to have cell line name in addition to DepMap ID
ccle_exp <- merge(ccle_exp, depmap_samples[, 1:2], by = "DepMap_ID")
# Move cell line name to the first column: just giving the column name to the function moves it to first place
setcolorder(ccle_exp, neworder = "stripped_cell_line_name")
ccle_exp[1:5, 1:20]

# Save
fwrite(ccle_exp, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_Expression.csv"), sep = ',')

# ccle_exp <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_Expression.csv"))
dim(ccle_exp)
ccle_exp[1:5, 1:5]
ccle_exp[, DepMap_ID := NULL]
ccle_exp[1:5, 1:5]
fwrite(ccle_exp, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_Expression.csv"), sep = ',')

# DIMENSIONS OF EXPRESSION DATA: 1303 X 19146

# ==== Copy number data cleanup ====
ccle_cn <- fread(paste0(path, "Data/DepMap/20Q2/CCLE_gene_copy_number.csv"))
dim(ccle_cn)
ccle_cn[1:5, 1:10]
# Change column names to only contain HGNC name: replace everything after first word with ""
colnames(ccle_cn) <- gsub(" .+", "", colnames(ccle_cn))
colnames(ccle_cn)[1] <- "DepMap_ID"
# Merge with sample info to have cell line name in addition to DepMap ID
ccle_cn <- merge(ccle_cn, depmap_samples[, 1:2], by = "DepMap_ID")
# Move cell line name to the first column: just giving the column name to the function moves it to first place
setcolorder(ccle_cn, neworder = "stripped_cell_line_name")
ccle_cn[1:5, 1:20]

# Save
fwrite(ccle_cn, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_CopyNumber.csv"), sep = ',')
# ccle_cn <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_CopyNumber.csv"))
dim(ccle_cn)
ccle_cn[1:5, 1:5]
ccle_cn[, DepMap_ID := NULL]

# DIMENSIONS OF COPY NUMBER DATA: 1742 X 27641

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
t <- merge(t, depmap_samples[, 1:2], by = "stripped_cell_line_name")

# Move to front
setcolorder(t, neworder = "DepMap_ID")
setcolorder(t, neworder = "stripped_cell_line_name")
t[1:5, 1:10]
# Reorder based on DepMap_ID
setorder(t, "DepMap_ID")
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
t <- merge(t, depmap_samples[, 1:2], by = "stripped_cell_line_name")
# Move to front
setcolorder(t, neworder = "DepMap_ID")
setcolorder(t, neworder = "stripped_cell_line_name")
t[1:5, 1:10]
# Reorder based on DepMap_ID
setorder(t, "DepMap_ID")
t[1:5, 1:10]
# Now we have ~5000 proteins that are available in all samples
dim(t)

# Save
fwrite(t, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv"), sep = ',')
# ccle_prot <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv"))
dim(ccle_prot)
ccle_prot[1:5, 1:5]
ccle_prot[, DepMap_ID := NULL]
fwrite(ccle_prot, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv"), sep = ',')
# DIMENSIONS OF PROTEIN QUANTITY DATA: 378 X 5155

ccle_prot <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv"))
anyDuplicated(ccle_prot$stripped_cell_line_name)

# ==== Mutation data cleanup ====
rm(list = ls(pattern = "ccle"))

ccle_mut <- fread(paste0(path, "Data/DepMap/20Q2/CCLE_mutations.csv"))
dim(ccle_mut)
colnames(ccle_mut)
# Calculate number of mutations per cell line
temp <- ccle_mut[, c("Variant_Type", "Tumor_Sample_Barcode")]
temp[, nMut := .N, by = "Tumor_Sample_Barcode"]
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

# Read COSMIC Cancer Gene Census data
cgc <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/COSMIC/CosmicMutantExportCensus.tsv")
length(unique(cgc$`Gene name`))
length(unique(cgc$HGVSG))
# Get Genes in this census
cgc_genes <- unique(cgc$`Gene name`)
cgc[Tier == 1]
length(unique(cgc$`Mutation genome position`))  # 922,732
# rm(cgc)

# Subset DepMap mutations based on the CGC genes
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
    library(liftOver)
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

length(unique(lifted_ccle_mut$DepMap_ID))

# Now find the overlap with CGC (which already has GRCh38)
subset <- lifted_ccle_mut[Mutation_Position %in% unique(cgc$`Mutation genome position`)]
table(subset$Variant_Type)
length(unique(subset$DepMap_ID))
# IMPORTANT! There is a loss of 8 cell lines (which do not have a mutation that is in
# CGC) using the Tier 1 data only


### Create a vector of mutations for each cell line with the CGC genes
length(unique(subset$Hugo_Symbol))
sub_dcast <- dcast.data.table(data = subset[, c("DepMap_ID", "Hugo_Symbol")],
                 formula = DepMap_ID ~ Hugo_Symbol, fun.aggregate = nrow)
dim(sub_dcast)

# Save
fwrite(sub_dcast, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_CGC_Mutations_by_Cell.csv"), sep = ',')
dim(cgc_muts)
cgc_muts[1:5, 1:5]
typeof(cgc_muts[1,2])

# Attached the cell line name
cgc_muts <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_CGC_Mutations_by_Cell.csv"))
depmap_samples <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv"))
cgc_muts <- merge(cgc_muts, depmap_samples[, c("DepMap_ID", "stripped_cell_line_name")], by = "DepMap_ID")
setcolorder(cgc_muts, neworder = c("stripped_cell_line_name", colnames(cgc_muts)[-ncol(cgc_muts)]))

# Save
fwrite(cgc_muts, paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_CGC_Mutations_by_Cell.csv"), sep = ',')
cgc_muts <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_CGC_Mutations_by_Cell.csv"))
cgc_muts[1:5, 1:5]
cgc_muts[, DepMap_ID := NULL]
# DIMENSIONS OF CGC MUTATIONAL DATA: 1733 X 697


# ==== Drug Sensitivity Data Cleanup ====
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
ctrp_curves <- fread(paste0(path, "Data/CTRP/v20.data.curves_post_qc.txt"))
exper_data <- fread(paste0(path, "Data/CTRP/v20.meta.per_experiment.txt"))
cell_data <- fread(paste0(path, "Data/CTRP/v20.meta.per_cell_line.txt"))


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