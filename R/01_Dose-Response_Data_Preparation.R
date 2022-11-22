# PharmacoGx.R
# BiocManager::install("PharmacoGx")
require(PharmacoGx)
require(data.table)
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/"
dir.create(paste0(path, "Data/DRP_Training_Data"))

depmap_samples <- fread(paste0(path, "Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv"))

depmap_samples$stripped_cell_line_name
PharmacoGx::availablePSets()

# CTRPv2 ====
??summarizeSensitivityProfiles
require(PharmacoGx)

ctrp2 <- downloadPSet("CTRPv2_2015")

ctrp2_aac <- summarizeSensitivityProfiles(object = ctrp2, sensitivity.measure = "aac_recomputed", summary.stat = "median", fill.missing = T,)

fwrite(as.data.table(ctrp2@drug, keep.rownames = T), "Data/DRP_Training_Data/CTRP_DRUG_INFO.csv")
drug_info <- ctrp2@drug[, c("gene_symbol_of_protein_target", "target_or_activity_of_compound", "cpd_status")]
drug_info[1:5,]
drug_info <- as.data.table(drug_info, keep.rownames = T)

drug_info[rn %like% "Paclitaxel"]
drug_info[gene_symbol_of_protein_target != "" & cpd_status == "clinical"]$rn
# ctrp2@drug[1:20, c("drugid", "cpd_name")]
# ctrp2_ic50 <- PharmacoGx::summarizeSensitivityProfiles(object = ctrp2, sensitivity.measure = "ic50_recomputed", summary.stat = "median", fill.missing = T)
# Convert to data.table
ctrp2_aac <- as.data.table(ctrp2_aac, keep.rownames = T)
# ctrp2_ic50 <- as.data.table(ctrp2_ic50, keep.rownames = T)
dim(ctrp2_aac)
# dim(ctrp2_ic50)
# Convert to long format
ctrp2_aac <- melt.data.table(data = ctrp2_aac)
# ctrp2_ic50 <- melt.data.table(data = ctrp2_ic50)
# Remove NAs
ctrp2_aac <- ctrp2_aac[!is.na(value)]
# ctrp2_ic50 <- ctrp2_ic50[!is.na(value)]
colnames(ctrp2_aac) <- c("cpd_name", "ccl_name", "area_above_curve")
# colnames(ctrp2_ic50) <- c("cpd_name", "ccl_name", "ic50")

# Add smiles data
dim(ctrp2_aac)  ### 363,634 dose-response curves

dim(ctrp2_ic50)
dim(ctrp2)
ctrp2_smiles <- data.table(drugid = ctrp2@drug$drugid, cpd_smiles = ctrp2@drug$cpd_smiles)

sum(unique(ctrp2@drug$drugid) %in% unique(ctrp2_aac$cpd_name)) / length(unique(ctrp2_aac$cpd_name))


length(unique(ctrp2_aac$cpd_name))
length(unique(ctrp2_ic50$cpd_name))
length(unique(ctrp2_smiles$drugid))

sum(unique(ctrp2_smiles$drugid) %in% unique(ctrp2_aac$cpd_name)) / length(unique(ctrp2_aac$cpd_name))
sum(unique(ctrp2_smiles$drugid) == unique(ctrp2_ic50$cpd_name))

sum(tolower(unique(ctrp2_smiles$drugid)) == tolower(unique(ctrp2_aac$cpd_name)))
sum(tolower(unique(ctrp2_smiles$drugid)) == tolower(unique(ctrp2_ic50$cpd_name)))
changed = !(tolower(unique(ctrp2_smiles$drugid)) == tolower(unique(ctrp2_aac$cpd_name)))
changed = !tolower(unique(ctrp2_smiles$drugid)) == tolower(unique(ctrp2_ic50$cpd_name))

temp = data.table(before = unique(ctrp2_smiles$drugid)[changed], after = unique(ctrp2_aac$cpd_name)[changed])
temp = data.table(before = unique(ctrp2_smiles$drugid)[changed], after = unique(ctrp2_ic50$cpd_name)[changed])
View(temp)

ctrp2_smiles
ctrp2_aac

unique(ctrp2_smiles$drugid)[!(unique(ctrp2_smiles$drugid) == unique(ctrp2_aac$cpd_name))]
unique(ctrp2_aac$cpd_name)[!(unique(ctrp2_smiles$cpd_name) == unique(ctrp2_aac$cpd_name))]

drug_table <- data.table(canonical_name = rownames(drugInfo(ctrp2)),
                         common_name = drugInfo(ctrp2)[, c("cpd_name")],
                         cpd_smiles = drugInfo(ctrp2)[, c("cpd_smiles")])
final_ctrpv2_aac <- merge(ctrp2_aac, drug_table[, c("canonical_name", "cpd_smiles")], by.x = "cpd_name", by.y = "canonical_name")

final_ctrpv2_ic50 <- merge(ctrp2_ic50, drug_table[, c("canonical_name", "cpd_smiles")], by.x = "drugid", by.y = "canonical_name")

# Save
fwrite(final_ctrpv2_aac, paste0(path, "Data/DRP_Training_Data/CTRP_AAC_SMILES.txt"))
fwrite(final_ctrpv2_ic50, paste0(path, "Data/DRP_Training_Data/CTRP_IC50_SMILES.txt"))
final_ctrpv2_aac <- fread(paste0(path, "Data/DRP_Training_Data/CTRP_AAC_SMILES.txt"))
final_ctrpv2_ic50 <- fread(paste0(path, "Data/DRP_Training_Data/CTRP_IC50_SMILES.txt"))

# Add disease information
require(stringr)
line_info <- fread("Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv")
final_ctrpv2_aac <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")

# Remove all hyphens, convert to upper case
line_info$other_ccl_name <- str_replace_all(toupper(line_info$stripped_cell_line_name), "-", "")
final_ctrpv2_aac$other_ccl_name <- str_replace_all(toupper(final_ctrpv2_aac$ccl_name), "-", "")
# Remove all spaces
line_info$other_ccl_name <- str_replace_all(toupper(line_info$other_ccl_name), " ", "")
final_ctrpv2_aac$other_ccl_name <- str_replace_all(toupper(final_ctrpv2_aac$other_ccl_name), " ", "")

sum(unique(final_ctrpv2_aac$other_ccl_name) %in% unique(line_info$other_ccl_name)) / length(unique(final_ctrpv2_aac$other_ccl_name))  ### 0.89177!!!
# A quarter of the data cannot be paired...

# Find what cannot be paired
final_ctrpv2_aac[other_ccl_name %in% unique(final_ctrpv2_aac$other_ccl_name)[!(unique(final_ctrpv2_aac$other_ccl_name) %in% unique(line_info$other_ccl_name))]]

final_ctrpv2_aac <- merge(final_ctrpv2_aac, line_info[, c("other_ccl_name", "primary_disease")], by = "other_ccl_name")
final_ctrpv2_aac$other_ccl_name <- NULL
setcolorder(final_ctrpv2_aac, neworder = c("cpd_name", "ccl_name", "primary_disease", "area_above_curve", "cpd_smiles"))

fwrite(final_ctrpv2_aac, "Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")

# ==== Update August 2021 =====
# Fix Tipifarnbi SMILES
require(data.table)
ctrp_aac <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")
ctrp_aac[cpd_name == "Tipifarnib"]$cpd_smiles <- "CN1C=NC=C1C(C2=CC=C(C=C2)Cl)(C3=CC4=C(C=C3)N(C(=O)C=C4C5=CC(=CC=C5)Cl)C)N"
fwrite(ctrp_aac, "Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")

# ctrp_aac <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")
# ==== Update September 2021 ====
# Add canonical SMILES for some drugs
require(data.table)
gdsc1_aac <- fread("Data/DRP_Training_Data/GDSC1_AAC_SMILES.txt")
gdsc1_aac[cpd_name == "Tamoxifen"]$cpd_smiles <- "CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3"
gdsc1_aac[cpd_name == "JW-7-52-1"]$cpd_smiles <- "CCC(=O)N1CCN(CC1)C2=C(C=C(C=C2)N3C(=O)C=CC4=CN=C5C=CC(=CC5=C43)C6=CC7=CC=CC=C7N=C6)C(F)(F)F"
gdsc1_aac[cpd_name == "KIN001-135"]$cpd_smiles <- "COC1=C(C=C2C(=C1)N=CN2C3=CC(=C(S3)C#N)OCC4=CC=CC=C4S(=O)(=O)C)OC"
fwrite(gdsc1_aac, "Data/DRP_Training_Data/GDSC1_AAC_SMILES.txt")

gdsc2_aac <- fread("Data/DRP_Training_Data/GDSC2_AAC_SMILES.txt")
gdsc2_aac[cpd_name == "Tamoxifen"]$cpd_smiles <- "CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3"
fwrite(gdsc2_aac, "Data/DRP_Training_Data/GDSC2_AAC_SMILES.txt")

# CCLE ====
# ccle <- downloadPSet("CCLE")
# ccle_auc <- summarizeSensitivityProfiles(pSet = ccle, sensitivity.measure = "auc_recomputed", summary.stat = "median", fill.missing = T)
# # Convert to data.table
# ccle_auc <- as.data.table(ccle_auc, keep.rownames = T)
# dim(ccle_auc)
# # Convert to long format
# ccle_auc <- melt.data.table(data = ccle_auc)
# # Remove NAs
# ccle_auc <- ccle_auc[!is.na(value)]
# colnames(ccle_auc) <- c("cpd_name", "ccl_name", "area_under_curve")
# drug_table <- data.table(canonical_name = rownames(drugInfo(ccle)), common_name = drugInfo(ccle)[, c("drug.name")], pubchem = drugInfo(ccle)[, "PubCHEM"])
# ccle_smiles <- webchem::pc_prop(cid = drug_table$pubchem, properties = "CanonicalSMILES")
# # Remove drugs without pubchem entries
# sum(is.na(drug_table$pubchem))  # 15
# drug_table <- drug_table[!is.na(pubchem)]
# drug_table <- merge(drug_table, ccle_smiles, by.x = "pubchem", by.y = "CID")
# 
# 
# ccle_final <- merge(ccle_auc, drug_table[, c("canonical_name", "CanonicalSMILES")], by.x = "cpd_name", by.y = "canonical_name")
# # Save
# fwrite(ccle_final, paste0(path, "Data/DRP_Training_Data/CCLE_AUC_SMILES.txt"))


# GDSC1 ====
gdsc2 <- downloadPSet("GDSC")
gdsc2_auc <- summarizeSensitivityProfiles(pSet = gdsc2, sensitivity.measure = "auc_recomputed", summary.stat = "median", fill.missing = T)
# Convert to data.table
gdsc2_auc <- as.data.table(gdsc2_auc, keep.rownames = T)
dim(gdsc2_auc)
# Convert to long format
gdsc2_auc <- melt.data.table(data = gdsc2_auc)
# Remove NAs
gdsc2_auc <- gdsc2_auc[!is.na(value)]
colnames(gdsc2_auc) <- c("cpd_name", "ccl_name", "area_under_curve")
drug_table <- data.table(canonical_name = rownames(drugInfo(gdsc2)), common_name = drugInfo(gdsc2)[, c("drug.name")], pubchem = drugInfo(gdsc2)[, "PubCHEM"])
gdsc2_smiles <- webchem::pc_prop(cid = drug_table$pubchem, properties = "CanonicalSMILES")
# Remove drugs without pubchem entries
sum(is.na(drug_table$pubchem))  # 15
drug_table <- drug_table[!is.na(pubchem)]
drug_table <- merge(drug_table, gdsc2_smiles, by.x = "pubchem", by.y = "CID")


gdsc2_final <- merge(gdsc2_auc, drug_table[, c("canonical_name", "CanonicalSMILES")], by.x = "cpd_name", by.y = "canonical_name")
# Save
fwrite(gdsc2_final, paste0(path, "Data/DRP_Training_Data/GDSC2_AUC_SMILES.txt"))

gdsc2_final <- fread(paste0(path, "Data/DRP_Training_Data/GDSC2_AUC_SMILES.txt"))
colnames(gdsc2_final)[4] <- "cpd_smiles"