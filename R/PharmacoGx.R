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

ctrp2 <- downloadPSet("CTRPv2_2015")

sensNumber(ctrp2)
?summarizeSensitivityProfiles
??PharmacoGx
??summarizeSensitivityProfiles
# ctrp2_auc <- summarizeSensitivityProfiles(pSet = ctrp2, sensitivity.measure = "auc_recomputed", summary.stat = "median", fill.missing = T)
ctrp2_ic50 <- PharmacoGx::summarizeSensitivityProfiles(object = ctrp2, sensitivity.measure = "ic50_recomputed", summary.stat = "median", fill.missing = T)
# Convert to data.table
ctrp2_auc <- as.data.table(ctrp2_auc, keep.rownames = T)
ctrp2_ic50 <- as.data.table(ctrp2_ic50, keep.rownames = T)
dim(ctrp2_auc)
dim(ctrp2_ic50)
# Convert to long format
ctrp2_auc <- melt.data.table(data = ctrp2_auc)
ctrp2_ic50 <- melt.data.table(data = ctrp2_ic50)
# Remove NAs
ctrp2_auc <- ctrp2_auc[!is.na(value)]
ctrp2_ic50 <- ctrp2_ic50[!is.na(value)]
colnames(ctrp2_auc) <- c("cpd_name", "ccl_name", "area_under_curve")
colnames(ctrp2_ic50) <- c("cpd_name", "ccl_name", "ic50")

# Add smiles data
dim(ctrp2_auc)
dim(ctrp2_ic50)
dim(ctrp2)
ctrp2$cpd_smiles
sum(unique(ctrp$cpd_name) %in% unique(ctrp2_auc$cpd_name))
sum(unique(ctrp2$cpd_name) %in% unique(ctrp2_ic50$cpd_name))
ctrp2_smiles <- data.table(cpd_name = ctrp2@drug$cpd_name, cpd_smiles = ctrp2@drug$cpd_smiles)

length(unique(ctrp2_auc$cpd_name))
length(unique(ctrp2_ic50$cpd_name))
length(unique(ctrp2_smiles$cpd_name))
sum(unique(ctrp2_smiles$cpd_name) == unique(ctrp2_auc$cpd_name))
sum(unique(ctrp2_smiles$cpd_name) == unique(ctrp2_ic50$cpd_name))

sum(tolower(unique(ctrp2_smiles$cpd_name)) == tolower(unique(ctrp2_auc$cpd_name)))
sum(tolower(unique(ctrp2_smiles$cpd_name)) == tolower(unique(ctrp2_ic50$cpd_name)))
changed = !tolower(unique(ctrp2_smiles$cpd_name)) == tolower(unique(ctrp2_auc$cpd_name))
changed = !tolower(unique(ctrp2_smiles$cpd_name)) == tolower(unique(ctrp2_ic50$cpd_name))
temp = data.table(before = unique(ctrp2_smiles$cpd_name)[changed], after = unique(ctrp2_auc$cpd_name)[changed])
temp = data.table(before = unique(ctrp2_smiles$cpd_name)[changed], after = unique(ctrp2_ic50$cpd_name)[changed])
View(temp)

unique(ctrp2_smiles$cpd_name)[!(unique(ctrp2_smiles$cpd_name) == unique(ctrp2_auc$cpd_name))]
unique(ctrp2_auc$cpd_name)[!(unique(ctrp2_smiles$cpd_name) == unique(ctrp2_auc$cpd_name))]

drug_table <- data.table(canonical_name = rownames(drugInfo(ctrp2)),
                         common_name = drugInfo(ctrp2)[, c("cpd_name")],
                         cpd_smiles = drugInfo(ctrp2)[, c("cpd_smiles")])
final_ctrpv2_auc <- merge(ctrp2_auc, drug_table[, c("canonical_name", "cpd_smiles")], by.x = "cpd_name", by.y = "canonical_name")
final_ctrpv2_ic50 <- merge(ctrp2_ic50, drug_table[, c("canonical_name", "cpd_smiles")], by.x = "cpd_name", by.y = "canonical_name")

# Save
fwrite(final_ctrpv2_auc, paste0(path, "Data/DRP_Training_Data/CTRP_AUC_SMILES.txt"))
fwrite(final_ctrpv2_ic50, paste0(path, "Data/DRP_Training_Data/CTRP_IC50_SMILES.txt"))
final_ctrpv2_auc <- fread(paste0(path, "Data/DRP_Training_Data/CTRP_AUC_SMILES.txt"))
final_ctrpv2_ic50 <- fread(paste0(path, "Data/DRP_Training_Data/CTRP_IC50_SMILES.txt"))

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