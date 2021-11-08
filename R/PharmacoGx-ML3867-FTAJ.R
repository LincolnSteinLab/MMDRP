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
require(data.table)

ctrp2 <- downloadPSet("CTRPv2_2015")

ctrp2_aac <- summarizeSensitivityProfiles(object = ctrp2, sensitivity.measure = "aac_recomputed", summary.stat = "median", fill.missing = T,)

# Convert to data.table
ctrp2_aac <- as.data.table(ctrp2_aac, keep.rownames = T)

dim(ctrp2_aac)  # 544 x 888
# Convert to long format
ctrp2_aac <- melt.data.table(data = ctrp2_aac)
# Remove NAs
ctrp2_aac <- ctrp2_aac[!is.na(value)]

colnames(ctrp2_aac) <- c("cpd_name", "ccl_name", "area_above_curve")

# Add smiles data
dim(ctrp2_aac)  ### 363,634 dose-response curves

ctrp2_smiles <- data.table(drugid = ctrp2@drug$drugid, cpd_smiles = ctrp2@drug$cpd_smiles)


unique(ctrp2_smiles$drugid)[!(unique(ctrp2_smiles$drugid) == unique(ctrp2_aac$cpd_name))]
unique(ctrp2_aac$cpd_name)[!(unique(ctrp2_smiles$cpd_name) == unique(ctrp2_aac$cpd_name))]

colnames(drugInfo(ctrp2))
drug_info <- as.data.table(drugInfo(ctrp2))
table(drug_info$target_or_activity_of_compound)
drug_table <- data.table(canonical_name = drug_info[, cpd_name],
                         drugid = drug_info[, c(drugid)],
                         pubchem = drug_info[, cid],
                         cpd_smiles = drug_info[, cpd_smiles])
sum(grepl(":", drug_table$canonical_name))  # 49 drugs are combinations
drug_table[grepl(":", drug_table$canonical_name)]$smiles
drug_table <- drug_table[!grepl(":", drug_table$canonical_name),]
# Discard information inside parenthesis of drugid (like alisertib:navitoclax (2:1 mol/mol))
require(stringr)
drug_table[is.na(cpd_smiles)]  # all compounds have SMILES

sum(unique(ctrp2_aac$cpd_name) %in% drug_table$canonical_name)  # 259
sum(unique(ctrp2_aac$cpd_name) %in% drug_table$drugid)  # 495
final_ctrpv2 <- merge(ctrp2_aac, drug_table[, c("drugid", "cpd_smiles")], by.x = "cpd_name", by.y = "drugid")

# Add disease information
require(stringr)
line_info <- fread("Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv")

# Remove names inside brackets
final_ctrpv2$other_ccl_name <- str_replace(final_ctrpv2$ccl_name, "(.+)\\[.+\\]", "\\1")
# Remove all non-alphanumeric characters, convert to upper case
line_info$other_ccl_name <- str_replace_all(toupper(line_info$stripped_cell_line_name), "\\W+", "")
final_ctrpv2$other_ccl_name <- str_replace_all(toupper(final_ctrpv2$other_ccl_name), "\\W+", "")
unique(final_ctrpv2$other_ccl_name)

sum(unique(final_ctrpv2$other_ccl_name) %in% unique(line_info$other_ccl_name)) / length(unique(final_ctrpv2$other_ccl_name))  # 94.25% overlap


# Find what cannot be paired
unique(final_ctrpv2[other_ccl_name %in% unique(final_ctrpv2$other_ccl_name)[!(unique(final_ctrpv2$other_ccl_name) %in% unique(line_info$other_ccl_name))], c("ccl_name", "other_ccl_name")])

# Manually update cell line names
final_ctrpv2[other_ccl_name == "A3KAWAKAMI"]$other_ccl_name <- "A3KAW"
final_ctrpv2[other_ccl_name == "A4FUKUDA"]$other_ccl_name <- "A4FUK"
final_ctrpv2[other_ccl_name == "JIYOYE"]$other_ccl_name <- "JIYOYEP2003"
final_ctrpv2[other_ccl_name == "MZMEL2"]$other_ccl_name <- "MZ2MEL"
final_ctrpv2[other_ccl_name == "MZMEL7"]$other_ccl_name <- "MZ7MEL"
final_ctrpv2[other_ccl_name == "MZPC1"]$other_ccl_name <- "MZ1PC"
final_ctrpv2[other_ccl_name == "NCIH510A"]$other_ccl_name <- "NCIH510"
final_ctrpv2[other_ccl_name == "NTERA2"]$other_ccl_name <- "NTERA2CLD1"
final_ctrpv2[other_ccl_name == "ONDA10"]$other_ccl_name <- "NO10"
final_ctrpv2[other_ccl_name == "ONDA11"]$other_ccl_name <- "NO11"
final_ctrpv2[other_ccl_name == "RXF393L"]$other_ccl_name <- "RXF393"
final_ctrpv2[other_ccl_name == "SJNB5"]$other_ccl_name <- "NB5"
final_ctrpv2[other_ccl_name == "SJNB6"]$other_ccl_name <- "NB6"
final_ctrpv2[other_ccl_name == "SJNB7"]$other_ccl_name <- "NB7"
final_ctrpv2[other_ccl_name == "SJNB10"]$other_ccl_name <- "NB10"
final_ctrpv2[other_ccl_name == "SJNB12"]$other_ccl_name <- "NB12"
final_ctrpv2[other_ccl_name == "SJNB13"]$other_ccl_name <- "NB13"
final_ctrpv2[other_ccl_name == "SJNB14"]$other_ccl_name <- "NB14"
final_ctrpv2[other_ccl_name == "SR"]$other_ccl_name <- "SR786"
final_ctrpv2[other_ccl_name == "U87MGATCC"]$other_ccl_name <- "U87MG"
final_ctrpv2[other_ccl_name == "VL51"]$other_ccl_name <- "SLVL"
final_ctrpv2[other_ccl_name == "HEPG2C3A"]$other_ccl_name <- "HEPG2"
final_ctrpv2[other_ccl_name == "KTCTL13"]$other_ccl_name <- "RCCER"
final_ctrpv2[other_ccl_name == "KTCTL140"]$other_ccl_name <- "RCCJF"
final_ctrpv2[other_ccl_name == "KTCTL195"]$other_ccl_name <- "RCCJW"
final_ctrpv2[other_ccl_name == "KTCTL1M"]$other_ccl_name <- "RCCMF"
final_ctrpv2[other_ccl_name == "KTCTL21"]$other_ccl_name <- "RCCAB"
final_ctrpv2[other_ccl_name == "KTCTL26A"]$other_ccl_name <- "RCCFG2"
final_ctrpv2[other_ccl_name == "NCIH2369"]$other_ccl_name <- "H2369"
final_ctrpv2[other_ccl_name == "NCIH2373"]$other_ccl_name <- "H2373"
final_ctrpv2[other_ccl_name == "NCIH2461"]$other_ccl_name <- "H2461"
final_ctrpv2[other_ccl_name == "NCIH2591"]$other_ccl_name <- "H2591"
final_ctrpv2[other_ccl_name == "NCIH2595"]$other_ccl_name <- "H2595"
final_ctrpv2[other_ccl_name == "NCIH2803"]$other_ccl_name <- "H2803"
final_ctrpv2[other_ccl_name == "NCIH2804"]$other_ccl_name <- "H2804"
final_ctrpv2[other_ccl_name == "NCIH2810"]$other_ccl_name <- "H2810"
final_ctrpv2[other_ccl_name == "NCIH2818"]$other_ccl_name <- "H2818"
final_ctrpv2[other_ccl_name == "NCIH2869"]$other_ccl_name <- "H2869"
final_ctrpv2[other_ccl_name == "NCIH290"]$other_ccl_name <- "H290"
final_ctrpv2[other_ccl_name == "NCIH3118"]$other_ccl_name <- "H3118"
final_ctrpv2[other_ccl_name == "NCIH513"]$other_ccl_name <- "H513"
final_ctrpv2[other_ccl_name == "OACM51C"]$other_ccl_name <- "OACM51"
final_ctrpv2[other_ccl_name == "PCI04B"]$other_ccl_name <- "PCI4B"
final_ctrpv2[other_ccl_name == "PCI06A"]$other_ccl_name <- "PCI6A"
final_ctrpv2[other_ccl_name == "SKNMCIXC"]$other_ccl_name <- "MCIXC"
final_ctrpv2[other_ccl_name == "WRO"]$other_ccl_name <- "RO82W1"
final_ctrpv2[other_ccl_name == "NBTU1"]$other_ccl_name <- "NBTU110"
final_ctrpv2[other_ccl_name == "SJNB17"]$other_ccl_name <- "NB17"
final_ctrpv2[other_ccl_name == "NCIH2722"]$other_ccl_name <- "H2722"
final_ctrpv2[other_ccl_name == "NCIH2731"]$other_ccl_name <- "H2731"
final_ctrpv2[other_ccl_name == "NCIH2795"]$other_ccl_name <- "H2795"
final_ctrpv2[other_ccl_name == "OVCAR3"]$other_ccl_name <- "NIHOVCAR3"
final_ctrpv2[other_ccl_name == "MO"]$other_ccl_name <- "MOT"
final_ctrpv2[other_ccl_name == "YAMATOSS"]$other_ccl_name <- "YAMATO"
final_ctrpv2[other_ccl_name == "TM8716"]$other_ccl_name <- "TM87"
final_ctrpv2[other_ccl_name == "RERFLCA1"]$other_ccl_name <- "RERFLCAI"
final_ctrpv2[other_ccl_name == "MDAMB435"]$other_ccl_name <- "MDA435"


sum(unique(final_ctrpv2$other_ccl_name) %in% unique(line_info$other_ccl_name)) / length(unique(final_ctrpv2$other_ccl_name))  # 95.15% overlap
unique(final_ctrpv2[other_ccl_name %in% unique(final_ctrpv2$other_ccl_name)[!(unique(final_ctrpv2$other_ccl_name) %in% unique(line_info$other_ccl_name))], c("ccl_name", "other_ccl_name")])


final_ctrpv2 <- merge(final_ctrpv2, line_info[, c("other_ccl_name", "primary_disease")], by = "other_ccl_name")
# Replace modified ccl_name with original
final_ctrpv2$ccl_name <- NULL
colnames(final_ctrpv2)[colnames(final_ctrpv2) == "other_ccl_name"] <- "ccl_name"
setcolorder(final_ctrpv2, neworder = c("cpd_name", "ccl_name", "primary_disease", "area_above_curve", "cpd_smiles"))
# colnames(final_ctrpv2)[colnames(final_ctrpv2) == "CanonicalSMILES"] <- "cpd_smiles"

fwrite(final_ctrpv2, "Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")

ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")
mean(ctrp$area_above_curve)
max(ctrp$area_above_curve)
min(ctrp$area_above_curve)
quantile(ctrp$area_above_curve)
hist(ctrp$area_above_curve)
# Mean by drug
ctrp[ , mean_by_drug := mean(area_above_curve), by = "cpd_name"]
ctrp[ , mean_by_cell := mean(area_above_curve), by = "ccl_name"]
# hist(unique(ctrp$mean_by_drug))

# sum(unique(final_ctrpv2_aac$ccl_name) %in% unique(line_info$stripped_cell_line_name))
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
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/"

require(PharmacoGx)
require(data.table)
PharmacoGx::availablePSets()

gdsc1 <- downloadPSet("GDSC_2020(v1-8.2)", timeout = 3600)
gdsc1_aac <- summarizeSensitivityProfiles(object = gdsc1, sensitivity.measure = "aac_recomputed", summary.stat = "median", fill.missing = T)
# Convert to data.table
gdsc1_aac <- as.data.table(gdsc1_aac, keep.rownames = T)
dim(gdsc1_aac)
# Convert to long format
gdsc1_aac <- melt.data.table(data = gdsc1_aac)
dim(gdsc1_aac)
# Remove NAs
gdsc1_aac <- gdsc1_aac[!is.na(value)]
dim(gdsc1_aac)
colnames(gdsc1_aac) <- c("cpd_name", "ccl_name", "area_above_curve")
sum(is.na(gdsc1@drug$smiles))
gdsc1@drug$smiles
gdsc1@drug$inchikey
colnames(drugInfo(gdsc1))
drug_info <- as.data.table(drugInfo(gdsc1))
drug_info[is.na(cid) & is.na(smiles)]
drug_table <- data.table(canonical_name = drug_info[, DRUG_NAME],
                         drugid = drug_info[, c(drugid)],
                         pubchem = drug_info[, cid],
                         smiles = drug_info[, smiles])
# Get pubchem IDs of those that don't have pubchem IDs
temp <- webchem::get_cid(query = drug_table[is.na(pubchem)]$canonical_name, from = "name")
temp <- as.data.table(temp)
# Merge with drug_table
drug_table <- merge(drug_table, temp, by.x = "canonical_name", by.y = "query", all.x = T)
drug_table[is.na(pubchem) & !is.na(cid), pubchem:= cid]
drug_table$cid <- NULL

# Get smiles using pubchem cid 
gdsc1_smiles <- webchem::pc_prop(cid = drug_table$pubchem, properties = "CanonicalSMILES")
gdsc1_smiles <- as.data.table(gdsc1_smiles)
gdsc1_smiles <- unique(gdsc1_smiles)
# Merge
drug_table <- merge(drug_table, gdsc1_smiles, by.x = "pubchem", by.y = "CID", all.x = T)
drug_table[is.na(CanonicalSMILES)]  # 53
drug_table[is.na(CanonicalSMILES) & !is.na(smiles), CanonicalSMILES := smiles]
drug_table[is.na(CanonicalSMILES)]  # 50

drug_table$smiles <- NULL

gdsc1_final <- merge(gdsc1_aac, drug_table[, c("drugid", "CanonicalSMILES")], by.x = "cpd_name", by.y = "drugid")
# Remove DRs without SMILES
gdsc1_final <- gdsc1_final[!is.na(CanonicalSMILES)]

# Save
# fwrite(gdsc1_final, paste0(path, "Data/DRP_Training_Data/GDSC1_AAC_SMILES.txt"))

# Add disease information ==
require(stringr)
line_info <- fread("Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv")

# Remove names inside brackets
gdsc1_final$other_ccl_name <- str_replace(gdsc1_final$ccl_name, "(.+)\\[.+\\]", "\\1")
# Remove all non-alphanumeric characters, convert to upper case
line_info$other_ccl_name <- str_replace_all(toupper(line_info$stripped_cell_line_name), "\\W+", "")
gdsc1_final$other_ccl_name <- str_replace_all(toupper(gdsc1_final$other_ccl_name), "\\W+", "")
unique(gdsc1_final$other_ccl_name)

sum(unique(gdsc1_final$other_ccl_name) %in% unique(line_info$other_ccl_name)) / length(unique(gdsc1_final$other_ccl_name))  # 94.5% overlap

# Find what cannot be paired
unique(gdsc1_final[other_ccl_name %in% unique(gdsc1_final$other_ccl_name)[!(unique(gdsc1_final$other_ccl_name) %in% unique(line_info$other_ccl_name))], c("ccl_name", "other_ccl_name")])

# Manually update cell line names
gdsc1_final[other_ccl_name == "A3KAWAKAMI"]$other_ccl_name <- "A3KAW"
gdsc1_final[other_ccl_name == "A4FUKUDA"]$other_ccl_name <- "A4FUK"
gdsc1_final[other_ccl_name == "JIYOYE"]$other_ccl_name <- "JIYOYEP2003"
gdsc1_final[other_ccl_name == "MZMEL2"]$other_ccl_name <- "MZ2MEL"
gdsc1_final[other_ccl_name == "MZMEL7"]$other_ccl_name <- "MZ7MEL"
gdsc1_final[other_ccl_name == "MZPC1"]$other_ccl_name <- "MZ1PC"
gdsc1_final[other_ccl_name == "NCIH510A"]$other_ccl_name <- "NCIH510"
gdsc1_final[other_ccl_name == "NTERA2"]$other_ccl_name <- "NTERA2CLD1"
gdsc1_final[other_ccl_name == "ONDA10"]$other_ccl_name <- "NO10"
gdsc1_final[other_ccl_name == "ONDA11"]$other_ccl_name <- "NO11"
gdsc1_final[other_ccl_name == "RXF393L"]$other_ccl_name <- "RXF393"
gdsc1_final[other_ccl_name == "SJNB5"]$other_ccl_name <- "NB5"
gdsc1_final[other_ccl_name == "SJNB6"]$other_ccl_name <- "NB6"
gdsc1_final[other_ccl_name == "SJNB7"]$other_ccl_name <- "NB7"
gdsc1_final[other_ccl_name == "SJNB10"]$other_ccl_name <- "NB10"
gdsc1_final[other_ccl_name == "SJNB12"]$other_ccl_name <- "NB12"
gdsc1_final[other_ccl_name == "SJNB13"]$other_ccl_name <- "NB13"
gdsc1_final[other_ccl_name == "SJNB14"]$other_ccl_name <- "NB14"
gdsc1_final[other_ccl_name == "SR"]$other_ccl_name <- "SR786"
gdsc1_final[other_ccl_name == "U87MGATCC"]$other_ccl_name <- "U87MG"
gdsc1_final[other_ccl_name == "VL51"]$other_ccl_name <- "SLVL"
gdsc1_final[other_ccl_name == "HEPG2C3A"]$other_ccl_name <- "HEPG2"
gdsc1_final[other_ccl_name == "KTCTL13"]$other_ccl_name <- "RCCER"
gdsc1_final[other_ccl_name == "KTCTL140"]$other_ccl_name <- "RCCJF"
gdsc1_final[other_ccl_name == "KTCTL195"]$other_ccl_name <- "RCCJW"
gdsc1_final[other_ccl_name == "KTCTL1M"]$other_ccl_name <- "RCCMF"
gdsc1_final[other_ccl_name == "KTCTL21"]$other_ccl_name <- "RCCAB"
gdsc1_final[other_ccl_name == "KTCTL26A"]$other_ccl_name <- "RCCFG2"
gdsc1_final[other_ccl_name == "NCIH2369"]$other_ccl_name <- "H2369"
gdsc1_final[other_ccl_name == "NCIH2373"]$other_ccl_name <- "H2373"
gdsc1_final[other_ccl_name == "NCIH2461"]$other_ccl_name <- "H2461"
gdsc1_final[other_ccl_name == "NCIH2591"]$other_ccl_name <- "H2591"
gdsc1_final[other_ccl_name == "NCIH2595"]$other_ccl_name <- "H2595"
gdsc1_final[other_ccl_name == "NCIH2803"]$other_ccl_name <- "H2803"
gdsc1_final[other_ccl_name == "NCIH2804"]$other_ccl_name <- "H2804"
gdsc1_final[other_ccl_name == "NCIH2810"]$other_ccl_name <- "H2810"
gdsc1_final[other_ccl_name == "NCIH2818"]$other_ccl_name <- "H2818"
gdsc1_final[other_ccl_name == "NCIH2869"]$other_ccl_name <- "H2869"
gdsc1_final[other_ccl_name == "NCIH290"]$other_ccl_name <- "H290"
gdsc1_final[other_ccl_name == "NCIH3118"]$other_ccl_name <- "H3118"
gdsc1_final[other_ccl_name == "NCIH513"]$other_ccl_name <- "H513"
gdsc1_final[other_ccl_name == "OACM51C"]$other_ccl_name <- "OACM51"
gdsc1_final[other_ccl_name == "PCI04B"]$other_ccl_name <- "PCI4B"
gdsc1_final[other_ccl_name == "PCI06A"]$other_ccl_name <- "PCI6A"
gdsc1_final[other_ccl_name == "SKNMCIXC"]$other_ccl_name <- "MCIXC"
gdsc1_final[other_ccl_name == "WRO"]$other_ccl_name <- "RO82W1"
gdsc1_final[other_ccl_name == "NBTU1"]$other_ccl_name <- "NBTU110"
gdsc1_final[other_ccl_name == "SJNB17"]$other_ccl_name <- "NB17"
gdsc1_final[other_ccl_name == "NCIH2722"]$other_ccl_name <- "H2722"
gdsc1_final[other_ccl_name == "NCIH2731"]$other_ccl_name <- "H2731"
gdsc1_final[other_ccl_name == "NCIH2795"]$other_ccl_name <- "H2795"
gdsc1_final[other_ccl_name == "OVCAR3"]$other_ccl_name <- "NIHOVCAR3"
gdsc1_final[other_ccl_name == "MO"]$other_ccl_name <- "MOT"


sum(unique(gdsc1_final$other_ccl_name) %in% unique(line_info$other_ccl_name)) / length(unique(gdsc1_final$other_ccl_name))  # 99.9% overlap


gdsc1_final <- merge(gdsc1_final, line_info[, c("other_ccl_name", "primary_disease")], by = "other_ccl_name")
# Replace modified ccl_name with original
gdsc1_final$ccl_name <- NULL
colnames(gdsc1_final)[colnames(gdsc1_final) == "other_ccl_name"] <- "ccl_name"
setcolorder(gdsc1_final, neworder = c("cpd_name", "ccl_name", "primary_disease", "area_above_curve", "CanonicalSMILES"))
colnames(gdsc1_final)[colnames(gdsc1_final) == "CanonicalSMILES"] <- "cpd_smiles"

fwrite(gdsc1_final, "Data/DRP_Training_Data/GDSC1_AAC_SMILES.txt")
 
# GDSC2 ==========================================================
path = "/Users/ftaj/OneDrive - University of Toronto/Drug_Response/"

require(PharmacoGx)
require(data.table)
PharmacoGx::availablePSets()

gdsc2 <- downloadPSet("GDSC_2020(v2-8.2)", timeout = 3600)
gdsc2_aac <- summarizeSensitivityProfiles(object = gdsc2, sensitivity.measure = "aac_recomputed", summary.stat = "median", fill.missing = T)
# Convert to data.table
gdsc2_aac <- as.data.table(gdsc2_aac, keep.rownames = T)
dim(gdsc2_aac)
# Convert to long format
gdsc2_aac <- melt.data.table(data = gdsc2_aac)
dim(gdsc2_aac)
# Remove NAs
gdsc2_aac <- gdsc2_aac[!is.na(value)]
dim(gdsc2_aac)
colnames(gdsc2_aac) <- c("cpd_name", "ccl_name", "area_above_curve")

drug_info <- as.data.table(drugInfo(gdsc2))
drug_info[is.na(cid) & is.na(smiles)]
drug_table <- data.table(canonical_name = drug_info[, DRUG_NAME],
                         drugid = drug_info[, c(drugid)],
                         pubchem = drug_info[, cid],
                         smiles = drug_info[, smiles])
# Get pubchem IDs of those that don't have pubchem IDs
temp <- webchem::get_cid(query = drug_table[is.na(pubchem)]$canonical_name, from = "name")
temp <- as.data.table(temp)
# Merge with drug_table
drug_table <- merge(drug_table, temp, by.x = "canonical_name", by.y = "query", all.x = T)
drug_table[is.na(pubchem) & !is.na(cid), pubchem:= cid]
drug_table$cid <- NULL

# Get smiles using pubchem cid 
gdsc2_smiles <- webchem::pc_prop(cid = drug_table$pubchem, properties = "CanonicalSMILES")
gdsc2_smiles <- as.data.table(gdsc2_smiles)
gdsc2_smiles <- unique(gdsc2_smiles)
# Merge
drug_table <- merge(drug_table, gdsc2_smiles, by.x = "pubchem", by.y = "CID", all.x = T)
drug_table[is.na(CanonicalSMILES)]  # 18
drug_table[is.na(CanonicalSMILES) & !is.na(smiles), CanonicalSMILES := smiles]
drug_table[is.na(CanonicalSMILES)]  # 18

drug_table$smiles <- NULL

gdsc2_final <- merge(gdsc2_aac, drug_table[, c("drugid", "CanonicalSMILES")], by.x = "cpd_name", by.y = "drugid")
# Remove DRs without SMILES
gdsc2_final <- gdsc2_final[!is.na(CanonicalSMILES)]
dim(gdsc2_final)  # 115,885 dose responses

# Save
# fwrite(gdsc2_final, paste0(path, "Data/DRP_Training_Data/GDSC2_AAC_SMILES.txt"))

# Add disease information
require(stringr)
line_info <- fread("Data/DRP_Training_Data/DepMap_20Q2_Line_Info.csv")

# Remove all hyphens, convert to upper case
# line_info$other_ccl_name <- str_replace_all(toupper(line_info$stripped_cell_line_name), "-", "")
# gdsc2_final$other_ccl_name <- str_replace_all(toupper(gdsc2_final$ccl_name), "-", "")
# # Remove all spaces
# line_info$other_ccl_name <- str_replace_all(toupper(line_info$other_ccl_name), " ", "")
# gdsc2_final$other_ccl_name <- str_replace_all(toupper(gdsc2_final$other_ccl_name), " ", "")
# # Remove all slashes
# line_info$other_ccl_name <- str_replace_all(toupper(line_info$other_ccl_name), "/", "")
# gdsc2_final$other_ccl_name <- str_replace_all(toupper(gdsc2_final$other_ccl_name), "/", "")
unique(gdsc2_final$ccl_name)
# Remove names inside brackets (like KS-1 [Human glioblastoma])
gdsc2_final$other_ccl_name <- str_replace(gdsc2_final$ccl_name, "(.+)\\[.+\\]", "\\1")
# Remove non-alphanumeric characters
line_info$other_ccl_name <- str_replace_all(toupper(line_info$stripped_cell_line_name), "\\W+", "")
gdsc2_final$other_ccl_name <- str_replace_all(toupper(gdsc2_final$other_ccl_name), "\\W+", "")

sum(unique(gdsc2_final$other_ccl_name) %in% unique(line_info$other_ccl_name)) / length(unique(gdsc2_final$other_ccl_name))  # 95.4% overlap

# Find what cannot be paired
# gdsc2_final[other_ccl_name %in% unique(gdsc2_final$other_ccl_name)[!(unique(gdsc2_final$other_ccl_name) %in% unique(line_info$other_ccl_name))]]
unique(gdsc2_final[other_ccl_name %in% unique(gdsc2_final$other_ccl_name)[!(unique(gdsc2_final$other_ccl_name) %in% unique(line_info$other_ccl_name))], c("ccl_name", "other_ccl_name")])

# Manually update cell line names (similar to GDSC1)
gdsc2_final[other_ccl_name == "A3KAWAKAMI"]$other_ccl_name <- "A3KAW"
gdsc2_final[other_ccl_name == "A4FUKUDA"]$other_ccl_name <- "A4FUK"
gdsc2_final[other_ccl_name == "JIYOYE"]$other_ccl_name <- "JIYOYEP2003"
gdsc2_final[other_ccl_name == "MZMEL2"]$other_ccl_name <- "MZ2MEL"
gdsc2_final[other_ccl_name == "MZMEL7"]$other_ccl_name <- "MZ7MEL"
gdsc2_final[other_ccl_name == "MZPC1"]$other_ccl_name <- "MZ1PC"
gdsc2_final[other_ccl_name == "NCIH510A"]$other_ccl_name <- "NCIH510"
gdsc2_final[other_ccl_name == "NTERA2"]$other_ccl_name <- "NTERA2CLD1"
gdsc2_final[other_ccl_name == "ONDA10"]$other_ccl_name <- "NO10"
gdsc2_final[other_ccl_name == "ONDA11"]$other_ccl_name <- "NO11"
gdsc2_final[other_ccl_name == "RXF393L"]$other_ccl_name <- "RXF393"
gdsc2_final[other_ccl_name == "SJNB5"]$other_ccl_name <- "NB5"
gdsc2_final[other_ccl_name == "SJNB6"]$other_ccl_name <- "NB6"
gdsc2_final[other_ccl_name == "SJNB7"]$other_ccl_name <- "NB7"
gdsc2_final[other_ccl_name == "SJNB10"]$other_ccl_name <- "NB10"
gdsc2_final[other_ccl_name == "SJNB12"]$other_ccl_name <- "NB12"
gdsc2_final[other_ccl_name == "SJNB13"]$other_ccl_name <- "NB13"
gdsc2_final[other_ccl_name == "SJNB14"]$other_ccl_name <- "NB14"
gdsc2_final[other_ccl_name == "SR"]$other_ccl_name <- "SR786"
gdsc2_final[other_ccl_name == "U87MGATCC"]$other_ccl_name <- "U87MG"
gdsc2_final[other_ccl_name == "VL51"]$other_ccl_name <- "SLVL"
gdsc2_final[other_ccl_name == "HEPG2C3A"]$other_ccl_name <- "HEPG2"
gdsc2_final[other_ccl_name == "KTCTL13"]$other_ccl_name <- "RCCER"
gdsc2_final[other_ccl_name == "KTCTL140"]$other_ccl_name <- "RCCJF"
gdsc2_final[other_ccl_name == "KTCTL195"]$other_ccl_name <- "RCCJW"
gdsc2_final[other_ccl_name == "KTCTL1M"]$other_ccl_name <- "RCCMF"
gdsc2_final[other_ccl_name == "KTCTL21"]$other_ccl_name <- "RCCAB"
gdsc2_final[other_ccl_name == "KTCTL26A"]$other_ccl_name <- "RCCFG2"
gdsc2_final[other_ccl_name == "NCIH2369"]$other_ccl_name <- "H2369"
gdsc2_final[other_ccl_name == "NCIH2373"]$other_ccl_name <- "H2373"
gdsc2_final[other_ccl_name == "NCIH2461"]$other_ccl_name <- "H2461"
gdsc2_final[other_ccl_name == "NCIH2591"]$other_ccl_name <- "H2591"
gdsc2_final[other_ccl_name == "NCIH2595"]$other_ccl_name <- "H2595"
gdsc2_final[other_ccl_name == "NCIH2803"]$other_ccl_name <- "H2803"
gdsc2_final[other_ccl_name == "NCIH2804"]$other_ccl_name <- "H2804"
gdsc2_final[other_ccl_name == "NCIH2810"]$other_ccl_name <- "H2810"
gdsc2_final[other_ccl_name == "NCIH2818"]$other_ccl_name <- "H2818"
gdsc2_final[other_ccl_name == "NCIH2869"]$other_ccl_name <- "H2869"
gdsc2_final[other_ccl_name == "NCIH290"]$other_ccl_name <- "H290"
gdsc2_final[other_ccl_name == "NCIH3118"]$other_ccl_name <- "H3118"
gdsc2_final[other_ccl_name == "NCIH513"]$other_ccl_name <- "H513"
gdsc2_final[other_ccl_name == "OACM51C"]$other_ccl_name <- "OACM51"
gdsc2_final[other_ccl_name == "PCI04B"]$other_ccl_name <- "PCI4B"
gdsc2_final[other_ccl_name == "PCI06A"]$other_ccl_name <- "PCI6A"
gdsc2_final[other_ccl_name == "SKNMCIXC"]$other_ccl_name <- "MCIXC"
gdsc2_final[other_ccl_name == "WRO"]$other_ccl_name <- "RO82W1"
gdsc2_final[other_ccl_name == "NBTU1"]$other_ccl_name <- "NBTU110"
gdsc2_final[other_ccl_name == "SJNB17"]$other_ccl_name <- "NB17"
gdsc2_final[other_ccl_name == "NCIH2722"]$other_ccl_name <- "H2722"
gdsc2_final[other_ccl_name == "NCIH2731"]$other_ccl_name <- "H2731"
gdsc2_final[other_ccl_name == "NCIH2795"]$other_ccl_name <- "H2795"
gdsc2_final[other_ccl_name == "MO"]$other_ccl_name <- "MOT"
gdsc2_final[other_ccl_name == "OVCAR3"]$other_ccl_name <- "NIHOVCAR3"

unique(gdsc2_final[other_ccl_name %in% unique(gdsc2_final$other_ccl_name)[!(unique(gdsc2_final$other_ccl_name) %in% unique(line_info$other_ccl_name))], c("ccl_name", "other_ccl_name")])

sum(unique(gdsc2_final$other_ccl_name) %in% unique(line_info$other_ccl_name)) / length(unique(gdsc2_final$other_ccl_name))  # 99.87% overlap


gdsc2_final <- merge(gdsc2_final, line_info[, c("other_ccl_name", "primary_disease")], by = "other_ccl_name")
# Replace modified ccl_name with original
gdsc2_final$ccl_name <- NULL
colnames(gdsc2_final)[colnames(gdsc2_final) == "other_ccl_name"] <- "ccl_name"
setcolorder(gdsc2_final, neworder = c("cpd_name", "ccl_name", "primary_disease", "area_above_curve", "CanonicalSMILES"))
colnames(gdsc2_final)[colnames(gdsc2_final) == "CanonicalSMILES"] <- "cpd_smiles"

fwrite(gdsc2_final, "Data/DRP_Training_Data/GDSC2_AAC_SMILES.txt")
