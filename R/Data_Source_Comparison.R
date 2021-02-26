# Find the percentage of cell lines with proteomics data that also have drug response data in DepMap

require(data.table)
d <- fread("/Users/ftaj/Downloads/primary-screen-replicate-collapsed-logfold-change.csv")
protein <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DepMap/20Q2/CCLE_protein_quant_current_normalized.csv")
cells <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/DepMap/20Q2/DepMap_Cell_Line_info-3.csv")

dim(d)

c <- c[, c("DepMap_ID", "CCLE_Name")]
cp <- merge(c, p, by.x = "CCLE_Name", by.y = "CCLE Code")
p
 
sum(cp$DepMap_ID %in% d$V1)/375
colnames(d)


# ==== Compare to GDSC ====
require(data.table)
gdsc1 <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/GDSC/GDSC1_Fitted_Dose_Response.csv")
length(unique(gdsc1$CELL_LINE_NAME))  # 987 cell lines in GDSC2
gdsc2 <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/GDSC/GDSC2_Fitted_Dose_Response.csv")
length(unique(gdsc2$CELL_LINE_NAME))  # 809 cell lines in GDSC2

gdsc_cells <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/GDSC/GDSC_Cell_Lines_Details.csv")

# Find overlap with DepMap cell lines
sum(toupper(gdsc_cells$`Sample Name`) %in% toupper(cells$stripped_cell_line_name)) / length(gdsc_cells$`Sample Name`)
sum(toupper(gdsc_cells$`COSMIC identifier`) %in% toupper(cells$COSMICID)) / length(gdsc_cells$`Sample Name`)
# COSMIC IDs have ~98% overlap
# This implies that different names are being used