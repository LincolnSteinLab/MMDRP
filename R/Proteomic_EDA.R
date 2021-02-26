# Proteomic_EDA.R

require(data.table)
require(ggplot2)
require(stringr)

# Cell lines getting engineered for DCAF1 (March 2020)
# NCIH-2170/NCIH-1915/NCIH-1703/NCI-H1373/A549A/NCI-H647/NCIH520

# ==== Create Long CCLE Proteomics Data with Cell Line Info ====
# Read CCLE proteomics data 
ccle_prot <- fread("Data/DepMap/20Q1/Cellular Models/CCLE_normalized_protein_expression.csv")
ccle_nonorm <- fread("Data/DepMap/20Q1/Cellular Models/CCLE_summed_sn_non_normalized.csv")
colnames(ccle_prot)


# Find extrema
min(ccle_prot[,-c(1:48)], na.rm = T)
min(ccle_nonorm[,-c(1:48)], na.rm = T)
max(ccle_prot[,-c(1:48)], na.rm = T)
max(ccle_nonorm[,-c(1:48)], na.rm = T)

# Extract cell line names and replace with column names
ccle_prot_lines <- gsub("\\_Ten.+", "", colnames(ccle_prot)[-c(1:48)])
colnames(ccle_prot)[-c(1:48)] <- ccle_prot_lines

### Attach tissue information for each cell line:
ccle_line_info <- fread("Data/DepMap/20Q1/sample_info.csv")
sum(ccle_prot_lines %in% ccle_line_info$CCLE_Name) / length(ccle_prot_lines)  # All cell line info is available

# Convert data to long format
long_ccle <- melt.data.table(ccle_prot[, c(2, 6, 49:ncol(ccle_prot)), with = F],
                             id.vars = colnames(ccle_prot)[c(2,6)],
                             variable.name = "line", value.name = "norm_quant")

# Divide cell line into ID and tissue
long_ccle$cell_line <- gsub("\\_.+", "", long_ccle$line)
long_ccle$tissue <- gsub(".*?\\_(.+)", "\\1", long_ccle$line)  # '?' means greedy, so the least '.' is used
# long_ccle$line <- NULL

# Merge with cell line info
long_ccle <- merge(long_ccle, ccle_line_info[, c("CCLE_Name", "lineage", "lineage_subtype",
                                                "lineage_sub_subtype", "sex", "disease",
                                                "disease_subtype", "age", "additional_info")],
                   by.x = "line", by.y = "CCLE_Name")

# Save
fwrite(long_ccle, "Data/DepMap/20Q1/long_ccle_prot_data.txt", sep = '\t')

# ==== Source Tissue Statistics ====
t1 <- unique(long_ccle[, c("line", "cell_line", "tissue")])
ggplot(t1) +
  geom_bar(aes(x = tissue)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("Plots/All_CCLE/CCLE_Line_Counts_per_Tissue.png")



# ==== Basal Levels of Proteins of Interest ====
require(data.table)
require(ggplot2)
require(ggridges)
dir.create("Plots")

# Some UniProt IDs
# DCAF1: Q9Y4B6
# COPB2: P35606
# FBXW11: Q9UKB1

long_ccle <- fread("Data/DepMap/20Q1/long_ccle_prot_data.txt")

long_ccle[Uniprot_Acc == "Q9Y4B6"]

# DCAF1
dcaf1_all <- ggplot(data = long_ccle[Uniprot_Acc == "Q9Y4B6"]) + 
  geom_jitter(aes(x = tissue, y = norm_quant), stat = "identity") +
  geom_jitter(data = long_ccle[Uniprot_Acc == "Q9Y4B6" & cell_line %in% c("SW1573", "NCIH460", "NCIH358")], 
              aes(x = tissue, y = norm_quant), colour = "red", size = 2) +
  geom_text(data = long_ccle[Uniprot_Acc == "Q9Y4B6" & cell_line %in% c("SW1573", "NCIH460", "NCIH358")],
            mapping = aes(x = tissue, y = norm_quant, label = cell_line), colour = "red", nudge_x = 2) +
  geom_hline(yintercept = 0, colour = "red") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

dcaf1_all
dir.create("Plots/All_CCLE")
dir.create("Plots/All_CCLE/DCAF1")
ggsave("Plots/All_CCLE/DCAF1/DCAF1_All_CCLE_jitter.png", dcaf1_all)

dcaf1_ridge <- ggplot(data = long_ccle[Uniprot_Acc == "Q9Y4B6"]) + 
  geom_jitter(aes(x = tissue, y = norm_quant), stat = "identity") +
  geom_jitter(data = long_ccle[Uniprot_Acc == "Q9Y4B6" & cell_line %in% c("SW1573", "NCIH460", "NCIH358")], 
              aes(x = tissue, y = norm_quant), colour = "red", size = 2) +
  geom_text(data = long_ccle[Uniprot_Acc == "Q9Y4B6" & cell_line %in% c("SW1573", "NCIH460", "NCIH358")],
            mapping = aes(x = tissue, y = norm_quant, label = cell_line), colour = "red", nudge_x = 2) +
  geom_hline(yintercept = 0, colour = "red") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

require(viridis)
require(hrbrthemes)

dcaf1_ridge <- ggplot(long_ccle[Uniprot_Acc == "Q9Y4B6"],
       aes(x = norm_quant, y = tissue, fill = ..x..)) +
  geom_density_ridges_gradient(scale = 2, rel_min_height = 0.01) +
  scale_fill_viridis(name = "norm_quant", option = "C") +
  labs(title = 'Protein Expression of DCAF1 in CCLE Tissues') +
  theme_ipsum() +
  theme(
    legend.position="none",
    panel.spacing = unit(0.1, "lines"),
    strip.text.x = element_text(size = 8)
  )

dcaf1_ridge
ggsave("Plots/All_CCLE/DCAF1/DCAF1_All_CCLE_ridge.png", dcaf1_ridge, width = 10)


dcaf1_ridge_hist <- ggplot(long_ccle[Uniprot_Acc == "Q9Y4B6"],
                      aes(x = norm_quant, y = tissue, fill = tissue)) +
  geom_density_ridges(alpha=0.6, stat="binline", bins=20) +
  theme_ridges() +
  labs(title = 'Protein Expression of DCAF1 in CCLE Tissues') +
  theme(
    legend.position="none",
    panel.spacing = unit(0.1, "lines"),
    strip.text.x = element_text(size = 8)
  )

dcaf1_ridge_hist
ggsave("Plots/All_CCLE/DCAF1/DCAF1_All_CCLE_ridge_hist.png", dcaf1_ridge_hist, width = 10)


# COPB2
ggplot(data = long_ccle[Uniprot_Acc == "P35606"]) + 
  geom_jitter(aes(x = tissue, y = norm_quant), stat = "identity") +
  geom_jitter(data = long_ccle[Uniprot_Acc == "P35606" & cell_line %in% c("PC9")], 
              aes(x = tissue, y = norm_quant), colour = "red", size = 2) +
  geom_text(data = long_ccle[Uniprot_Acc == "P35606" & cell_line %in% c("PC9")],
            mapping = aes(x = tissue, y = norm_quant, label = cell_line), colour = "red", nudge_x = 2) +
  geom_hline(yintercept = 0, colour = "red") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


ccle_line_info[CCLE_Name %like% "PC9"]
unique(long_ccle[tissue == "LUNG"]$cell_line)


# Cell lines we already tested
ccle_prot[Uniprot_Acc == "Q9Y4B6"]$SW1573_LUNG_TenPx33
ccle_prot[Uniprot_Acc == "Q9Y4B6"]$NCIH358_LUNG_TenPx06
ccle_prot[Uniprot_Acc == "Q9Y4B6"]$NCIH460_LUNG_TenPx22

# Cell lines to be tested (and available in CCLE)
ccle_prot[Uniprot_Acc == "Q9Y4B6"]$NCIH2170_LUNG_TenPx12
ccle_prot[Uniprot_Acc == "Q9Y4B6"]$NCIH1703_LUNG_TenPx19
ccle_prot[Uniprot_Acc == "Q9Y4B6"]$A549_LUNG_TenPx12
ccle_prot[Uniprot_Acc == "Q9Y4B6"]$NCIH520_LUNG_TenPx10

ccle_prot[Uniprot_Acc == "Q9Y4B6", -c(1:48)]
