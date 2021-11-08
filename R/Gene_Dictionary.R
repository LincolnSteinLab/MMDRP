# Gene_Dictionary.R

if (!require(data.table)) {
    install.packages("data.table")
    library(data.table)
}
if (!require(biomaRt)) {
    BiocInstaller::biocLite("biomaRt")
    library(biomaRt)
}
if (!require(curl)) {
    install.packages("curl")
    library(curl)
}


# ==== Create a dictionary for all genes ====
# Load a gene expression file
exp_file <- get(load("Data/TCGA/SummarizedExperiments/Exp/TCGA-ACC_GeneExpression.rdata"))
all_genes <- rownames(exp_file)
# Use an older version of biomaRt to ensure hg19 coordinates are retrieved
ensembl_75 = useMart(
    biomart = "ENSEMBL_MART_ENSEMBL",
    host = "feb2014.archive.ensembl.org",
    path = "/biomart/martservice",
    dataset = "hsapiens_gene_ensembl"
)
atts <- listAttributes(ensembl_75, what = "name")
atts[grep(pattern = "gene", x = atts, ignore.case = T)]
atts[grep(pattern = "exon", x = atts, ignore.case = T)]
atts[grep(pattern = "location", x = atts, ignore.case = T)]
atts[grep(pattern = "entrez", x = atts, ignore.case = T)]
# Retrieve relevant attributes
length(all_genes)
dict <- biomaRt::getBM(attributes = c("ensembl_gene_id", "entrezgene",
                                      "external_gene_id",
                                      "gene_biotype",
                                      'chromosome_name', 'start_position',
                                      'end_position', 'strand'),
                       filters = "ensembl_gene_id", values = all_genes,
                       mart = ensembl_75)
dict_dt <- as.data.table(dict)
nrow(dict_dt)
t <- biomaRt::getBM(attributes = c("ensembl_gene_id", "ensembl_exon_id"),
                    filters = "ensembl_gene_id", values = all_genes,
                    mart = ensembl_75)
t <- as.data.table(t)
t[, exon_count := nrow(.SD), by = ensembl_gene_id]
exon_counts <- unique(t[, c("ensembl_gene_id", "exon_count")])
dict_dt <- merge(x = dict_dt, y = exon_counts)
fwrite(dict_dt, file = "Data/gene_dictionary.txt", sep = "\t")

# ==== Find the source of miRNAs ====
# Download miRNA data from mirBase
download.file(url = "ftp://mirbase.org/pub/mirbase/20/genomes/hsa.gff3",
              destfile = "mirbase_data_hsa.gff3")
mirbase <- fread("mirbase_data_hsa.gff3")
# Extract miRNA names
temp <- gsub(pattern = "Derives.*", replacement = "", x = mirbase$V9)
temp <- gsub(pattern = ".*Name=", replacement = "", x = temp)
temp <- gsub(pattern = ";", replacement = "", x = temp)

mirbase <-
    data.table(
        chr = mirbase$V1,
        start = mirbase$V4,
        end = mirbase$V5,
        name = temp,
        strand = mirbase$V7
    )

# Create a GRanges object
miRNA_gr <- makeGRangesFromDataFrame(
    df = mirbase,
    keep.extra.columns = F,
    seqnames.field = "chr",
    start.field = "start",
    end.field = "end",
    strand.field = "strand"
)
names(miRNA_gr) <- temp

# Save
saveRDS(miRNA_gr, "Human_miRNA_GRanges.rds")