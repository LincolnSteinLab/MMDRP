# Read_Achilles_DRIVE_RNAi.R
# install.packages("fastmatch")
library(data.table)
library(fastmatch)
library(cmapR)
if (!require(biomaRt)) {
  BiocManager::install("biomaRt", version = "3.8")
  library(biomaRt)
}
library(HGNChelper)
if (!require(Biostrings)) {
    BiocManager::install("Biostrings", version = "3.8")
    library(Biostrings)
}
# cur_map <- getCurrentHumanMap()
# saveRDS(cur_map, "Data/HGNC_Map.rds")
cur_map <- readRDS("Data/HGNC_Map.rds")

# ==== Expression Data ====
ccle_rna <- fread("Data/DepMap/20Q1/Cellular Models/CCLE_mRNA_expression.csv")
ccle_rna[1:5, 1:5]
dim(ccle_rna)
# ccle_drug_data <- fread("Data/DepMap/CCLE/CCLE_NP24.2009_Drug_data_2015.02.24.csv")
ccle_line_info <- fread("Data/DepMap/20Q1/sample_info.csv")

# Merge DepMap_ID and CCLE_Name with RNA data
ccle_rna_with_names <- merge(ccle_line_info[, c("DepMap_ID", "CCLE_Name")],
                             ccle_rna, by.x = "DepMap_ID", by.y = "V1")
# Get tumor type and subtype
ccle_sub_info <- ccle_line_info[match(ccle_rna_with_names$CCLE_Name, ccle_line_info$CCLE_Name),]
ccle_sub_info <- ccle_sub_info[, c("CCLE_Name", "disease", "disease_subtype")]

ccle_rna_with_names$cell_line <- gsub("_.*", "", ccle_rna_with_names$CCLE_Name)
ccle_sub_info$cell_line <- gsub("_.*", "", ccle_sub_info$CCLE_Name)
ccle_rna_with_names[, c("DepMap_ID", "CCLE_Name") := NULL]
ccle_sub_info[, CCLE_Name := NULL]

setcolorder(ccle_rna_with_names, c("cell_line",
                                   colnames(ccle_rna_with_names)[-ncol(ccle_rna_with_names)]))

dim(ccle_sub_info)
dim(ccle_rna_with_names)
all(ccle_rna_with_names$cell_line == ccle_sub_info$cell_line) # TRUE
# colnames(ccle_rna_with_names)[ncol(ccle_rna_with_names)]
length(unique(ccle_sub_info$disease))
unique(ccle_sub_info$disease)
unique(ccle_sub_info$disease_subtype)
anyNA(ccle_sub_info)

fwrite(ccle_sub_info, "Data/RNAi/Train_Data/ccle_line_info.txt")
fwrite(ccle_rna_with_names, "Data/RNAi/Train_Data/ccle_exp_data.txt")

ccle_long_exp <- melt.data.table(ccle_rna_with_names, id.vars = "cell_line",
                                 measure.vars = colnames(ccle_rna_with_names)[-1],
                                 value.name = "exp")
colnames(ccle_long_exp)[2] <- "gene_name"
fwrite(ccle_long_exp, "Data/RNAi/Train_Data/ccle_long_exp.txt")

# Get expression data gene sequences
ccle_rna_with_names <- fread("Data/RNAi/Train_Data/ccle_exp_data.txt")
anyDuplicated(colnames(ccle_rna_with_names))
cur_genes <- colnames(ccle_rna_with_names)[-1]
cur_genes <- gsub(".+ \\((ENSG\\d+)\\)", "\\1", cur_genes)
anyDuplicated(cur_genes)  # None
grch37 <- useMart(host = "http://grch37.ensembl.org",
                  biomart='ENSEMBL_MART_ENSEMBL', 
                  dataset='hsapiens_gene_ensembl')
biomaRt::searchAttributes(grch37, pattern = "gene")
# Get the transcripts of the current genes
if (!require(TxDb.Hsapiens.UCSC.hg38.knownGene)) {
  BiocManager::install("TxDb.Hsapiens.UCSC.hg38.knownGene")
  library(TxDb.Hsapiens.UCSC.hg38.knownGene)
}
if (!require(BSgenome.Hsapiens.UCSC.hg38)) {
  BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")
  library(BSgenome.Hsapiens.UCSC.hg38)
}
if (!require(AnnotationHub)) {
  BiocManager::install("AnnotationHub")
  library(AnnotationHub)
}

# Get gene boundaries from 
ahub <- AnnotationHub()
qh <- query(ahub, c("hg38", "transcript", "homo sapiens"))
genes <- qh[["AH5046"]]
anyDuplicated(genes$name)

hg19 <- BSgenome.Hsapiens.UCSC.hg19
# cdsBy(txdb, by = "tx", use.names = T)
tran_seqs <- extractTranscriptSeqs(hg19, txdb, use.names = T)
length(unique(names(tran_seqs)))
keytypes(txdb)
# Get ENTREZ IDs
gene_ids <- select(txdb, keys = names(tran_seqs), columns="GENEID", keytype="TXNAME")
length(unique(gene_ids$GENEID))
biomaRt::searchAttributes(grch37, pattern = "entrez")
# Translate to ENSG and ENST
ensembl_ids <- getBM(attributes = c("ensembl_transcript_id", "ensembl_gene_id", "transcript_length", "entrezgene"),
                     filters = "entrezgene", values = gene_ids$GENEID, mart = grch37)
# Get ENST of genes in CCLE
ccle_genes <- gsub(".+ \\((ENSG\\d+)\\)", "\\1", colnames(ccle_rna_with_names)[-1])
sum(ccle_genes %in% ensembl_ids$ensembl_gene_id) 
length(unique(ccle_genes))
length(unique(ensembl_ids$ensembl_gene_id))
length(unique(ensembl_ids$ensembl_transcript_id))
sum(duplicated(ensembl_ids$ensembl_gene_id))
ccle_match <- ensembl_ids[ensembl_ids$ensembl_gene_id %in% ccle_genes, ]
length(unique(ccle_match$ensembl_transcript_id))
length(unique(ccle_match$ensembl_gene_id))

length(unique(merge_seq$Annotation_Transcript))
merge_seq$Tumor_Sample_Barcode <- gsub("\\_.+", "", merge_seq$Tumor_Sample_Barcode)
length(unique(merge_seq$Tumor_Sample_Barcode))
length(unique(ccle_rna_with_names$cell_line))

sum(drive_cells %in% merge_seq$Tumor_Sample_Barcode) / length(drive_cells)

ensembl_trans <- Biostrings::readDNAStringSet("Data/RNAi/Train_Data/Homo_sapiens.GRCh37.cdna.all.fa")
names(ensembl_trans)[1]
tran_names <- gsub(" .*", "", names(ensembl_trans))
gene_names <- gsub(".*gene:(ENSG\\d+) .*", "\\1", names(ensembl_trans))

names(ensembl_trans) <- gsub("\\.\\d+", "", names(ensembl_trans))
sum(ensembl_ids$ensembl_transcript_id %in% tran_names) / length(ensembl_ids$ensembl_transcript_id)
sum(unique(gene_names) %in% ccle_genes)
# All mutated transcripts are available in the data
sum(unique(merge_seq$Annotation_Transcript) %in% tran_names) /length(unique(merge_seq$Annotation_Transcript))

names(ensembl_trans) <- tran_names
sub_trans <- ensembl_trans[names(ensembl_trans) %in% merge_seq$Annotation_Transcript]

merge_seq



promoters(genes)
transcripts(genes)
trans_seqs <- Views(hg19, genes$blocks)

txdb <- TxDb.Hsapiens.UCSC.hg19.knownGene
txdb

all_trans <- transcriptsBy(txdb, by = c("gene"))
all_trans <- transcripts(txdb)
head(all_trans)

gene_transcripts = getBM(attributes = c("ensembl_transcript_id", "ensembl_gene_id"),
                    filters = "ensembl_gene_id", values = cur_genes, mart = grch37)
anyDuplicated(gene_transcripts$ensembl_transcript_id)
transcript_seqs <- getSequence(id = gene_transcripts$ensembl_transcript_id,
                               type = "ensembl_transcript_id", seqType = "cdna",
                               mart = grch37)
dim(transcript_seqs)

getSequence()


# ==== CCLE Mutation Data ====
ccle_mutation <- fread("Data/RNAi/CCLE_mutation_data.csv")
unique(ccle_mutation$Variant_Classification)
# length(unique(ccle_mutation$Hugo_Symbol))
# length(unique(ccle_mutation$Annotation_Transcript))
cur_mutation_data <- ccle_mutation[, c("Variant_Classification", "Hugo_Symbol", "Tumor_Sample_Barcode")]
cur_mutation_data$cell_name <- gsub(pattern = "_.*", replacement = "",
                                    x = cur_mutation_data$Tumor_Sample_Barcode)
cur_mutation_data[, Tumor_Sample_Barcode := NULL]
cur_mutation_data <- unique(cur_mutation_data)
cur_mutation_data$Variant_Labels <- as.numeric(as.factor(cur_mutation_data$Variant_Classification))
cur_mutation_data[, Variant_Classification := NULL]
colnames(cur_mutation_data)[2] <- "cell_line"
fwrite(cur_mutation_data, "Data/RNAi/Train_Data/labeled_ccle_mut_data.txt")

dcast_mut_data <- dcast.data.table(cur_mutation_data,
                                   cell_line ~ Hugo_Symbol + Variant_Labels,
                                   drop = F, # Ensures all possible combinations are kept
                                   fill = 0, fun.aggregate = length)
sum(dcast_mut_data$A1CF_6)
fwrite(dcast_mut_data, "Data/RNAi/Train_Data/simple_ccle_mut_data.txt")

dcast_mut_data <- dcast.data.table(cur_mutation_data,
                                   cell_line + Hugo_Symbol ~ Variant_Labels,
                                   drop = F, # Ensures all possible combinations are kept
                                   fill = 0, fun.aggregate = length)
fwrite(dcast_mut_data, "Data/RNAi/Train_Data/long_ccle_mut_data.txt")


### Generate per gene features
length(unique(ccle_mutation$Hugo_Symbol))
length(unique(ccle_mutation$Annotation_Transcript))
# Check and update current HUGO symbols
cur_hgnc <- HGNChelper::checkGeneSymbols(x = ccle_mutation$Hugo_Symbol, map = cur_map)
ccle_mutation$updated_hugo <- cur_hgnc$Suggested.Symbol
# Get gene lengths from biomaRt
hugo_list <- unique(ccle_mutation$updated_hugo)
enst_list <- ccle_mutation$Annotation_Transcript
enst_list <- gsub("\\.\\d+", "", enst_list)
ccle_mutation$Annotation_Transcript <- enst_list


hugo_list <- sort(hugo_list)
nchar(hugo_list[1:5])
grch37 <- useMart(host = "http://grch37.ensembl.org",
                  biomart='ENSEMBL_MART_ENSEMBL', 
                  dataset='hsapiens_gene_ensembl')
biomaRt::searchAttributes(grch37, pattern = "length")
biomaRt::searchAttributes(grch37, pattern = "sequence")
# t1 <- getSequence(id = gene_coords$ensembl_transcript_id[2], type = "ensembl_transcript_id", seqType = "coding", mart = grch37)
# t2 <- getSequence(id = gene_coords$ensembl_transcript_id[2], type = "ensembl_transcript_id", seqType = "transcript_exon_intron", mart = grch37)
# t3 <- getSequence(id = gene_coords$ensembl_transcript_id[2], type = "ensembl_transcript_id", seqType = "gene_exon_intron", mart = grch37)
# t4 <- getSequence(id = gene_coords$ensembl_transcript_id[2], type = "ensembl_transcript_id", seqType = "gene_exon", mart = grch37)
# t5 <- getSequence(id = gene_coords$ensembl_transcript_id[2], type = "ensembl_transcript_id", seqType = "cdna", mart = grch37)
# t6 <- getSequence(id = gene_coords$ensembl_transcript_id[2], type = "ensembl_transcript_id", seqType = "3utr", mart = grch37)
# t7 <- getSequence(id = gene_coords$ensembl_transcript_id[2], type = "ensembl_transcript_id", seqType = "5utr", mart = grch37)
# 
# t1$coding == t2$transcript_exon_intron
# nchar(t2$transcript_exon_intron)
# nchar(t4$gene_exon)
# nchar(t5$cdna)
# nchar(t6$`3utr`) + nchar(t1$coding) + nchar(t7$`5utr`)

gene_coords = getBM(attributes = c("ensembl_transcript_id", "transcript_length"),
                    filters = "ensembl_transcript_id", values = enst_list, mart = grch37)
gene_coords <- as.data.table(gene_coords)

transcript_seqs <- getSequence(id = gene_coords$ensembl_transcript_id,
                               type = "ensembl_transcript_id", seqType = "cdna",
                               mart = grch37)
transcript_seqs <- as.data.table(transcript_seqs)
# Save
fwrite(transcript_seqs, "Data/RNAi/Train_Data/transcript_seqs.txt")
transcript_seqs <- fread("Data/RNAi/Train_Data/transcript_seqs.txt")
colnames(transcript_seqs)
head(transcript_seqs$ensembl_transcript_id)
head(ccle_mutation$Annotation_Transcript)
# Merge sequences with mutational data
merge_seq <- merge(ccle_mutation[, c("Annotation_Transcript",
                                     "cDNA_Change", "Variant_Type",
                                     "Tumor_Sample_Barcode")],
                   transcript_seqs,
                   by.x = "Annotation_Transcript", by.y = "ensembl_transcript_id")
colnames(merge_seq)[5] <- "ref_cdna"
merge_seq$alt_cdna <- merge_seq$ref_cdna
dim(merge_seq)

 library(stringr)
mutations <- merge_seq$cDNA_Change
vartypes <- merge_seq$Variant_Type
cur_mutations <- gsub("c\\.", "", mutations[vartypes == "SNP"])
snps <- gsub("(\\d+)(\\w)(>)(\\w)", "\\1 \\2 \\4", cur_mutations)
snp_split <- str_split(snps, " ", simplify = T)
subseq(merge_seq[vartypes == "SNP"]$alt_cdna,
          as.integer(snp_split[, 1]), as.integer(snp_split[, 1])) <- snp_split[, 3]
sum(merge_seq$ref_cdna == merge_seq$alt_cdna) / nrow(merge_seq)

cur_mutations <- gsub("c\\.", "", mutations[vartypes == "INS"])
ins <- gsub("(\\d+)(\\_)(\\d+)ins(\\w+)", "\\1 \\3 \\4", cur_mutations)
ins_split <- str_split(ins, " ", simplify = T)
# For insertion in a sequence, the second position is 1 minus the first position
subseq(merge_seq[vartypes == "INS"]$alt_cdna,
          as.integer(ins_split[, 1]), as.integer(ins_split[, 1])-1) <- ins_split[, 3]
sum(merge_seq$ref_cdna == merge_seq$alt_cdna) / nrow(merge_seq)

cur_mutations <- gsub("c\\.", "", mutations[vartypes == "DEL"])
dels <- gsub("(\\d+)\\_+(\\d+)*del\\w+", "\\1 \\2", cur_mutations)
dels <- gsub("(\\d+)del\\w+", "\\1", dels)

dels_split <- str_split(dels, " ", simplify = T)
# For deletion in a sequence, copy the first position if there only a single deletion
dels_split[dels_split[, 2] == "", 2] <- dels_split[dels_split[, 2] == "", 1]
subseq(merge_seq[vartypes == "DEL"]$alt_cdna,
       as.integer(dels_split[, 1]), as.integer(dels_split[, 2])) <- ""
sum(merge_seq$ref_cdna == merge_seq$alt_cdna) / nrow(merge_seq)

# The remaining variants are outside the transcript, but may affect the transcript indirectly,
# e.g. splice site variants
anyNA(merge_seq$cDNA_Change)
sum(merge_seq$cDNA_Change == "")
ccle_mutation[cDNA_Change == ""]

# Remove invalid cDNA changes that result in empty alt seqs
sum(merge_seq$alt_len == 0)
sum(merge_seq$cDNA_Change == "")
merge_seq <- merge_seq[cDNA_Change != ""]
merge_seq[, ref_len := nchar(ref_cdna), by = "Annotation_Transcript"]
colnames(merge_seq)
# Save 
fwrite(merge_seq, "Data/RNAi/Train_Data/transcripts_alt_and_ref.txt")

merge_seq <- fread("Data/RNAi/Train_Data/transcripts_alt_and_ref.txt")
colnames(merge_seq)
(sum(unique(merge_seq[, c("Annotation_Transcript", "ref_len")])$ref_len) / 8) * 32 * 4 / 1e6

rm(list = c("dels", "dels_split", "ins", "ins_split", "snps", "snp_split"))

merge_seq <- fread("Data/RNAi/Train_Data/transcripts_alt_and_ref.txt")
colnames(merge_seq)
length(unique(merge_seq[,Tumor_Sample_Barcode]))

# Save only the transcripts as well
library(data.table)
alt_transcripts <- unique(merge_seq[, alt_cdna])
ref_transcripts <- unique(merge_seq[, ref_cdna])
length(alt_transcripts)
length(ref_transcripts)
all_transcripts <-
  rbindlist(list(data.table(alt_transcripts), data.table(ref_transcripts)), use.names = F)
dim(all_transcripts)
colnames(all_transcripts) <- "transcript"
all_transcripts <- unique(all_transcripts)
all_transcripts[, len := nchar(transcript)]
# Remove longest 1% transcripts
quantile(unique(all_transcripts$len), 0.99)
all_transcripts <- all_transcripts[len < 20326]
# Order by increasing length (want to train on smaller sequences first)
data.table::setorder(all_transcripts, -len)
# all_transcripts$len

library(data.table)
fwrite(unique(all_transcripts[, 'transcript']), "Data/RNAi/Train_Data/all_transcript_seqs.txt")
all_transcripts <- fread("Data/RNAi/Train_Data/all_transcript_seqs.txt")
all_transcripts

nrow(all_transcripts)

unique(merge_seq[is.nan(cDNA_Change)])
which(is.na(merge_seq$alt_cdna))
merge_seq[34,]
merge_seq[Annotation_Transcript == "ENST00000001146"]
sum(merge)
sum(merge_seq$ref_len == 0)
colnames(merge_seq)
length(unique(merge_seq[Variant_Type == "SNP"]$Annotation_Transcript))
merge_seq[, ref_len := nchar(ref_cdna), by = "Annotation_Transcript"]
merge_seq[, alt_len := nchar(alt_cdna), by = "Annotation_Transcript"]
merge_seq[Variant_Type == "SNP", "Annotation_Transcript"]
sum(unique(merge_seq[, c("Annotation_Transcript", "ref_len")])$ref_len)
max(unique(merge_seq[, c("Annotation_Transcript", "ref_len")])$ref_len)
min(unique(merge_seq[, c("Annotation_Transcript", "ref_len")])$ref_len)
median(unique(merge_seq[, c("Annotation_Transcript", "ref_len")])$ref_len)
plot(unique(merge_seq[, c("Annotation_Transcript", "ref_len")])$ref_len)

sum(nchar(merge_seq$alt_cdna))


colnames(merge_seq)
unique(ccle_mutation$Variant_Type)

mean(gene_coords$transcript_length)
median(gene_coords$transcript_length)
max(gene_coords$transcript_length)
min(gene_coords$transcript_length)

gene_coords[, size := as.integer(mean(size)), by = "hgnc_symbol"]
# Update CCLE mutation data
ccle_mutation$Annotation_Transcript <- gsub("\\.\\d+", "", ccle_mutation$Annotation_Transcript)
updated_ccle_mutation <- merge(ccle_mutation, gene_coords,
                          by.x = "Annotation_Transcript",
                          by.y = "ensembl_transcript_id")

ccle_mut <- updated_ccle_mutation  # make copy
# Number of silent mutations per gene divided by length
all_var_types <- unique(ccle_mut$Variant_Classification)
unique(ccle_mut$Variant_Type)
# c("Missense_Mutation", "Nonsense_Mutation", "Splice_Site", "Frame_Shift_Del", "Frame_Shift_Ins")
for (var_type in c("SNP", "DEL", "INS")) {
  ccle_mut[Variant_Type == var_type,
           paste0("Num_", var_type) := nrow(.SD),
           by = c("Annotation_Transcript", "Tumor_Sample_Barcode")]
}
ncol(ccle_mut)
# Replace NAs with 0
ccle_mut[, 34:36][is.na(ccle_mut[, 34:36, with = F])] <- 0

mut_sub <- as.data.table(ccle_mut[, c("Annotation_Transcript", "Tumor_Sample_Barcode", "Num_SNP", "Num_DEL", "Num_INS")])
mut_sub$cell_line <- gsub("\\_.+", "", mut_sub$Tumor_Sample_Barcode)
setkey(mut_sub, "cell_line")

fwrite(mut_sub[, -2], "ccle_muts_by_type.txt")

# Get original transcript sequence
human <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
listMarts(host = "http://grch37.ensembl.org")
listEnsemblArchives()
listDatasets(human)

listDatasets(grch37)

biomaRt::searchAttributes(human, pattern = "length")
biomaRt::searchAttributes(human, pattern = "transcript")


# ==== DRIVE Data ====
drive_count <- readRDS("Data/RNAi/drive-raw-data/DriveCountData.RDS")
drive_count <- as.data.table(drive_count)
unique(drive_count$CLEANNAME)
drive_cells <- toupper(unique(drive_count$CLEANNAME))

shrna_map <- readRDS("Data/RNAi/drive-raw-data/ShrnaGeneMap.RDS")
shrna_map <- as.data.table(shrna_map)
length(unique(drive_count$EXPERIMENT_ID))
length(unique(shrna_map$GENESYMBOLS))


all_merged <- merge(drive_count, shrna_map, by = "SEQ")

# Get DRIVE cell line mutation data from CCLE
drive_mutation <- cur_mutation_data[cell_name %in% drive_cells]

# Get DRIVE cell line expression data from CCLE
ccle_rna_with_names$CCLE_Name
drive_rna <- ccle_rna_with_names[CCLE_Short_Name %in% drive_cells]
dim(drive_rna)

# Order shRNA screens with mutation and expression data
all_merged[PLASMID_COUNT > 0, num_shrna_exp := .N, by = "EXPERIMENT_ID"]

temp_exp <- all_merged[EXPERIMENT_ID == 1004182]
temp <- temp_exp[SAMPLE_COUNT > 1000 & !is.na(GENESYMBOLS)]
length(unique(temp$GENESYMBOLS))
all_merged

all_merged[POOL == "BGPD" & !is.na(PLASMID_COUNT)]
drive_rna[1:5, 1:5]


# ==== Read LFCs ====
achilles_batch_1 <- fread("Data/RNAi/LFCs/achilles55kbatch1repcollapsedlfc.csv")
achilles_batch_2 <- fread("Data/RNAi/LFCs/achilles55kbatch2repcollapsedlfc.csv")
achilles_batch_3 <- fread("Data/RNAi/LFCs/achilles98krepcollapsedlfc.csv")
colnames(achilles_batch_1)[-1] <- gsub("_.*", "", colnames(achilles_batch_1)[-1])
colnames(achilles_batch_2)[-1] <- gsub("_.*", "", colnames(achilles_batch_2)[-1])
colnames(achilles_batch_3)[-1] <- gsub("_.*", "", colnames(achilles_batch_3)[-1])
all_achilles <- Reduce(function(...) merge(..., all = T, by = "V1"), list(achilles_batch_1, achilles_batch_2, achilles_batch_3))
dim(all_achilles)
all_molten <- melt(data = all_achilles, id.vars = "V1",
     measure.vars = colnames(all_achilles)[-1],
     variable.name = "cell_line", value.name = "lfc")
colnames(all_molten)[1] <- "shRNA"

anyNA(all_achilles)
anyNA(all_molten$shRNA)
anyNA(all_molten$cell_line)
anyNA(all_molten$lfc)
all_molten <- all_molten[!is.na(lfc),]
setkey(all_molten, "cell_line")
fwrite(all_molten, "Data/RNAi/Train_Data/achilles_shrna_cell_lfc.txt")
rm(list("achilles_batch_1", "achilles_batch_2", "achilles_batch_3"))

# Match cell line expression values
# ccle_rna_with_names[fmatch(all_molten$cell_line, ccle_rna_with_names$CCLE_Short_Name),]

# Read DRIVE data
drive_bgpd <- fread("Data/RNAi/LFCs/drivebgpdlfcmat.csv")
drive_a <- fread("Data/RNAi/LFCs/drivepoolalfcmat.csv")
drive_b <- fread("Data/RNAi/LFCs/drivepoolblfcmat.csv")

dim(drive_bgpd)
dim(drive_a)
dim(drive_b)
colnames(drive_bgpd)[-1] <- gsub("_.*", "", colnames(drive_bgpd)[-1])
colnames(drive_a)[-1] <- gsub("_.*", "", colnames(drive_a)[-1])
colnames(drive_b)[-1] <- gsub("_.*", "", colnames(drive_b)[-1])

all_drive <- Reduce(function(...) merge(..., all = T, by = "V1"), list(drive_bgpd, drive_a, drive_b))
dim(all_drive)

all_drive[, c("U87MG.y", "U87MG.x")]
drive_molten <- melt(data = all_drive, id.vars = "V1",
                   measure.vars = colnames(all_drive)[-1],
                   variable.name = "cell_line", value.name = "lfc")
drive_molten <- drive_molten[!is.na(lfc),]
drive_molten$cell_line <- gsub("\\.x", "", drive_molten$cell_line)
unique(drive_molten$cell_line)
drive_molten$cell_line <- gsub("\\.y", "", drive_molten$cell_line)
unique(drive_molten$cell_line)
unique(drive_molten[, c("V1", "cell_line")])
# Average lfc by cell line for each shRNA
drive_molten[, avg_lfc := mean(lfc), by = c("V1", "cell_line")]
drive_molten[, lfc := NULL]
colnames(drive_molten) <- c("shRNA", "cell_line", "lfc")
drive_molten <- unique(drive_molten)
setkey(drive_molten, "cell_line")

fwrite(drive_molten, "Data/RNAi/Train_Data/drive_shrna_cell_lfc.txt")


# Read Marcotte Data
marcotte <- fread("Data/RNAi/LFCs/Marcotte_LFC_matrix.csv")
dim(marcotte)
colnames(marcotte)[-1] <- gsub("_.*", "", colnames(marcotte)[-1])

molten_marcotte <- melt(data = marcotte, id.vars = "V1",
                       measure.vars = colnames(marcotte)[-1],
                       variable.name = "cell_line", value.name = "lfc")
colnames(molten_marcotte)[1] <- "shRNA"
unique(molten_marcotte[, c("shRNA", "cell_line")])
fwrite(molten_marcotte, "Data/RNAi/Train_Data/marcotte_shrna_cell_lfc.txt")


# ==== Batch and Match Training Data ====
ccle_exp <- fread("Data/RNAi/Train_Data/ccle_exp_data.txt")
ccle_long_exp <- fread("Data/RNAi/Train_Data/ccle_long_exp.txt")
typeof(ccle_exp$`TSPAN6 (ENSG00000000003)`[1])

min(ccle_exp)
max(ccle_exp)

ccle_mut <- fread("Data/RNAi/Train_Data/simple_ccle_mut_data.txt")

achilles <- fread("Data/RNAi/Train_Data/achilles_shrna_cell_lfc.txt")
achilles[, sh_length := nchar(shRNA), by = c("cell_line", "lfc")]
unique(achilles$sh_length)

achilles_cells <- unique(achilles$cell_line)
ccle_cells <- unique(ccle_long_exp$cell_line)
sum(achilles_cells %in% ccle_cells)
achilles_cells[!(achilles_cells %in% ccle_cells)]
rm(list=c("ccle_long_exp", "achilles"))
drive <- fread("Data/RNAi/Train_Data/drive_shrna_cell_lfc.txt")
drive_cells <- unique(drive$cell_line)
sum(drive_cells %in% ccle_cells)
drive_cells[!(drive_cells %in% ccle_cells)]
rm(drive)

marcotte <- fread("Data/RNAi/Train_Data/marcotte_shrna_cell_lfc.txt")
marcotte_cells <- unique(marcotte$cell_line)
sum(marcotte_cells %in% ccle_cells)
marcotte_cells[!(marcotte_cells %in% ccle_cells)]

shared_cells <- unique(c(achilles_cells, drive_cells, marcotte_cells))

# Subset the expression data based on these cells
ccle_exp <- fread("Data/RNAi/Train_Data/ccle_exp_data.txt")
achilles_exp <- ccle_exp[cell_line %in% achilles_cells]
dim(achilles_exp)
fwrite(achilles_exp, "Data/RNAi/Train_Data/achilles_exp.txt")
rm(achilles_exp)

drive_exp <- ccle_exp[cell_line %in% drive_cells]
dim(drive_exp)
fwrite(drive_exp, "Data/RNAi/Train_Data/drive_exp.txt")
rm(drive_exp)

marcotte_exp <- ccle_exp[cell_line %in% marcotte_cells]
dim(marcotte_exp)
fwrite(marcotte_exp, "Data/RNAi/Train_Data/marcotte_exp.txt")






# Batch 1 million training cases
for (sub in seq(1, nrow(achilles) - 1e6, by = 1e6)) {
  print(sub)
  cur_sub <- achilles[sub:(sub + 1e6),]
  # Match cell line exp
  exp_sub <- ccle_exp[fmatch(cur_sub$cell_line, ccle_exp$cell_line),]
}
achilles


# ==== Merge all shRNA sequences for training, subset to match sequencing data ====
library(data.table)
# Consider sequencing data the reference with which everything is matched
merge_seq <- fread("Data/RNAi/Train_Data/transcripts_alt_and_ref.txt")
# Subset only cell lines available in the CCLE sequencing data
ccle_lines <- unique(merge_seq$Tumor_Sample_Barcode)
ccle_data <- fread("Data/RNAi/Train_Data/ccle_exp_data.txt")
# length(ccle_rna_with_names$cell_line)
ccle_data <- ccle_data[cell_line %in% ccle_lines]
# Update CCLE lines with those that have both sequencing and expression data
ccle_lines <- ccle_data$cell_line

achilles <- fread("Data/RNAi/Train_Data/achilles_shrna_cell_lfc.txt")
achilles <- achilles[cell_line %in% ccle_lines]
length(unique(achilles$cell_line))
drive <- fread("Data/RNAi/Train_Data/drive_shrna_cell_lfc.txt")
drive <- drive[cell_line %in% ccle_lines]
length(unique(drive$cell_line))
marcotte <- fread("Data/RNAi/Train_Data/marcotte_shrna_cell_lfc.txt")
marcotte <- marcotte[cell_line %in% ccle_lines]
length(unique(marcotte$cell_line))

# Extract unique sequences for autoencoder training
all_shrna_seqs <- rbindlist(list(achilles[,'shRNA'], drive[,'shRNA'], marcotte[,'shRNA']))
all_shrna_seqs <- unique(all_shrna_seqs)

# Save
fwrite(all_shrna_seqs, "Data/RNAi/Train_Data/all_shrna_seqs.txt")


# Extract all info for matching with sequencing and expression data
ccle_shrna_seqs <- rbindlist(list(achilles, drive, marcotte))
ccle_shrna_seqs <- unique(ccle_shrna_seqs)

# Save CCLE ShRNA
fwrite(ccle_shrna_seqs, "Data/RNAi/Train_Data/ccle_shrna_seqs.txt")
fread()

# Separate each cell line shRNA data
dir.create("Data/RNAi/Train_Data/shRNA_by_line")
all_lines <- unique(ccle_shrna_seqs$cell_line)
for (line in all_lines) {
  cur_sub <- ccle_shrna_seqs[cell_line == line]
  fwrite(cur_sub, paste0("Data/RNAi/Train_Data/shRNA_by_line/", line, ".txt"))
}

# Save CCLE Expression Data
fwrite(ccle_data, "Data/RNAi/Train_Data/ccle_sub_exp.txt")

rm(list = c("achilles", "drive", "marcotte"))
# Get an index in the 'all_transcript_seqs' files for every cell line for easier subsetting
# All cell lines should have the same number of transcripts; those that don't have a mutated
# transcript must be assigned the reference sequence
merge_sub <- merge_seq[Variant_Type == "SNP"]

length(merge_sub[Tumor_Sample_Barcode == 'OELE']$alt_cdna)
length(merge_sub[Tumor_Sample_Barcode == 'LN18']$alt_cdna)
colnames(merge_sub)
head(merge_sub$Tumor_Sample_Barcode)
colnames(ccle_data)
all_transcripts <- fread("Data/RNAi/Train_Data/all_transcript_seqs.txt")

all_reference <- unique(merge_sub[, c("Annotation_Transcript", "ref_cdna")])
nrow(all_reference)
merge_sub <- merge_sub[ref_len < 20326]
library(fastmatch)
# Find indices of reference/normal transcripts in 'all_transcripts'
ref_idx <- fmatch(x = all_reference$ref_cdna, table = all_transcripts$transcript)
ref_idx <- data.table(ref_id = all_reference$Annotation_Transcript, ref_idx = ref_idx)

dir.create("Data/RNAi/Train_Data/Cell_Line_Indices")
for (line in ccle_lines) {
  # Find the indices of the altered sequences in 'all_transcripts'
  cur_idx <- fmatch(x = merge_sub[Tumor_Sample_Barcode == line]$alt_cdna,
                table = all_transcripts$transcript)
  cur_idx <- data.table(alt_id = merge_sub[Tumor_Sample_Barcode == line]$Annotation_Transcript,
                        alt_idx = cur_idx)
  ref_copy <- ref_idx
  # Merge and replace transcript ids in the reference; this now represents the indices of
  # transcripts in 'all_transcripts' that are relevant to this cell line
  matched_idx <- fmatch(cur_idx$alt_id, ref_idx$ref_id)
  ref_copy[matched_idx,] <- cur_idx
  
  fwrite(ref_copy, paste0("Data/RNAi/Train_Data/Cell_Line_Indices/", line, ".txt"))
}
# fread("Data/RNAi/Train_Data/Cell_Line_Indices/22RV1.txt")

line_name_indices <- fread("Data/RNAi/Train_Data/ccle_exp_data.txt")
colnames(line_name_indices)[1]
line_name_indices <- data.table(seq_along(line_name_indices$cell_line),
                                line_name_indices$cell_line)
colnames(line_name_indices)[1] <- "index"
fwrite(line_name_indices, "Data/RNAi/Train_Data/ccle_name_index.txt")

colnames(merge_seq)
fmatch(unique(merge_seq$Annotation_Transcript), merge_seq$Annotation_Transcript)
sum(merge_seq[fmatch(unique(merge_seq$Annotation_Transcript), merge_seq$Annotation_Transcript)]$ref_len) / 8


# ==== shRNA mapping ====
library(stringr)
sh_map <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/RNAi/Train_Data/shRNAmapping.csv")
sh_map[`Gene Symbol` %like% "NO_CURRENT"]$`Gene Symbol` <- "NO_CURRENT"
sh_map[`Gene Symbol` %like% "NO_CURRENT"]$`Gene ID` <- "NO_CURRENT"
sh_map[`Gene Symbol` %like% "-"]$`Gene Symbol`[781]
sh_map[`Gene Symbol` %like% "-"]$`Gene ID`[781]
# colnames(sh_map) <- c("shRNA", "HGNC", "ENTREZ")
sh_map <- sh_map[, c(1,3)]

# sh_split <- str_split(sh_map$`Gene Symbol`, "-", simplify = T)
# head(sh_split)
# sh_map$`Gene ID`
# 
# temp <- sh_map[, strsplit(as.character(`Gene Symbol`), ",", fixed=TRUE),
#                      by = .(`Barcode Sequence`, `Gene Symbol`)][, `Gene Symbol` := NULL][
#                        , setnames(.SD, "Barcode Sequence", "Gene Symbol")]


# length(unique(sh_map$`Gene Symbol`))
# length(unique(sh_map$`Barcode Sequence`))
# anyDuplicated(sh_map$`Barcode Sequence`)
# which(duplicated(sh_map$`Barcode Sequence`))
# sh_map[c(28,29),]

colnames(sh_map) <- c("shRNA", "ENTREZ")
# sh_long <- melt(data = sh_map, id.vars = c("shRNA", "Gene"))
# sh_long <- dcast.data.table(sh_map, formula = shRNA ~ Gene, fill = 0,
#                             fun.aggregate = function(x) {1L})
# dim(sh_long)
# (sh_long[1:5, 2:5])
# 
# sum(sh_long[1,2:5000])

# Create a one-hot vector
# library(caret)
# Pair shRNA sequences with one-hot encoded gene target vector
# Each shRNA sequence will have a ~22000 length vector indicating its target
ccle_shrna_seqs <- fread("Data/RNAi/Train_Data/ccle_shrna_seqs.txt")
setkey(ccle_shrna_seqs, shRNA)
ccle_shrna_seqs$INDEX <- 1:nrow(ccle_shrna_seqs)
setkey(sh_map, shRNA)

length(unique(ccle_shrna_seqs$shRNA))
sh_map <- sh_map[shRNA %in% unique(ccle_shrna_seqs$shRNA)]
temp <- merge(ccle_shrna_seqs[, c(1,4)], sh_map, by = "shRNA", allow.cartesian = TRUE)
temp <- unique(temp)

# anyDuplicated(temp$INDEX)

# install.packages("onehot")
library(onehot)

# cur_sub$ENTREZ <- as.factor(cur_sub$ENTREZ)
# cur_sub$shRNA <- as.factor(cur_sub$shRNA)
# class(cur_sub$ENTREZ)
sh_map$ENTREZ <- as.factor(sh_map$ENTREZ)
sh_map$shRNA <- as.factor(sh_map$shRNA)
class(sh_map$ENTREZ)

# Separate into files by indices
# cur_dummy <- dummyVars(formula = '~ ENTREZ', data = sh_map,
#                        fullRank = T, sep = "_", levelsOnly = F)

onehot_encoder <- onehot::onehot(data = as.data.frame(sh_map),
                                 max_levels = length(unique(sh_map$ENTREZ)))
options(scipen=999)
dir.create("Data/RNAi/Train_Data/shRNA_by_index")

# all_results <- onehot_results
for (idx in seq(1, nrow(temp), by = 100000)[-(1:230)]) {
  # cur_sub <- sh_map[cell_line == line]
  cur_sub <- temp[idx:(idx+100000-1),]
  onehot_results <- predict(onehot_encoder, cur_sub)
  # dim(onehot_results)
  rownames(onehot_results) <- cur_sub$shRNA
  onehot_results <- data.table(onehot_results, keep.rownames = T)
  colnames(onehot_results)[1] <- "shRNA"
  
  onehot_results <- onehot_results[, lapply(.SD, sum), by = shRNA, .SDcols = colnames(onehot_results)[-1]]
  
  fwrite(onehot_results, paste0("Data/RNAi/Train_Data/shRNA_by_index/", idx, "_",
                         idx+100000-1, ".txt"))
}
library(data.table)
all_shrna_files <- list.files("Data/RNAi/Train_Data/shRNA_by_index/", full.names = T)
all_shrna_indices <- fread(all_shrna_files[1])

for (i in 2:length(all_shrna_files)) {
  cur_file <- fread(all_shrna_files[i])
  all_shrna_indices <- rbindlist(list(all_shrna_indices, cur_file))
}
all_shrna_indices <- unique(all_shrna_indices)

fwrite(x = all_shrna_indices, "Data/RNAi/Train_Data/all_shrna_indices.txt")

# dool <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/RNAi/Train_Data/shRNA_by_index/1_100000.txt")
# dim(dool)
dir.create("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/RNAi/Train_Data/shRNA_by_index")
library(stringr)
sh_map <- fread("/Users/ftaj/OneDrive - University of Toronto/Drug_Response/Data/RNAi/shRNAmapping.csv")
sh_map[`Gene Symbol` %like% "NO_CURRENT"]$`Gene Symbol` <- "NO_CURRENT"
sh_map[`Gene Symbol` %like% "NO_CURRENT"]$`Gene ID` <- "NO_CURRENT"
sh_map[`Gene Symbol` %like% "-"]$`Gene Symbol`[781]
sh_map[`Gene Symbol` %like% "-"]$`Gene ID`[781]
colnames(sh_map) <- c("shRNA", "HGNC", "ENTREZ")

dummification <- function(dummifier, shrna_data, idx) {
  cur_sub <- temp[idx:(idx+100000-1),]
  cur_onehot <- data.frame(predict(cur_dummy, cur_sub))
  fwrite(cur_onehot, paste0("Data/RNAi/Train_Data/shRNA_by_index/", idx, "_",
                            idx+100000-1, ".txt"))
  
}

mclapply(X = seq(1, nrow(temp), by = 100000), FUN = dummification,
         mc.cores = 6, mc.cleanup = T)



ccle_shrna_seqs <- ccle_shrna_seqs[, 1:2]
ccle_shrna_seqs <- unique(ccle_shrna_seqs)
all_merger <- merge(ccle_shrna_seqs, sh_map, by = "shRNA")

length(unique(ccle_shrna_seqs$shRNA))
# Add cell line to sh map
temp <- unique(ccle_shrna_seqs[, 1:2])
temp <- unique(temp)
sh_map <- unique(sh_map)
# Put all genes targeted by an shRNA in the same column
sh_map[, all_genes := paste0(Gene, collapse = ","), by = shRNA]
which(sh_map$Gene != sh_map$all_genes)
sh_map[27:28,]
sh_map[, Gene := NULL]
sh_map <- unique(sh_map)

sh_map$shRNA <- factor(sh_map$shRNA)
sh_map$Gene <- factor(sh_map$Gene)

sh_line_gene <- merge(temp, sh_map, by = "shRNA")
sh_line_gene$all_genes <- factor(sh_line_gene$all_genes)
sh_line_gene$shRNA <- factor(sh_line_gene$shRNA)

temp <- sh_line_gene[, strsplit(as.character(all_genes), ",", fixed=TRUE),
   by = .(shRNA, cell_line, all_genes)][, all_genes := NULL][
     , setnames(.SD, "shRNA", "cell_line", "Genes")]


merge_sub <- fread("Data/RNAi/Train_Data/transcripts_alt_and_ref.txt")
merge_sub <- merge_sub[Variant_Type == "SNP"]
colnames(merge_sub)
merge_sub <- merge_sub[ref_len < 20326]

# Separate each cell line shRNA data
line <- "TEN"
dir.create("Data/RNAi/Train_Data/shRNA_by_line_and_target")
all_lines <- unique(ccle_shrna_seqs$cell_line)
for (line in all_lines) {
  # cur_sub <- sh_map[cell_line == line]
  cur_sub <- sh_map
  sh_map$ENTREZ <- factor(sh_map$ENTREZ)
  sh_map$shRNA <- factor(sh_map$shRNA)
  sh_map[, HGNC := NULL]
  levels(sh_map$ENTREZ)
  cur_dummy <- dummyVars(formula = '~ ENTREZ', data = sh_map,
                         fullRank = T, sep = "_", levelsOnly = F)
  temp <- data.frame(predict(cur_dummy, sh_map[1:100,]))
  dim(temp)
  temp[1:5, 10:15]
  fwrite(cur_sub, paste0("Data/RNAi/Train_Data/shRNA_by_line/", line, ".txt"))
}

when <- data.frame(time = c("afternoon", "night", "afternoon",
                            "morning", "morning", "morning",
                            "morning", "afternoon", "afternoon"),
                   day = c("Mon", "Mon", "Mon",
                           "Wed", "Wed", "Fri",
                           "Sat", "Sat", "Fri"))

levels(when$time) <- list(morning="morning",
                          afternoon="afternoon",
                          night="night")
levels(when$day) <- list(Mon="Mon", Tue="Tue", Wed="Wed", Thu="Thu",
                         Fri="Fri", Sat="Sat", Sun="Sun")
model.matrix(~day, when)

interactionModel <- dummyVars(~ day + time + day:time,
                              data = when,
                              sep = ".")
predict(interactionModel, when[1:3,])
noNames <- dummyVars(~ day + time + day:time,
                     data = when,
                     levelsOnly = TRUE)




sh_onehot <- dummyVars(formula = " shRNA~ . ", data = sh_map)
temp <- data.table(predict(sh_onehot, sh_map[1:10000, ]), keep.rownames = T)
dim(temp)
temp[1:5, 1:5]
rownames(temp)[1:5]

View(sh_map)


