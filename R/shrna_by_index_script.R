library(data.table)
library(parallel)
# library(fastmatch)
# library(cmapR)
# if (!require(biomaRt)) {
#   BiocManager::install("biomaRt", version = "3.8")
#   library(biomaRt)
# }
# library(HGNChelper)
# if (!require(Biostrings)) {
#   BiocManager::install("Biostrings", version = "3.8")
#   library(Biostrings)
# }
if (!require(onehot)) {
  install.packages("onehot")
  library(onehot)
}
library(stringr)
print(paste0("Total number of cores: ", detectCores()))

sh_map <- fread("/u/ftaj/anaconda3/envs/Drug_Response/Data/RNAi/Train_Data/shRNAmapping.csv")
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
ccle_shrna_seqs <- fread("/u/ftaj/anaconda3/envs/Drug_Response/Data/RNAi/Train_Data/ccle_shrna_seqs.txt")
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
dir.create("/u/ftaj/anaconda3/envs/Drug_Response/Data/RNAi/Train_Data/shRNA_by_index")

onehot_encoder <- onehot::onehot(data = as.data.frame(sh_map),
                                 max_levels = length(unique(sh_map$ENTREZ)))

dummification <- function(idx, encoder, shrna_data) {
  # for (idx in seq(1, nrow(temp), by = 100000)) {
    # cur_sub <- sh_map[cell_line == line]
  cur_sub <- shrna_data[idx:(idx+100000-1),]
  print(paste0("Encoding ", as.character(idx)))
  onehot_results <- predict(encoder, cur_sub)
  dim(onehot_results)
  head(onehot_results)
  rownames(onehot_results) <- cur_sub$shRNA
  onehot_results <- data.table(onehot_results, keep.rownames = T)
  colnames(onehot_results)[1] <- "shRNA"
  
  onehot_results <- onehot_results[, lapply(.SD, sum), by = shRNA, .SDcols = colnames(onehot_results)[-1]]
  fwrite(onehot_results, paste0("/u/ftaj/anaconda3/envs/Drug_Response/Data/RNAi/Train_Data/shRNA_by_index/", idx, "_",
                                idx+100000-1, ".txt"))
  # }  
}
# [-(1:230)]
mc_results <- mclapply(X = seq(1, nrow(temp), by = 100000)[1:4], FUN = dummification, encoder = onehot_encoder,
         shrna_data = temp, mc.cores = 2,
         mc.cleanup = T, mc.preschedule = F)
print(mc_results)
