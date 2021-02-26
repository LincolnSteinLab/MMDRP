require(data.table)
d <- fread("/Users/ftaj/Downloads/primary-screen-replicate-collapsed-logfold-change.csv")
p <- fread("/Users/ftaj/Downloads/Table_S1_Sample_Information.csv")
c <- fread("/Users/ftaj/Downloads/sample_info-2.csv")

c <- c[, c("DepMap_ID", "CCLE_Name")]
cp <- merge(c, p, by.x = "CCLE_Name", by.y = "CCLE Code")
p
 
sum(cp$DepMap_ID %in% d$V1)/375
colnames(d)