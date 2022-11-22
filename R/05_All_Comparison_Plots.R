# Targeted_vs_Broad_Drugs.R
require(data.table)
setDTthreads(8)
require(ggplot2)
library(dplyr)
targeted_drugs <- c("Idelalisib", "Olaparib", "Venetoclax", "Crizotinib", "Regorafenib", 
                    "Tretinoin", "Bortezomib", "Cabozantinib", "Dasatinib", "Erlotinib", 
                    "Sonidegib", "Vandetanib", "Axitinib", "Ibrutinib", "Gefitinib", 
                    "Nilotinib", "Tamoxifen", "Bosutinib", "Pazopanib", "Lapatinib", 
                    "Dabrafenib", "Bexarotene", "Temsirolimus", "Belinostat", 
                    "Sunitinib", "Vorinostat", "Trametinib", "Fulvestrant", "Sorafenib", 
                    "Vemurafenib", "Alpelisib")

# mysubset <- function(df, ...) {
#   ssubset <- deparse(substitute(...))
#   subset(df, eval(parse(text = ssubset)))
# }

dodge2 <- position_dodge2(width = 0.9, padding = 0)
rsq <- function (x, y) cor(x, y, method = "pearson") ^ 2
rmse <- function(x, y) sqrt(mean((x - y)^2))
mae <- function(x, y) mean(abs(x - y))
# Moving average
ma <- function(x, n = 5) filter(x, rep(1 / n, n), sides = 2)

# install.packages("ggrepel")
# require(ggrepel)
my_plot_function <- function(avg_loss_by, sub_results_by, fill_by, data_order, bar_level_order,
                             facet_by, facet_level_order, facet_nrow = 2,
                             legend_title, y_lim = 0.1, y_lab = "Average MAE Loss",
                             plot_type = "bar_plot", target_sub_by = "Target Above 0.7",
                             cur_comparisons = NULL, test = "wilcox.test", paired = F,
                             calculate_avg_mae = T,
                             hide_outliers = F, step_increase = 0.1,
                             add_mean = F, min_diff = 0.05) {
  
  if (plot_type == "bar_plot") {
    if (calculate_avg_mae == F) {
      y_lab <- "Total RMSE Loss"
    }
    
    # all_results_long_copy <- data.table::melt(unique(all_results_copy[, c(avg_loss_by, "loss_by_config"), with = F]),
    #                                           id.vars = avg_loss_by)
    
    # all_results_long_copy[, cv_mean := mean(value), by = eval(avg_loss_by[!avg_loss_by %in% c("fold")])]
    # all_results_long_copy[, cv_sd := sd(value), by = eval(avg_loss_by[!avg_loss_by %in% c("fold")])]
    all_results_copy[, unique_sample := paste0(cpd_name, "_", cell_name)]
    shared_unique_samples <- Reduce(intersect, split(all_results_copy$unique_sample, all_results_copy$data_types))
    # uniqueN(shared_unique_samples)
    all_results_copy <- all_results_copy[unique_sample %in% shared_unique_samples]
    
    if (calculate_avg_mae == T) {
      all_results_copy[, cv_mean := mean(RMSELoss), by = eval(avg_loss_by[!avg_loss_by %in% c("fold")])]
      all_results_copy[, cv_sd := sd(RMSELoss), by = eval(avg_loss_by[!avg_loss_by %in% c("fold")])]
      cur_data <- unique(all_results_copy[, c(eval(avg_loss_by[!avg_loss_by %in% c("fold")]), "cv_mean", "cv_sd"), with = F])
    } else {
      # Calculate RMSE instead
      all_results_copy[, cv_mean := rmse(target, predicted), by = eval(avg_loss_by[!avg_loss_by %in% c("fold")])]
      cur_data <- unique(all_results_copy[, c(eval(avg_loss_by[!avg_loss_by %in% c("fold")]), "cv_mean"), with = F])
      # all_results_copy[, cv_sd := sd(RMSELoss), by = eval(avg_loss_by[!avg_loss_by %in% c("fold")])]
    }
    
    # ssubset <- deparse(substitute(sub_results_by))
    # baseline <- subset.data.table(all_results_long_copy, eval(parse(text = ssubset)))
    cur_data <- subset(cur_data, eval(sub_results_by))
    
    # baseline <- mysubset(all_results_long_copy, eval(sub_results_by))
    
    # Order bars the same as the error bars by changing data frame order via left join
    bar_level_df <- data.frame(x1 = bar_level_order)
    colnames(bar_level_df) <- as.character(fill_by)
    cur_data <- left_join(bar_level_df,  
                             cur_data,
                             by = as.character(fill_by))
    
    cur_data <- as.data.table(cur_data)
    if (y_lim == "full") {
      cur_ylim <- ylim(0, 1)
    } else {
      if (add_mean == T) {
        cur_ylim <- ylim(0, max(cur_data$cv_mean) + y_lim)
      } else {
        if (calculate_avg_mae == T) {
          cur_ylim <- ylim(0, max(cur_data$cv_mean) + max(cur_data$cv_sd) + y_lim)
        } else {
          cur_ylim <- ylim(0, max(cur_data$cv_mean) + y_lim)
        }
      }
    }
    
    # cur_data[, diff := abs(cv_mean - shift(cv_mean)), by = c("data_types")]
    p <- ggplot(cur_data)
    
    if (add_mean == T) {
      if (!is.null(facet_by)) {
        cur_data[, diff := abs(diff(cv_mean)), by = c("data_types", facet_by)]
        cur_data[, max_y := max(cv_mean), by = c("data_types", facet_by)]
        # "first higher" depends on the bar order given to the function (left to right)
        cur_data[, first_higher := ifelse(diff(cv_mean) < 0, T, F), by = c("data_types", facet_by)]
        
      } else {
        cur_data[, diff := abs(diff(cv_mean)), by = c("data_types")]
        cur_data[, max_y := max(cv_mean), by = "data_types"]
        cur_data[, first_higher := ifelse(diff(cv_mean) < 0, T, F), by = c("data_types")]
      }
      cur_data[, diff_too_small := ifelse(diff < min_diff, T, F)]
      
      p <- p + geom_text(aes(x=data_types,
                             label = round(cv_mean, 3), y = cv_mean),
                vjust = 1, hjust = -0.25, angle = 90, position = position_dodge2(width = .9)) +
      geom_bar(aes(x = data_types, y = max_y),
               stat = "identity", fill = "grey80", width = 0.4, position = "dodge") +
        # geom_text(data = cur_data[first_higher == T],
        geom_text(data = unique(cur_data[diff_too_small == F,
                                         c("data_types", "diff", "max_y", facet_by), with = F]),
                  aes(x = data_types, label = round(diff, 3), y = max_y),
                  vjust = 0.5, hjust = -0.25, angle = 45, color = "red")
        
    } else {
      if (calculate_avg_mae == T) {
        p <- p + geom_text(aes(x=factor(data_types, levels = bar_level_order),
                               label = round(cv_mean, 3), y = cv_mean + cv_sd),
                      vjust = 0.5, hjust = -0.25, angle = 90, position = position_dodge2(width = .9)) +
          geom_errorbar(aes(x=data_types,
                            y=cv_mean,
                            ymax=cv_mean + cv_sd, 
                            ymin=cv_mean - 0.01, col='black'),
                        linetype=1, show.legend = FALSE, position = dodge2, width = 0.9)
      } else {
        p <- p + geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean),
                           vjust = 0.5, hjust = -0.1, angle = 90, position = position_dodge2(width = .9))
          
      }
    }
      
      # Set bar order
    p <- p + geom_bar(mapping = aes(x = data_types, y = cv_mean,
                               fill = factor(eval(fill_by),
                                             levels = bar_level_order)),
                 stat = "identity", position="dodge", width = .9) +
    scale_x_discrete(limits = data_order) +
      scale_fill_discrete(name = legend_title) +
      scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                                   "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
      theme(text = element_text(size = 14),
            # axis.text.x = element_text(angle = 45, hjust = 1),
            axis.title.x = element_blank(),
            # legend.position = c(.85,.85),
            # legend.position=c(1,1),
            legend.direction="horizontal",
            legend.position="top",
            legend.justification="right",
            # legend.justification=c(1, 0),
            # plot.margin = unit(c(5, 1, 0.5, 0.5), "lines")
            ) +
      # theme_gray(base_size = 14) +
      ylab(y_lab) +
      # ylim(0, max(cur_data$cv_mean) + max(cur_data$cv_sd) + 0.05) +
      # ylim(0, max(cur_data$cv_mean) + max(cur_data$cv_sd) + y_lim) +
      # ylim(0, 1) +
      cur_ylim
    
    if (!is.null(facet_by)) {
      if (length(facet_by) > 1) {
        for (i in 1:length(facet_by)) {
          # If the length is more than 1, it is assumed that facet_level_order is a list
          set(cur_data, j = eval(facet_by)[i], value = factor(unlist(cur_data[, as.character(facet_by)[i], with = F]),
                                                              levels = facet_level_order[[i]]))
        }
      } else {
        set(cur_data, j = as.character(facet_by), value = factor(unlist(cur_data[, as.character(facet_by), with = F]),
                                                                 levels = facet_level_order))
        
      }
      p <- p + facet_wrap(facet_by,
                          ncol = length(facet_level_order),
                          nrow = facet_nrow)
    }
    
    return(p)
    
  } else if (plot_type == "box_plot" | plot_type == "violin_plot") {
    # Subset all results
    require(ggpubr)
    all_results_subset <- subset(all_results_copy, eval(sub_results_by))
    
    # Find unique samples shared between all given models that use different data types
    all_results_subset[, unique_sample := paste0(cpd_name, "_", cell_name)]
    
    if (uniqueN(all_results_subset$split_method) > 1) {
      all_results_subset[, unique_group := paste0(data_types, "_", split_method)]
      shared_unique_samples <- Reduce(intersect, split(all_results_subset$unique_sample, all_results_subset$unique_group))
    } else {
      shared_unique_samples <- Reduce(intersect, split(all_results_subset$unique_sample, all_results_subset$data_types))
    }
    # shared_unique_samples <- intersect(shared_unique_samples_by_data_types, shared_unique_samples_by_split_method)
    all_results_subset <- all_results_subset[unique_sample %in% shared_unique_samples]
    # uniqueN(all_results_subset)  # 2003392 for cell line and drug scaffold, 2191444 for all 3
    # all_results_subset[data_types == "PROT" & split_method == "Split By Cell Line"]
    # all_results_subset[data_types == "PROT" & split_method == "Split By Drug Scaffold"]
    # all_results_subset[data_types == "PROT" & split_method == "Split By Both Cell Line & Drug Scaffold"]
    if (length(target_sub_by) == 1) {
      all_results_sub_sub <- all_results_subset[TargetRange == target_sub_by]
    } else {
      all_results_sub_sub <- all_results_subset[TargetRange %in% target_sub_by]
    }
    # Order data for the facet
    all_results_sub_sub[, data_types := factor(data_types, levels = data_order)]
    all_results_sub_sub[, as.character(fill_by) := factor(unlist(all_results_sub_sub[, as.character(fill_by), with = F]),
                                                          levels = bar_level_order)]
    # all_results_sub_sub[, cv_mean := mean(RMSELoss), by = eval(avg_loss_by[!avg_loss_by %in% c("fold")])]
    # all_results_sub_sub[, cv_sd := sd(RMSELoss), by = eval(avg_loss_by[!avg_loss_by %in% c("fold")])]
    
    if (paired == T) {
      # Set order within each group by the unique ID, so that each group has the same order (for pairing?)
      setorder(all_results_sub_sub, data_types, unique_sample)
      # uniqueN(all_results_sub_sub) / 8
      
      # table(all_results_sub_sub[split_method == "Split By Drug Scaffold"]$data_types)
      # table(all_results_sub_sub$data_types)
      # # Confirm:
      # all_results_sub_sub[, head(unique_sample,2),by=data_types]
      
    }
    if (plot_type == "box_plot") {
      p <- ggboxplot(data = all_results_sub_sub, x = as.character(fill_by),
                     y = "RMSELoss", color = as.character(fill_by),
                     outlier.shape = ifelse(hide_outliers, NA, 19))
    } else {
      p <- ggviolin(data = all_results_sub_sub, x = as.character(fill_by),
                    y = "RMSELoss", color = as.character(fill_by),
                    draw_quantiles = 0.5,
                    # add = "mean_range"
                    # add = "boxplot"
                    )
    }
    p <- set_palette(p, "jco")
    p <- facet(p = p, facet.by = facet_by, nrow = 1, strip.position = "bottom") +
      theme(
        # axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.x = element_blank(),
        axis.ticks.x =  element_blank(),
        axis.title.x = element_blank(),
        text = element_text(size = 14)
            ) +
      labs(color = legend_title) +
      ylab(y_lab) +
      scale_y_continuous(breaks = seq(0, 1, 0.2))
      
    # if (plot_difference == T) {
    #   p + geom_text(data = unique(all_results_sub_sub[, c("data_types", "cv_mean", "cv_sd")]),
    #                      aes(x = data_types,
    #                            label = round(cv_mean, 3),
    #                            y = cv_mean + cv_sd),
    #             vjust = 0.5, hjust = -0.25, angle = 90, position = position_dodge2(width = .9))
      # p + annotate("text", x=0.1, y=0.1, label= "boat")
      
    # }
    
    if (!is.null(cur_comparisons)) {
      if (test == "ks.test") {
        # facet_by
        # all_results_sub_sub[eval(fill_by) == cur_comparisons[[i]][1]]$RMSELoss
        # bar_level_order
        all_stats <- vector("list", length = length(cur_comparisons))
        for (i in 1:length(cur_comparisons)) {
          all_results_sub_sub[eval(fill_by) %in% cur_comparisons[[i]], c("ks_D", "ks_p") := ks.test(x = .SD[eval(fill_by) == cur_comparisons[[i]][1]]$RMSELoss,
                                                             y = .SD[eval(fill_by) == cur_comparisons[[i]][2]]$RMSELoss,
                                                             alternative = "two.sided")[1:2], by = facet_by]
          cur_stat <- unique(all_results_sub_sub[!is.na(ks_D), c(facet_by, as.character(fill_by), "ks_D", "ks_p"), with = F])
          all_results_sub_sub$ks_D <- NULL
          all_results_sub_sub$ks_p <- NULL
          
          cur_stat[, ks_D := round(ks_D, 3)]
          cur_stat[, ks_p := round(ks_p, 3)]
          temp <-  melt(cur_stat, id.vars = c(facet_by, "ks_D", "ks_p"))
          
          dcast_formula <- as.formula(paste0(paste(facet_by, collapse=" + "), " + ks_D + ks_p ~ value"))
          final_stat <- dcast(temp, formula = dcast_formula)
          
          col_pos <- (length(facet_by) + 2 + 1)
          colnames(final_stat)[col_pos:(col_pos+1)] <- c("group1", "group2")
          all_stats[[i]] <- final_stat
        }
        all_stats <- rbindlist(all_stats)
        p <- p + stat_pvalue_manual(
          # data = all_stats, label = "KS D: {ks_D}", y.position = 1, step.increase = step_increase
          data = all_stats, label = "D = {ks_D}\np: {ks_p}", y.position = 1,
          step.group.by = facet_by[length(facet_by)], step.increase = step_increase,
        )
      } else {
        # Add pairwise comparisons p-value
        p <- p + stat_compare_means(comparisons = cur_comparisons,
                                method = test,
                                method.args = list(alternative = "two.sided"),
                                # label.y.npc = "top",
                                paired = paired)
        # compare_means(RMSELoss ~ data_types, data = all_results_sub_sub, group.by = c("data_types", "Targeted"))
      }
    }
  return(p)
  }
}


# Generate shared unique cell line and drug combinations between data specific models
# all_results <- fread("Data/all_results.csv")

temp <- all_results[merge_method == "Base Model" &
                      loss_type == "Base Model" &
                      drug_type == "Base Model" &
                      bottleneck != "With Data Bottleneck" &
                      nchar(data_types) <= 5]
table(temp$split_method)
all_results_subset <- subset(all_results,
                             (split_method == "Split By Cell Line" &
                                             merge_method == "Base Model" &
                                             loss_type == "Base Model" &
                                             drug_type == "Base Model" &
                                             bottleneck != "With Data Bottleneck" &
                                             nchar(data_types) <= 5))
all_results_subset$fold <- NULL
all_results_subset <- unique(all_results_subset)
# Find samples that are shared between all data types
all_results_subset[, unique_sample := paste0(cpd_name, "_", cell_name)]
shared_unique_samples <- Reduce(intersect, split(all_results_subset$unique_sample, all_results_subset$data_types))
all_results_subset <- all_results_subset[unique_sample %in% shared_unique_samples]
# all_results_shared_subset$unique_sample <- NULL
uniqueN(all_results_subset) / 8  # 125,212 samples in each model that are paired
table(all_results_subset$data_types)


# Save unique samples
fwrite(unique(all_results_subset[, c("cpd_name", "cell_name")]), "Data/shared_unique_combinations.csv")


# all_results <- fread("Data/all_results.csv")
# CTRPv2 Targeted vs Untargeted Therapeutics Distributions ====
drug_info <- fread("Data/DRP_Training_Data/CTRP_DRUG_INFO.csv")

# drug_info$gene_symbol_of_protein_target
# drug_info[target_or_activity_of_compound == "inhibitor of p53-MDM2 interaction"]
# table(targeted_drugs <- drug_info[gene_symbol_of_protein_target != "" & (cpd_status == "clinical" | cpd_status == "FDA")]$target_or_activity_of_compound)
# 
# # TODO: Get the list of targeted therapies from NCI-MATCH
# # Drugs with shared targets or activities
# drug_info[target_or_activity_of_compound == "inhibitor of BCL2, BCL-xL, and BCL-W"]
# drug_info[target_or_activity_of_compound == "inhibitor of BRAF"]
# drug_info[target_or_activity_of_compound == "inhibitor of cyclin-dependent kinases"]
# drug_info[target_or_activity_of_compound == "inhibitor of DNA methyltransferase"]
# drug_info[target_or_activity_of_compound == "inhibitor of EGFR and HER2"]
# drug_info[target_or_activity_of_compound == "inhibitor of gamma-secretase"]
# drug_info[target_or_activity_of_compound == "inhibitor of HDAC1, HDAC2, HDAC3, HDAC6, and HDAC8"]
# drug_info[target_or_activity_of_compound == "inhibitor of HMG-CoA reductase"]
# drug_info[target_or_activity_of_compound == "inhibitor of HSP90"]
# drug_info[target_or_activity_of_compound == "inhibitor of Janus kinases 1 and 2"]
# drug_info[target_or_activity_of_compound == "inhibitor of Janus kinase 2"]
# drug_info[target_or_activity_of_compound == "inhibitor of MEK1 and MEK2"]
# drug_info[target_or_activity_of_compound == "inhibitor of mTOR"]
# drug_info[target_or_activity_of_compound == "inhibitor of nicotinamide phosphoribosyltransferase"]
# drug_info[target_or_activity_of_compound == "inhibitor of PI3K and mTOR kinase activity"]
# drug_info[target_or_activity_of_compound == "inhibitor of polo-like kinase 1 (PLK1)"]
# drug_info[target_or_activity_of_compound == "inhibitor of VEGFRs"]
# drug_info[target_or_activity_of_compound == "inhibitor of VEGFRs, c-KIT, and PDGFR alpha and beta"]


table(drug_info$target_or_activity_of_compound)
# targeted_drugs <- drug_info[gene_symbol_of_protein_target != ""]$rn
ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")
# ctrp[ , mean_by_drug := mean(area_above_curve), by = "cpd_name"]
# ctrp[ , mean_by_cell := mean(area_above_curve), by = "ccl_name"]
# ctrp[, Dataset := "CTRPv2"]


# mean(ctrp[Targeted == T]$area_above_curve)
# mean(ctrp[Targeted == F]$area_above_curve)
ctrp[, Targeted := ifelse(cpd_name %in% targeted_drugs, "TargetedDrug", "UntargetedDrug")]

unique(ctrp[, c("cpd_name", "Targeted")])
unique(ctrp[Targeted == "TargetedDrug"]$cpd_name)
unique(ctrp[Targeted == "UntargetedDrug"]$cpd_name)
table(ctrp$Targeted)

ctrp[Targeted == "UntargetedDrug", Targeted := "Untargeted Drug"]
ctrp[Targeted == "TargetedDrug", Targeted := "Targeted Drug"]
colnames(ctrp)[8] <- "Drug Type"

ggplot(ctrp, aes(x = area_above_curve, colour = Targeted)) +
  # geom_density(bins=100) +
  geom_freqpoly(bins=100) +
  geom_vline(aes(xintercept = mean(area_above_curve)), color="blue", linetype="dashed", size=1) +
  geom_vline(aes(xintercept = median(area_above_curve)), color="blue", linetype="dashed", size=1) +
  scale_x_continuous(breaks=c(0, round(median(ctrp$area_above_curve), 3), round(mean(ctrp$area_above_curve), 3), 0.25, 0.5, 0.75, 1)) +
  annotate(x=mean(ctrp$area_above_curve), y=20000,label="CTRPv2 Mean",vjust=1.5,geom="text", angle = 90) + 
  annotate(x=median(ctrp$area_above_curve), y=20000,label="CTRPv2 Median",vjust=1.5,geom="text", angle = 90) + 
  ggtitle(label = "AAC Frequency Polygon for CTRPv2: Targeted vs Untargeted Drugs") +
  xlab("Area Above Curve") + ylab("Count")

ggsave(filename = "Plots/Dataset_Exploration/CTRP_AAC_Distribution_Targeted_vs_Untargeted.pdf")

ggplot(ctrp, aes(x = `Drug Type`, y = area_above_curve)) +
  geom_boxplot() +
  ylab("Area Above Curve")
  # theme(legend.position = c(.9,.85)) +
  # geom_vline(aes(xintercept = mean(area_above_curve)), color="blue", linetype="dashed", size=1) +
  # geom_vline(aes(xintercept = median(area_above_curve)), color="blue", linetype="dashed", size=1) +
  # scale_x_continuous(breaks=c(0, round(median(ctrp$area_above_curve), 3),
  #                             round(mean(ctrp$area_above_curve), 3),
  #                             0.25, 0.5, 0.75, 1)) +
  # scale_fill_discrete(name = "Drug Type:") +
  # annotate(x=mean(ctrp$area_above_curve), y=20000,label="CTRPv2 Mean",vjust=1.5,geom="text", angle = 90) + 
  # annotate(x=median(ctrp$area_above_curve), y=20000,label="CTRPv2 Median",vjust=1.5,geom="text", angle = 90) + 
  # ggtitle(label = "AAC Frequency Polygon for CTRPv2: Targeted vs Untargeted Drugs") +

ggsave(filename = "Plots/Dataset_Exploration/CTRP_AAC_Distribution_Targeted_vs_Untargeted_BoxPlot.pdf")


# ggplot(ctrp, aes(x = `Drug Type`, y = area_above_curve)) +
#   geom_violin(draw_quantiles = c(0.25, 0.5, 0.75)) +
#   # geom_boxplot() +
#   ylab("Area Above Curve")

require(ggpubr)
p <- ggviolin(data = ctrp, x = "Drug Type", y = "area_above_curve",
         add = "boxplot") +
  stat_compare_means(comparisons = list(c("Targeted Drug", "Untargeted Drug")),
                     method = "wilcox.test",
                     method.args = list(alternative = "two.sided")) +
    ylab("Area Above Curve") +
  xlab("") +
  scale_y_continuous(breaks = c(seq(0, 1, 0.2),
                                round(median(ctrp[`Drug Type` == "Targeted Drug"]$area_above_curve), 3),
                                round(median(ctrp[`Drug Type` == "Untargeted Drug"]$area_above_curve), 3))) +
  geom_hline(yintercept = median(ctrp[`Drug Type` == "Targeted Drug"]$area_above_curve), linetype = "dotted") +
  geom_hline(yintercept = median(ctrp[`Drug Type` == "Untargeted Drug"]$area_above_curve), linetype = "dotted") +
  theme(text = element_text(size = 18))
  
# p <- set_palette(p, "jco")
ggsave(plot = p, filename = "Plots/Dataset_Exploration/CTRP_AAC_Distribution_Targeted_vs_Untargeted_ViolinPlot.pdf")

## Validation Subset ====
require(data.table)
require(ggpubr)
ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")
drug_info <- fread("Data/DRP_Training_Data/CTRP_DRUG_INFO.csv")
shared_valid <- fread("Data/shared_unique_combinations.csv")
shared_valid[, unique_sample := paste0(cpd_name, "_", cell_name)]

ctrp[, Targeted := ifelse(cpd_name %in% targeted_drugs, "TargetedDrug", "UntargetedDrug")]
ctrp[Targeted == "UntargetedDrug", Targeted := "Untargeted Drug"]
ctrp[Targeted == "TargetedDrug", Targeted := "Targeted Drug"]
colnames(ctrp)[8] <- "Drug Type"

ctrp[, unique_sample := paste0(cpd_name, "_", ccl_name)]

ctrp_sub <- ctrp[unique_sample %in% shared_valid$unique_sample]

table(ctrp_sub$`Drug Type`)
table(ctrp$`Drug Type`)
# Subset CTRPv2 by shared validation samples
p <- ggviolin(data = ctrp_sub, x = "Drug Type", y = "area_above_curve",
              add = "boxplot") +
  stat_compare_means(comparisons = list(c("Targeted Drug", "Untargeted Drug")),
                     method = "wilcox.test",
                     method.args = list(alternative = "two.sided")) +
  ylab("Area Above Curve") +
  xlab("") +
  scale_y_continuous(breaks = c(seq(0, 1, 0.2),
                                round(median(ctrp[`Drug Type` == "Targeted Drug"]$area_above_curve), 3),
                                round(median(ctrp[`Drug Type` == "Untargeted Drug"]$area_above_curve), 3))) +
  geom_hline(yintercept = median(ctrp[`Drug Type` == "Targeted Drug"]$area_above_curve), linetype = "dotted") +
  geom_hline(yintercept = median(ctrp[`Drug Type` == "Untargeted Drug"]$area_above_curve), linetype = "dotted") +
  theme(text = element_text(size = 18))

ggsave(plot = p, filename = "Plots/Dataset_Exploration/CTRP_AAC_Distribution_Targeted_vs_Untargeted_Validation_Subset_ViolinPlot.pdf")

# Combine and compare both
ctrp_sub[, Type := "Validation Subset"]
ctrp[, Type := "All Training Data"]

both_combined <- rbindlist(list(ctrp, ctrp_sub))

require(rstatix)

ks_results_targeted <- ks.test(both_combined[DrugType == "Targeted Drug" & Type == "All Training Data"]$area_above_curve,
        both_combined[DrugType == "Targeted Drug" & Type == "Validation Subset"]$area_above_curve,
        alternative = "two.sided")
ks_results_untargeted <- ks.test(both_combined[DrugType == "Untargeted Drug" & Type == "All Training Data"]$area_above_curve,
        both_combined[DrugType == "Untargeted Drug" & Type == "Validation Subset"]$area_above_curve,
        alternative = "two.sided")

stat_test <- both_combined %>%
  group_by(DrugType) %>%
  wilcox_test(area_above_curve ~ Type,
              p.adjust.method = "fdr", alternative = "two.sided")

stat_test %>% adjust_pvalue(method = "fdr")

stat_test <- tibble::tribble(
  ~DrugType, ~group1, ~group2, ~`D`,
  "Targeted Drug", "All Training Data", "Validation Subset", round(ks_results_targeted$statistic, 5),
  "Untargeted Drug", "All Training Data", "Validation Subset", round(ks_results_untargeted$statistic, 5),
)


colnames(both_combined)[8] <- "DrugType"
p <- ggviolin(data = both_combined, x = "Type", y = "area_above_curve",
              add = "boxplot", facet.by = "DrugType") +
  stat_pvalue_manual(data = stat_test,
                     # label = "D Statistic",
                     label = "KS-test, D = {D}",
                     y.position = 1.1, ) +
  # stat_compare_means(comparisons = list(c("Validation Subset", "All Training Data")),
  #                    method = "wilcox.test",
  #                    method.args = list(alternative = "two.sided")) +
  ylab("Area Above Curve") +
  xlab("") +
  scale_y_continuous(breaks = c(seq(0, 1, 0.2),
                                round(median(ctrp[`Drug Type` == "Targeted Drug"]$area_above_curve), 3),
                                round(median(ctrp[`Drug Type` == "Untargeted Drug"]$area_above_curve), 3))) +
  geom_hline(yintercept = median(ctrp[`Drug Type` == "Targeted Drug"]$area_above_curve), linetype = "dotted", color = "red") +
  geom_hline(yintercept = median(ctrp[`Drug Type` == "Untargeted Drug"]$area_above_curve), linetype = "dotted", color = "red") +
  theme(text = element_text(size = 18))

ggsave(plot = p, filename = "Plots/Dataset_Exploration/CTRP_AAC_Distribution_Targeted_vs_Untargeted_Validation_Subset_Comparison_ViolinPlot.pdf")

# Load CV Fold Results ====
# Select per fold validation files
all_cv_files <- list.files("Data/CV_Results/", recursive = T,
                           pattern = ".*final_validation.*", full.names = T)
# ".+drug_.{3,5}_HyperOpt.+"
# bimodal_cv_files <- grep(pattern = ".+_.*drug_\\w{3,5}_HyperOpt.+", all_cv_files, value = T)
# bimodal_baseline_cv_files <- grep(pattern = ".+_.*drug_\\w{3,5}_HyperOpt.+MergeByConcat_RMSELoss_MorganDrugs.+", all_cv_files, value = T)
# trimodal_baseline_cv_files <- grep(pattern = ".+_.*drug_\\w{6,11}_HyperOpt.+MergeByConcat_RMSELoss_MorganDrugs.+", all_cv_files, value = T)

# cur_cv_files <- grep(pattern = ".ResponseOnly_.*drug_\\w{3,5}_.+", cur_cv_files, value = T)
# cur_cv_files <- grep(pattern = ".ResponseOnly_+drug_exp_HyperOpt.+", cur_cv_files, value = T)
# cur_cv_files_2 <- grep(pattern = ".Baseline_ElasticNet.+", all_cv_files, value = T)
# lineage_cv_files <- grep(pattern = ".LINEAGE.+", all_cv_files, value = T)
# bottleneck_cv_files <- grep(pattern = ".WithBottleNeck.+", all_cv_files, value = T)
# final_cv_files <- c(bimodal_cv_files, cur_cv_files_2)
# final_cv_files <- bimodal_cv_files
# trimodal_cv_files <- grep(pattern = ".ResponseOnly_.*gnndrug_.{6,11}_HyperOpt.+", all_cv_files, value = T)
# multimodal_cv_files <- grep(pattern = ".ResponseOnly_.*gnndrug_.{12,}_HyperOpt.+", all_cv_files, value = T)
# final_cv_files <- lineage_cv_files
# final_cv_files <- bottleneck_cv_files
# final_cv_files <- bimodal_cv_files
# final_cv_files <- trimodal_baseline_cv_files
# final_cv_files <- trimodal_cv_files
final_cv_files <- all_cv_files
length(final_cv_files)
sum(grepl(".*ElasticNet.*", final_cv_files))
sum(grepl(".*WithBottleNeck.*", final_cv_files))
sum(grepl(".*NoBottleNeck.*", final_cv_files))

# Read all data
all_results <- vector(mode = "list", length = length(final_cv_files))
rm(list = c("all_results_copy", "all_results_long_copy", "all_results_sub", "cur_res", "cur_p", "unique_combos"))
gc()
for (i in 1:length(final_cv_files)) {
  cur_res <- fread(final_cv_files[i])
  if (!grepl(".*Baseline_ElasticNet.*", final_cv_files[i])) {
    data_types <- gsub(".+_\\w*drug_(.+)_HyperOpt.+", "\\1", final_cv_files[i])
    data_types <- toupper(data_types)
    merge_method <- gsub(".+MergeBy(\\w+)_.*RMSE.+", "\\1", final_cv_files[i])
    loss_method <- gsub(".+_(.*)RMSE.+", "\\1RMSE", final_cv_files[i])
    drug_type <- gsub(".+_(\\w*)drug.+_HyperOpt.+", "\\1drug", final_cv_files[i])
    drug_type <- toupper(drug_type)
    split_method <- gsub(".+Split_(\\w+)_\\w+BottleNeck.+", "\\1", final_cv_files[i])
    bottleneck <- gsub(".+Split_\\w+_(\\w+BottleNeck).+", "\\1", final_cv_files[i])
    # data_types <- strsplit(data_types, "_")[[1]]
    # cur_res$epoch <- as.integer(epoch)
    cur_res$data_types <- data_types
    cur_res$merge_method <- merge_method
    cur_res$loss_type <- loss_method
    cur_res$drug_type <- drug_type
    cur_res$split_method <- split_method
    cur_res$bottleneck <- bottleneck
    
  } else {
    split_method <- gsub(".+Baseline_ElasticNet_Split_(\\w+)_drug_.+", "\\1", final_cv_files[i])
    data_types <- gsub(".+Baseline_ElasticNet_Split_\\w+_drug_(\\w+).+", "\\1", final_cv_files[i])
    data_types <- toupper(data_types)
    cur_res$data_types <- data_types
    cur_res$split_method <- split_method
    cur_res$merge_method <- "Merge By Early Concat"
    cur_res$loss_type <- "Base Model"
    cur_res$drug_type <- "Base Model"
    cur_res$bottleneck <- "No Data Bottleneck"
  }
  
  cur_fold <- gsub(".+CV_Index_(\\d)_.+", "\\1", final_cv_files[i])
  cur_res$fold <- cur_fold
  
  all_results[[i]] <- cur_res
}
rm(cur_res)
gc()

all_results <- rbindlist(all_results, fill = T)
if (any(all_results$merge_method == "Merge By Early Concat")) {
  all_results[is.na(rmse_loss), RMSELoss := abs(target - predicted), by = .I]
  all_results[!is.na(rmse_loss), RMSELoss := rmse_loss, by = .I]
  all_results$rmse_loss <- NULL
} else {
  all_results[, RMSELoss := abs(target - predicted), by = .I]
}

# all_results[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold")]
all_results$V1 <- NULL

# Update CV splitting method names
all_results[split_method == "BOTH", split_method := "Split By Both Cell Line & Drug Scaffold"]
all_results[split_method == "DRUG", split_method := "Split By Drug Scaffold"]
all_results[split_method == "CELL_LINE", split_method := "Split By Cell Line"]
all_results[split_method == "LINEAGE", split_method := "Split By Cancer Type"]

# all_results[merge_method == "MergeByEarlyConcat"]$merge_method <- "Merge By Early Concat"
# Update model names based on used techniques
all_results[loss_type == "RMSE", loss_type := "Base Model"]
all_results[loss_type == "WeightedRMSE", loss_type := "Base Model + LDS"]
all_results[merge_method == "Concat", merge_method := "Base Model"]
all_results[merge_method == "LMF", merge_method := "Base Model + LMF"]
all_results[merge_method == "Sum", merge_method := "Base Model + Sum"]
all_results[drug_type == "DRUG", drug_type := "Base Model"]
all_results[drug_type == "GNNDRUG", drug_type := "Base Model + GNN"]

# Update data bottleneck names
all_results[bottleneck == "NoBottleNeck", bottleneck := "No Data Bottleneck"]
all_results[bottleneck == "WithBottleNeck", bottleneck := "With Data Bottleneck"]

all_results[, Targeted := fifelse(cpd_name %in% targeted_drugs, "Targeted Drug", "Untargeted Drug")]

all_results[, TargetRange := fifelse(target >= 0.7, "Target Above 0.7", "Target Below 0.7")]

# table(all_results$Targeted)
# table(all_results$TargetRange)
# 
# all_results[RMSELoss > 1]
# table(all_results[RMSELoss > 1]$data_types)
# table(all_results$data_types)


# Save 
fwrite(all_results, "Data/all_results.csv")
# fwrite(all_results, "Data/all_bimodal_results.csv")

# Identify duplicated folds
unique_combos <- fread("Data/shared_unique_combinations.csv")
unique_combos[, unique_samples := paste0(cpd_name, "_", cell_name)]
all_results[, unique_samples := paste0(cpd_name, "_", cell_name)]
all_results_sub <- all_results[unique_samples %in% unique_combos$unique_samples]

all_results_sub[, num_samples := .N, by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "bottleneck")]
unique(all_results_sub[num_samples > 125212][, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "bottleneck", "num_samples")])

# Check for missing folds per config
all_results[, num_folds := uniqueN(fold), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "bottleneck")]

unique(all_results[num_folds != 5][, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "bottleneck", "num_folds")])
# data_types     merge_method        loss_type        drug_type                            split_method           bottleneck
# 1:  CNV_METAB       Base Model       Base Model       Base Model                      Split By Cell Line With Data Bottleneck
# 2:        CNV Base Model + Sum Base Model + LDS Base Model + GNN                      Split By Cell Line   No Data Bottleneck
# 3:        CNV Base Model + LMF       Base Model Base Model + GNN                  Split By Drug Scaffold   No Data Bottleneck
# 4:  HIST_RPPA Base Model + LMF Base Model + LDS Base Model + GNN                      Split By Cell Line   No Data Bottleneck
# 5:  HIST_RPPA Base Model + LMF Base Model + LDS Base Model + GNN                  Split By Drug Scaffold   No Data Bottleneck
# 6: MIRNA_HIST Base Model + LMF Base Model + LDS Base Model + GNN                      Split By Cell Line   No Data Bottleneck
# 7:      MIRNA Base Model + LMF       Base Model Base Model + GNN                  Split By Drug Scaffold   No Data Bottleneck
# 8: MIRNA_RPPA Base Model + LMF Base Model + LDS Base Model + GNN                      Split By Cell Line   No Data Bottleneck
# 9: MIRNA_RPPA Base Model + LMF Base Model + LDS Base Model + GNN                  Split By Drug Scaffold   No Data Bottleneck
# 10:    MUT_CNV Base Model + LMF Base Model + LDS Base Model + GNN Split By Both Cell Line & Drug Scaffold   No Data Bottleneck
# 11:        MUT       Base Model Base Model + LDS Base Model + GNN Split By Both Cell Line & Drug Scaffold   No Data Bottleneck
# 12:       PROT Base Model + LMF       Base Model Base Model + GNN                  Split By Drug Scaffold   No Data Bottleneck

# Targeted vs Untargeted in Baseline ====
# targeted_drugs <- fread("Data/DRP_Training_Data/CANCER_GOV_TARGETED_DRUGS.csv", fill = T)
# targeted_drugs <- targeted_drugs$Targeted_Drugs
all_results_copy <- fread("Data/all_results.csv")
all_results_copy <- all_results[nchar(data_types) <= 5]
# all_results_copy[, cv_mean := mean(RMSELoss), by = c("cpd_name", "cell_name", "data_types", "merge_method", "loss_type", "drug_type", "split_method")]

# baseline_with_gnn <- all_results_long_copy[(merge_method == "Concat" & loss_type == "RMSE" & split_method == "DRUG")]
baseline <- all_results_copy[merge_method == "MergeByConcat" & loss_type == "UnweightedLoss" & data_types %in% c("EXP", "PROT") &
                               drug_type == "Morgan" & split_method == "SplitByBoth" & nchar(data_types) <= 5]

p <- ggplot(baseline, mapping = aes(x = Targeted, y = cv_mean)) +
  geom_boxplot() +
  facet_wrap(~data_types+TargetRange, ncol = 2) +
  ggtitle(label = tools::toTitleCase("Comparison of GNN drug representation on targeted and untargeted drugs"),
          subtitle = "5-fold validation RMSE loss using strict splitting, True Target >= 0.7") +


scale_fill_discrete(name = "CV Fold:") +
scale_x_discrete() +
scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                             "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
geom_errorbar(aes(x=data_types,
                  y=cv_mean,
                  ymax=cv_mean, 
                  ymin=cv_mean, col='red'), linetype=2, show.legend = FALSE) +
geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean), vjust = -0.5)
# targeted_drug_results <- all_results[cpd_name %in% targeted_drugs]

# all_results_copy[, Targeted := ifelse(cpd_name %ilike% paste0(targeted_drugs, collapse = "|"), T, F)]


# unique(all_results_copy[Targeted == T]$cpd_name)
# dput(unique(all_results_copy[Targeted == T]$cpd_name))
# all_results_copy <- all_results_copy[Targeted == T]
# all_results_copy <- all_results_copy[target >= 0.9]

# all_results_copy_sub <- all_results_copy[target >= 0.7]
# all_results_copy_sub <- all_results_copy[target >= 0.7]

# temp <- all_results_copy_sub[data_types == "EXP" & merge_method == "Concat"]
# temp[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "Targeted")]



# Bi-modal Baseline Bottleneck Comparison (split by cell line) ====
all_results_copy <- fread("Data/all_results.csv")
all_results_copy <- all_results[nchar(data_types) <= 5]
# all_results_copy <- all_results
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "bottleneck")
# all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

# Bar plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((split_method == "Split By Cell Line" &
                                                    merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5)),
                          fill_by = quote(bottleneck),
                          bar_level_order = c("With Data Bottleneck", "No Data Bottleneck"),
                          facet_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          legend_title = "Data Type:",
                          calculate_avg_mae = F,
)
cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold")) 

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_Bottleneck_Comparison_BarPlot.pdf")

# Violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((split_method == "Split By Cell Line" &
                                                    merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5)),
                          fill_by = quote(bottleneck),
                          bar_level_order = c("With Data Bottleneck", "No Data Bottleneck"),
                          facet_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "data_types"),
                          legend_title = "Data Type:",
                          plot_type = "violin_plot", 
                          target_sub_by = c("Target Above 0.7", "Target Below 0.7"),
                          # target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("With Data Bottleneck", "No Data Bottleneck")),
                          test = "ks.test",
                          paired = T
)

cur_p <- cur_p + theme(text = element_text(size = 18, face = "bold")) + expand_limits(y = c(0, 1.5))
ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_Bottleneck_Comparison_ViolinPlot.pdf",
       height = 8)

## Concordance between different models ====
all_results_copy <- all_results[bottleneck == "With Data Bottleneck"]
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "bottleneck")
# all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")
all_comparisons <- utils::combn(data_order, 2, simplify = T)
all_comparisons <- list(c("MUT", "CNV"), c("CNV", "EXP"), c("HIST", "RPPA"))

cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((split_method == "Split By Cell Line" &
                                                    merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5)),
                          fill_by = quote(data_types),
                          # bar_level_order = c("With Data Bottleneck", "No Data Bottleneck"),
                          bar_level_order = data_order,
                          facet_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          data_order = data_order,
                          # facet_by = c("TargetRange"),
                          facet_by = NULL,
                          legend_title = "Model:",
                          plot_type = "box_plot", 
                          # target_sub_by = c("Target Above 0.7", "Target Below 0.7"),
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = NULL,
                          test = "ks.test",
                          paired = T,
)

all_results_subset <- subset(all_results_copy, (split_method == "Split By Cell Line" &
                                                             merge_method == "Base Model" &
                                                             loss_type == "Base Model" &
                                                             drug_type == "Base Model" &
                                                             nchar(data_types) <= 5))
# all_results_sub_sub <- all_results_subset[TargetRange %in% c("Target Above 0.7", "Target Below 0.7")]
# Order data for the facet
all_results_subset[, data_types := factor(data_types, levels = data_order)]
all_results_subset[, TargetRange := factor(unlist(all_results_subset[, "TargetRange", with = F]),
                                                      levels = c("Target Above 0.7", "Target Below 0.7"))]
all_results_subset[, cv_mean := mean(RMSELoss), by = eval(avg_loss_by[!avg_loss_by %in% c("fold")])]

# all_results_sub_sub[, cv_sd := sd(RMSELoss), by = eval(avg_loss_by[!avg_loss_by %in% c("fold")])]

data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")
all_comparisons <- utils::combn(data_order, 2, simplify = F)
all_stat_tests <- vector(mode = "list", length = length(all_comparisons))
for (i in 1:length(all_stat_tests)) {
  all_stat_tests[[i]] <- ks.test(all_results_subset[data_types == all_comparisons[[i]][1]]$RMSELoss,
                                 all_results_subset[data_types == all_comparisons[[i]][2]]$RMSELoss,)
}


all_stat_tests <- vector(mode = "list", length = 8)
for (i in 1:length(data_order)) {
  all_stat_tests[[i]] <- compare_means(RMSELoss ~ data_types, all_results_subset,
                                 ref.group = data_order[i], 
                                 method = "wilcox.test", alternative = "two.sided",
                                 p.adjust.method = "fdr", paired = F)
}


cur_palette <- get_palette(palette = "jco", 8)

final_p <- cur_p + theme(axis.text.x = element_text(), legend.position = "none")
for (i in 1:length(data_order)) {
  final_p <- final_p +
    # theme(axis.text.x = element_text()) +
    geom_bracket(
    aes(xmin = group1,
        xmax = group2,
        # label = p.adj),
        label = signif(p.adj, 2)), position = "identity",
    data = all_stat_tests[[i]], y.position = 0.3 + (0.3 * i),
    step.increase = 0.015,
    label.size = 3,
    tip.length = 0.01, color = cur_palette[i])
}
final_p 

ggsave(plot = final_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_Bottleneck_Concordance_Comparison_BoxPlot.pdf",
       height = 12)

## R-squared Plot ====
all_results_subset <- subset(all_results_copy, (split_method == "Split By Cell Line" &
                                                  merge_method == "Base Model" &
                                                  loss_type == "Base Model" &
                                                  drug_type == "Base Model" &
                                                  nchar(data_types) <= 5))
# all_results_sub_sub <- all_results_subset[TargetRange %in% c("Target Above 0.7", "Target Below 0.7")]
# Order data for the facet
# all_results_subset[, data_types := factor(data_types, levels = data_order)]
# all_results_subset[, TargetRange := factor(unlist(all_results_subset[, "TargetRange", with = F]),
#                                            levels = c("Target Above 0.7", "Target Below 0.7"))]
# all_results_subset[, cv_mean := mean(RMSELoss), by = eval(avg_loss_by[!avg_loss_by %in% c("fold")])]

# Find samples that are shared between all data types
all_results_subset[, unique_sample := paste0(cpd_name, "_", cell_name)]
shared_unique_samples <- Reduce(intersect, split(all_results_subset$unique_sample, all_results_subset$data_types))
all_results_copy <- all_results_subset[unique_sample %in% shared_unique_samples]
# all_results_shared_subset$unique_sample <- NULL
uniqueN(all_results_copy) / 8  # 125,212 samples in each model that are paired

# Set order within each group by the unique ID, so that each group has the same order (for pairing?)
setorder(all_results_copy, data_types, unique_sample)
# Confirm:
all_results_copy[, head(unique_sample,2),by=data_types]


cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((split_method == "Split By Cell Line" &
                                                    merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5)),
                          fill_by = quote(data_types),
                          # bar_level_order = c("With Data Bottleneck", "No Data Bottleneck"),
                          bar_level_order = data_order,
                          facet_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          data_order = data_order,
                          # facet_by = c("TargetRange"),
                          facet_by = NULL,
                          legend_title = "Model:",
                          plot_type = "box_plot", 
                          # target_sub_by = c("Target Above 0.7", "Target Below 0.7"),
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = NULL,
                          test = "wilcox.test",
                          paired = F, hide_outliers = T
)

all_stat_tests <- vector(mode = "list", length = 8)
for (i in 1:length(data_order)) {
  all_stat_tests[[i]] <- compare_means(RMSELoss ~ data_types, all_results_copy,
                                       ref.group = data_order[i],
                                       method = "wilcox.test", alternative = "two.sided",
                                       p.adjust.method = "fdr", paired = T)
}


cur_palette <- get_palette(palette = "jco", 8)

final_p <- cur_p + theme(axis.text.x = element_text(), legend.position = "none")
for (i in 1:length(data_order)) {
  final_p <- final_p +
    # theme(axis.text.x = element_text()) +
    geom_bracket(
      aes(xmin = group1,
          xmax = group2,
          # label = p.adj),
          label = signif(p.adj, 2)), position = "identity",
      data = all_stat_tests[[i]], y.position = 0.3 + (0.3 * i),
      step.increase = 0.015,
      label.size = 2, vjust = 1,
      tip.length = 0.01, color = cur_palette[i])
}
final_p 

rsq <- function (x, y) cor(x, y, method = "pearson") ^ 2
rmse <- function(x, y) sqrt(mean((x - y)^2))
mae <- function(x, y) mean(abs(x - y))

all_results_copy[, r2_by_range := rsq(target, predicted), by = c("data_types", "TargetRange", "Targeted")]
all_results_copy[, rmse_by_range := rmse(target, predicted), by = c("data_types", "TargetRange", "Targeted")]
all_results_copy[, avg_rmseloss_by_range := mean(RMSELoss), by = c("data_types", "TargetRange", "Targeted")]
all_results_copy[, mae_by_range := mae(target, predicted), by = c("data_types", "TargetRange", "Targeted")]
# all_results_copy[, avg_rmseloss_by_range := mean(RMSELoss), by = c("data_types", "TargetRange")]
unique(all_results_copy[, c("data_types", "mae_by_range", "avg_rmseloss_by_range", "rmse_by_range", "r2_by_range", "TargetRange", "Targeted")])

# Upper AAC range correlation, targeted
all_upper_targeted_results_copy <- all_results_copy[TargetRange == "Target Above 0.7" & Targeted == "Targeted Drug"]
# upper_targeted_cors <- all_upper_targeted_results_copy[all_upper_targeted_results_copy, allow.cartesian=T, on = "unique_sample"][, cor(predicted, i.predicted), by=list(data_types, i.data_types)]
upper_targeted_r2 <- all_upper_targeted_results_copy[all_upper_targeted_results_copy, allow.cartesian=T, on = "unique_sample"][, rsq(predicted, i.predicted), by=list(data_types, i.data_types)]
upper_targeted_r2_dt <- dcast(upper_targeted_r2, data_types~i.data_types, value.var = "V1")
upper_targeted_r2_mat <- as.matrix(upper_targeted_r2_dt[, 2:9])
rownames(upper_targeted_r2_mat) <- upper_targeted_r2_dt$data_types

# Upper AAC range correlation, untargeted
all_upper_untargeted_results_copy <- all_results_copy[TargetRange == "Target Above 0.7" & Targeted == "Untargeted Drug"]
# upper_untargeted_cors <- all_upper_untargeted_results_copy[all_upper_untargeted_results_copy, allow.cartesian=T, on = "unique_sample"][, cor(predicted, i.predicted), by=list(data_types, i.data_types)]
upper_untargeted_r2 <- all_upper_untargeted_results_copy[all_upper_untargeted_results_copy, allow.cartesian=T, on = "unique_sample"][, rsq(predicted, i.predicted), by=list(data_types, i.data_types)]
upper_untargeted_r2_dt <- dcast(upper_untargeted_r2, data_types~i.data_types, value.var = "V1")
upper_untargeted_r2_mat <- as.matrix(upper_untargeted_r2_dt[, 2:9])
rownames(upper_untargeted_r2_mat) <- upper_untargeted_r2_dt$data_types

# Lower AAC range correlation, targeted
all_lower_targeted_results_copy <- all_results_copy[TargetRange == "Target Below 0.7" & Targeted == "Targeted Drug"]
# lower_targeted_cors <- all_lower_targeted_results_copy[all_lower_targeted_results_copy, allow.cartesian=T, on = "unique_sample"][, cor(predicted, i.predicted), by=list(data_types, i.data_types)]
lower_targeted_r2 <- all_lower_targeted_results_copy[all_lower_targeted_results_copy, allow.cartesian=T, on = "unique_sample"][, rsq(predicted, i.predicted), by=list(data_types, i.data_types)]

lower_targeted_r2_dt <- dcast(lower_targeted_r2, data_types~i.data_types, value.var = "V1")
lower_targeted_r2_mat <- as.matrix(lower_targeted_r2_dt[, 2:9])
rownames(lower_targeted_r2_mat) <- lower_targeted_r2_dt$data_types

# Lower AAC range correlation, untargeted
all_lower_untargeted_results_copy <- all_results_copy[TargetRange == "Target Below 0.7" & Targeted == "Untargeted Drug"]
# lower_untargeted_cors <- all_lower_untargeted_results_copy[all_lower_untargeted_results_copy, allow.cartesian=T, on = "unique_sample"][, cor(predicted, i.predicted), by=list(data_types, i.data_types)]
lower_untargeted_r2 <- all_lower_untargeted_results_copy[all_lower_untargeted_results_copy, allow.cartesian=T, on = "unique_sample"][, rsq(predicted, i.predicted), by=list(data_types, i.data_types)]

lower_untargeted_r2_dt <- dcast(lower_untargeted_r2, data_types~i.data_types, value.var = "V1")
lower_untargeted_r2_mat <- as.matrix(lower_untargeted_r2_dt[, 2:9])
rownames(lower_untargeted_r2_mat) <- lower_untargeted_r2_dt$data_types

# install.packages("corrplot")
# require(corrplot)
# install.packages("ggcorrplot")
# install.packages("patchwork")
require(ggcorrplot)
require(patchwork)
require(ggplot2)

g_upper_targeted <- ggcorrplot(upper_targeted_r2_mat, hc.order = TRUE, outline.color = "white",
           type = "lower", 
           ggtheme = ggplot2::theme_gray,
           colors = c("#E46726", "white", "#6D9EC1"),
           lab = TRUE) + ggtitle("AAC >= 0.7, Targeted") + 
  theme(text = element_text(size = 12, face = "bold"), 
        legend.position = 'none')
g_upper_untargeted <- ggcorrplot(upper_untargeted_r2_mat, hc.order = TRUE, outline.color = "white",
           type = "lower", 
           ggtheme = ggplot2::theme_gray,
           colors = c("#E46726", "white", "#6D9EC1"),
           lab = TRUE) + ggtitle("AAC >= 0.7, Untargeted") + 
  theme(text = element_text(size = 12, face = "bold"), 
        legend.position = 'none')

# g_upper <- ggplot(upper_r2, aes(data_types, i.data_types, fill = V1)) +
#   geom_tile() +
#   ggtitle("AAC >= 0.7") + 
#   theme(text = element_text(size = 14, face = "bold"), 
#         legend.position = 'none')

g_lower_targeted <- ggcorrplot(lower_targeted_r2_mat, hc.order = TRUE, outline.color = "white",
           type = "lower",
           ggtheme = ggplot2::theme_gray,
           colors = c("#E46726", "white", "#6D9EC1"),
           lab = TRUE) + ggtitle("AAC < 0.7, Targeted") +
  theme(text = element_text(size = 12, face = "bold"),
        legend.position = 'none')
g_lower_untargeted <- ggcorrplot(lower_untargeted_r2_mat, hc.order = TRUE, outline.color = "white",
           type = "lower",
           ggtheme = ggplot2::theme_gray,
           colors = c("#E46726", "white", "#6D9EC1"),
           lab = TRUE) + ggtitle("AAC < 0.7, Untargeted") +
  theme(text = element_text(size = 12, face = "bold"),
        legend.position = 'none')


full <- (g_upper_targeted | g_upper_untargeted) / (g_lower_targeted | g_lower_untargeted)


ggsave("Plots/CV_Results/Baseline_R2_Matrix_ByDataType.pdf",
       height = 8, width = 8, units = "in",
       full)
# corrplot(final_cor_mat, method = 'square', order = 'AOE', type = "lower",
#          addCoef.col = 'white')

# pdf(file = "Plots/CV_Results/Baseline_Correlation_Matrix_ByDataType.pdf")

corrplot(final_cor_mat, method = 'square', order = 'AOE', type = "lower",
         addCoef.col = 'white')

dev.off()




rsq(all_results_copy[data_types == "EXP"]$target, all_results_copy[data_types == "EXP"]$predicted)
rsq(all_results_copy[data_types == "CNV"]$target, all_results_copy[data_types == "CNV"]$predicted)
rsq(all_results_copy[data_types == "PROT"]$target, all_results_copy[data_types == "PROT"]$predicted)
rsq(all_results_copy[data_types == "MUT"]$target, all_results_copy[data_types == "MUT"]$predicted)
rsq(all_results_copy[data_types == "MUT"]$target, all_results_copy[data_types == "MUT"]$predicted)

ggsave(plot = final_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_Bottleneck_Paired_Concordance_Comparison_BoxPlot.pdf",
       height = 12)


# Bi-Modal Baseline Upper vs Lower AAC Range Comparison ====
all_results_copy <- all_results
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "bottleneck")
# all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

# Violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((split_method == "Split By Cell Line" &
                                                    merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    bottleneck == "No Data Bottleneck" &
                                                    nchar(data_types) <= 5)),
                          fill_by = quote(TargetRange),
                          bar_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          # facet_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          data_order = data_order,
                          facet_by = "data_types",
                          legend_title = "AAC Range:",
                          plot_type = "violin_plot", 
                          target_sub_by = c("Target Above 0.7", "Target Below 0.7"),
                          # target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Target Above 0.7", "Target Below 0.7")),
                          test = "ks.test",
                          paired = T
)

cur_p <- cur_p + theme(text = element_text(size = 18, face = "bold"))
# +
#   geom_text(data = all_results_copy, aes(x=data_types, label = round(cv_mean, 3), y = cv_mean + cv_sd),
#             vjust = 0.5, hjust = -0.25, angle = 90, position = position_dodge2(width = .9))


ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_UpperVsLower_Comparison_ViolinPlot.pdf",
       height = 8)

# Bar plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((split_method == "Split By Cell Line" &
                                                    merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    bottleneck == "No Data Bottleneck" &
                                                    nchar(data_types) <= 5)),
                          fill_by = quote(TargetRange),
                          bar_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          # facet_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          data_order = data_order,
                          facet_by = NULL,
                          legend_title = "AAC Range:",
                          plot_type = "bar_plot",
                          add_mean = T,
                          calculate_avg_mae = F,
                          # target_sub_by = c("Target Above 0.7", "Target Below 0.7"),
                          # # target_sub_by = "Target Above 0.7",
                          # cur_comparisons = list(c("Target Above 0.7", "Target Below 0.7")),
                          # test = "wilcox.test",
                          # paired = F
)

cur_p <- cur_p + theme(text = element_text(size = 18, face = "bold"))
ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_UpperVsLower_Diff_Comparison_BarPlot.pdf")

# Bi-Modal Baseline Targeted vs Untargeted Drug Comparison ====
all_results_copy <- fread("Data/all_results.csv")
all_results_copy <- all_results[nchar(data_types) <= 5]

all_results_copy <- all_results[bottleneck == "No Data Bottleneck"]
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "TargetRange", "bottleneck", "Targeted")
# all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")
# merge_method %in% c("Base Model") &
#   loss_type == "Base Model" & drug_type == "Base Model" &
#   split_method == "Split By Both Cell Line & Drug Scaffold" &
#   nchar(data_types) <= 5 & data_types != "MUT"

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((split_method == "Split By Cell Line" &
                                                    merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    bottleneck == "No Data Bottleneck" &
                                                    nchar(data_types) <= 5)),
                          fill_by = quote(Targeted),
                          bar_level_order = c("Targeted Drug", "Untargeted Drug"),
                          # facet_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "data_types"),
                          legend_title = "AAC Range:",
                          plot_type = "box_plot", 
                          target_sub_by = c("Target Above 0.7", "Target Below 0.7"),
                          # target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Targeted Drug", "Untargeted Drug")),
                          test = "ks.test",
                          paired = T,
                          hide_outliers = T
)

cur_p <- cur_p + theme(text = element_text(size = 18, face = "bold")) + expand_limits(y = c(0, 1.5))

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_UpperVsLower_Comparison_BoxPlot.pdf",
       height = 8)

## Difference between models ====
all_results_copy <- all_results_copy[TargetRange == "Target Above 0.7"]
# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((split_method == "Split By Cell Line" &
                                                    merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    bottleneck == "No Data Bottleneck" &
                                                    nchar(data_types) <= 5)),
                          fill_by = quote(Targeted),
                          bar_level_order = c("Targeted Drug", "Untargeted Drug"),
                          facet_level_order = c("Target Above 0.7"),
                          data_order = data_order,
                          facet_by = "TargetRange",
                          legend_title = "AAC Range:",
                          plot_type = "bar_plot", 
                          add_mean = T,
                          calculate_avg_mae = F
                          # target_sub_by = c("Target Above 0.7", "Target Below 0.7"),
                          # target_sub_by = "Target Above 0.7",
                          # cur_comparisons = list(c("Targeted Drug", "Untargeted Drug")),
                          # test = "wilcox.test",
                          # paired = F,
                          # hide_outliers = T,
                          
)

cur_p <- cur_p + theme(text = element_text(size = 18, face = "bold"))

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_TargetedVsUntargeted_Upper0.7_Comparison_BarPlot.pdf")

# Bi-modal Baseline Split Comparison ====
all_results_copy <- all_results
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "Targeted", "bottleneck")
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

# all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]

# TODO Must ensure different splitting methods also are compared on the same validation data
## Wilcox box plot (cell line and drug scaffold) ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck" &
                                                    # split_method %in% c("Split By Cell Line", "Split By Drug Scaffold", "Split By Both Cell Line & Drug Scaffold"))),
                                                    split_method %in% c("Split By Cell Line", "Split By Drug Scaffold"))),
                          facet_by = c("Targeted", "data_types"),
                          fill_by = quote(split_method),
                          data_order = data_order,
                          # bar_level_order = c("Split By Cell Line", "Split By Drug Scaffold", "Split By Both Cell Line & Drug Scaffold"),
                          bar_level_order = c("Split By Cell Line", "Split By Drug Scaffold"),
                          facet_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          plot_type = "box_plot",
                          legend_title = "Splitting Method:",
                          hide_outliers = T,
                          # target_sub_by = c("Target Above 0.7", "Target Below 0.7"),
                          # cur_comparisons = c("Targeted Drug", "Untargeted Drug"),
                          # cur_comparisons = list(c("Split By Cell Line", "Split By Drug Scaffold"),
                          #                        c("Split By Cell Line", "Split By Both Cell Line & Drug Scaffold"),
                          #                        c("Split By Both Cell Line & Drug Scaffold", "Split By Drug Scaffold")),
                          cur_comparisons = list(c("Split By Cell Line", "Split By Drug Scaffold")),
                          test = "wilcox.test",
                          paired = T, step_increase = 0.01,
                          y_lim = 0.05)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))
ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Baseline_Split_CellLineDrugScaffold_Wilcox_Comparison_BoxPlot.pdf",
       height = 8)

## KS boxplot (cell line and drug scaffold) ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck" &
                                                    # split_method %in% c("Split By Cell Line", "Split By Drug Scaffold", "Split By Both Cell Line & Drug Scaffold"))),
                                                    split_method %in% c("Split By Cell Line", "Split By Drug Scaffold"))),
                          facet_by = c("Targeted", "data_types"),
                          fill_by = quote(split_method),
                          data_order = data_order,
                          # bar_level_order = c("Split By Cell Line", "Split By Drug Scaffold", "Split By Both Cell Line & Drug Scaffold"),
                          bar_level_order = c("Split By Cell Line", "Split By Drug Scaffold"),
                          facet_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          plot_type = "box_plot",
                          legend_title = "Splitting Method:",
                          hide_outliers = T,
                          # target_sub_by = c("Target Above 0.7", "Target Below 0.7"),
                          # cur_comparisons = c("Targeted Drug", "Untargeted Drug"),
                          # cur_comparisons = list(c("Split By Cell Line", "Split By Drug Scaffold"),
                          #                        c("Split By Cell Line", "Split By Both Cell Line & Drug Scaffold"),
                          #                        c("Split By Both Cell Line & Drug Scaffold", "Split By Drug Scaffold")),
                          cur_comparisons = list(c("Split By Cell Line", "Split By Drug Scaffold")),
                          test = "ks.test",
                          paired = T, step_increase = 0.01,
                          y_lim = 0.05)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))
ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Baseline_Split_CellLineDrugScaffold_KS_Comparison_BoxPlot.pdf",
       height = 8)

## KS violin plot (cell line and drug scaffold) ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck" &
                                                    # split_method %in% c("Split By Cell Line", "Split By Drug Scaffold", "Split By Both Cell Line & Drug Scaffold"))),
                                                    split_method %in% c("Split By Cell Line", "Split By Drug Scaffold"))),
                          facet_by = c("Targeted", "data_types"),
                          fill_by = quote(split_method),
                          data_order = data_order,
                          # bar_level_order = c("Split By Cell Line", "Split By Drug Scaffold", "Split By Both Cell Line & Drug Scaffold"),
                          bar_level_order = c("Split By Cell Line", "Split By Drug Scaffold"),
                          facet_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          plot_type = "violin_plot",
                          legend_title = "Splitting Method:",
                          hide_outliers = T,
                          # target_sub_by = c("Target Above 0.7", "Target Below 0.7"),
                          # cur_comparisons = c("Targeted Drug", "Untargeted Drug"),
                          # cur_comparisons = list(c("Split By Cell Line", "Split By Drug Scaffold"),
                          #                        c("Split By Cell Line", "Split By Both Cell Line & Drug Scaffold"),
                          #                        c("Split By Both Cell Line & Drug Scaffold", "Split By Drug Scaffold")),
                          cur_comparisons = list(c("Split By Cell Line", "Split By Drug Scaffold")),
                          test = "ks.test",
                          paired = T, step_increase = 0.00,
                          y_lim = 0.05)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold")) + expand_limits(y = c(0, 1.5))
ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Baseline_Split_CellLineDrugScaffold_KS_Comparison_ViolinPlot.pdf",
       height = 8)

## Bar plot RMSE (cell line and drug scaffold) ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck" &
                                                    # split_method %in% c("Split By Cell Line", "Split By Drug Scaffold", "Split By Both Cell Line & Drug Scaffold"))),
                                                    split_method %in% c("Split By Cell Line", "Split By Drug Scaffold"))),
                          facet_by = c("Targeted", "TargetRange"),
                          fill_by = quote(split_method),
                          data_order = data_order,
                          # bar_level_order = c("Split By Cell Line", "Split By Drug Scaffold", "Split By Both Cell Line & Drug Scaffold"),
                          bar_level_order = c("Split By Cell Line", "Split By Drug Scaffold"),
                          facet_level_order = list(c("Targeted Drug", "Untargeted Drug"), 
                                                   c("Target Above 0.7", "Target Below 0.7")),
                          plot_type = "bar_plot",
                          legend_title = "Splitting Method:",
                          hide_outliers = T,
                          calculate_avg_mae = F,
                          y_lab = "Total RMSE Loss",
                          # target_sub_by = c("Target Above 0.7", "Target Below 0.7"),
                          # cur_comparisons = c("Targeted Drug", "Untargeted Drug"),
                          # cur_comparisons = list(c("Split By Cell Line", "Split By Drug Scaffold"),
                          #                        c("Split By Cell Line", "Split By Both Cell Line & Drug Scaffold"),
                          #                        c("Split By Both Cell Line & Drug Scaffold", "Split By Drug Scaffold")),
                          # cur_comparisons = list(c("Split By Cell Line", "Split By Drug Scaffold")),
                          # test = "ks.test",
                          # paired = T, step_increase = 0.01,
                          y_lim = 0.1)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))
ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Baseline_Split_CellLineDrugScaffold_RMSE_Comparison_BarPlot.pdf",
       height = 8)


# Bi-modal Baseline vs ElasticNet Baseline (Split By Cell Line) ====

## Without separating target ranges ====
all_results_copy <- fread("Data/all_results.csv")
all_results_copy <- all_results_copy[nchar(data_types) <= 5]

# Don't average loss by TargetRange
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "bottleneck")
all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
all_results_copy[merge_method == "Base Model", merge_method := "Baseline Neural Network"]
all_results_copy[merge_method == "Merge By Early Concat", merge_method := "Elastic Net"]
all_results_copy[merge_method == "Elastic Net", bottleneck := "No Data Bottleneck"]
all_results_copy <- all_results_copy[data_types != "MUT"]
# Order data types by mut, cnv, exp, prot, mirna, metab, hist, rppa
data_order <- c('CNV', 'EXP', 'PROT', 'MIRNA', 'METAB', 'HIST', 'RPPA')

# Bar plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method %in% c("Baseline Neural Network", "Elastic Net") &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Elastic Net", "Baseline Neural Network"),
                          data_order = data_order,
                          facet_by = NULL,
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          plot_type = "bar_plot",
                          calculate_avg_mae = F,
                          add_mean = T,
                          y_lim = 0.05)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_ANN_Baseline_vs_ElasticNet_No_TargetRange_Separation_SplitByCellLine_Comparison.pdf")

# my_comparisons <- list( c("Base Model", "Base Model + LMF"), c("Base Model + Sum", "Base Model + LMF"), c("Base Model", "Base Model + Sum"))
# my_comparisons <- list( c("Elastic Net", "Baseline Neural Network"))

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method %in% c("Baseline Neural Network", "Elastic Net") &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Baseline Neural Network", "Elastic Net"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          cur_comparisons = list(c("Elastic Net", "Baseline Neural Network")))

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))
ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_ANN_Baseline_vs_ElasticNet_No_TargetRange_Separation_SplitByBoth_Comparison_BoxPlot.pdf")


## with separating target ranges ====
all_results_copy <- fread("Data/all_results.csv")
all_results_copy <- all_results_copy[nchar(data_types) <= 5]

# Average loss by TargetRange
all_results_copy <- all_results[merge_method %in% c("Base Model", "Merge By Early Concat") &
                                  loss_type == "Base Model" & drug_type == "Base Model" &
                                  split_method == "Split By Both Cell Line & Drug Scaffold" &
                                  nchar(data_types) <= 5 & data_types != "MUT"]
all_results_copy[merge_method == "Base Model", merge_method := "Baseline Neural Network"]
all_results_copy[merge_method == "Merge By Early Concat", merge_method := "Elastic Net"]
all_results_copy[merge_method == "Elastic Net", bottleneck := "No Data Bottleneck"]

# all_results_copy_sub <- all_results_copy[TargetRange == "TargetAbove 0.7"]
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "bottleneck", "TargetRange")
all_results_copy <- all_results_copy[data_types != "MUT"]
data_order <- c('CNV', 'EXP', 'PROT', 'MIRNA', 'METAB', 'HIST', 'RPPA')

# Bar plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method %in% c("Baseline Neural Network", "Elastic Net") &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Elastic Net", "Baseline Neural Network"),
                          data_order = data_order,
                          facet_by = c("TargetRange"),
                          facet_level_order = c("Target Above 0.7", "Target Below 0.7"),
                          legend_title = "Model Type:",
                          plot_type = "bar_plot",
                          calculate_avg_mae = F,
                          add_mean = T, 
                          # facet_nrow = 1,
                          y_lim = 0.05,
                          min_diff = 0.03)


cur_p <- cur_p + theme(text = element_text(size = 18, face = "bold"))

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_ANN_Baseline_vs_ElasticNet_SplitByCellLine_Comparison.pdf")

# Violin plot
all_results_copy <- all_results_copy[TargetRange == "Target Above 0.7"]
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method %in% c("Baseline Neural Network", "Elastic Net") &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Baseline Neural Network", "Elastic Net"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "violin_plot",
                          cur_comparisons = list(c("Elastic Net", "Baseline Neural Network")),
                          test = "ks.test",
                          paired = T)

cur_p <- cur_p + theme(text = element_text(size = 18, face = "bold"))

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_ANN_Baseline_vs_ElasticNet_SplitByCellLine_Upper_0.7_Comparison_ViolinPlot.pdf")

## Separating Targeted vs Untargeted drugs in upper AAC ====
all_results_copy <- fread("Data/all_results.csv")
all_results_copy <- all_results_copy[nchar(data_types) <= 5]

all_results_copy[merge_method == "Base Model", merge_method := "Baseline Neural Network"]
all_results_copy[merge_method == "Merge By Early Concat", merge_method := "Elastic Net"]
all_results_copy[merge_method == "Elastic Net", bottleneck := "No Data Bottleneck"]

# all_results_copy_sub <- all_results_copy[TargetRange == "TargetAbove 0.7"]
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "bottleneck", "Targeted", "TargetRange")
all_results_copy <- all_results_copy[data_types != "MUT"]
data_order <- c('CNV', 'EXP', 'PROT', 'MIRNA', 'METAB', 'HIST', 'RPPA')


# Bar plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method %in% c("Baseline Neural Network", "Elastic Net") &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    TargetRange == "Target Above 0.7" &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Elastic Net", "Baseline Neural Network"),
                          data_order = data_order,
                          facet_by = "Targeted",
                          facet_level_order = c("Targeted Drug", "Untargeted Drug"),
                          legend_title = "Model Type:",
                          plot_type = "bar_plot",
                          add_mean = T,
                          calculate_avg_mae = F, 
                          facet_nrow = 1,
                          min_diff = 0.03,
                          y_lim = 0.05)

cur_p <- cur_p + theme(text = element_text(size = 18, face = "bold")) 
  
ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_ANN_Baseline_vs_ElasticNet_Targeted_vs_Untargeted_Upper_SplitByCellLine_Comparison_BarPlot.pdf")

# violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method %in% c("Baseline Neural Network", "Elastic Net") &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    TargetRange == "Target Above 0.7" &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Elastic Net", "Baseline Neural Network"),
                          data_order = data_order,
                          facet_by = c("Targeted", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "violin_plot",
                          cur_comparisons = list(c("Elastic Net", "Baseline Neural Network")),
                          test = "ks.test", 
                          paired = T)

cur_p <- cur_p + theme(text = element_text(size = 18, face = "bold")) + 
  expand_limits(y = c(0, 1.5))

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_ANN_Baseline_Targeted_vs_Untargeted_Upper_SplitByCellLine_Comparison_ViolinPlot.pdf",
       height = 8)

# Bi-Modal Baseline vs LDS ====
all_results <- fread("Data/all_results.csv")
all_results <- all_results[nchar(data_types) <= 5]

all_results_copy <- all_results
all_results_copy[target > 0.7 & target < 0.9, TargetRange := "Target Between 0.7 & 0.9"]
all_results_copy[target >= 0.9, TargetRange := "Target Above 0.9"]

avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "bottleneck", "TargetRange", "Targeted")
# all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

## Split By Both Cell Line & Drug Scaffold ====
# Bar plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("Base Model", "Base Model + LDS"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.9",
                                                "Target Between 0.7 & 0.9",
                                                "Target Below 0.7"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_vs_LDS_SplitByBoth_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("Base Model", "Base Model + LDS"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = c("Target Between 0.7 & 0.9", "Target Above 0.9"),
                          cur_comparisons = list(c("Base Model", "Base Model + LDS")),
                          test = "wilcox.test",
                          paired = F
                          )

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_vs_LDS_SplitByBoth_Comparison_BoxPlot.pdf",
       height = 8)

# Violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("Base Model", "Base Model + LDS"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "violin_plot",
                          target_sub_by = c("Target Between 0.7 & 0.9", "Target Above 0.9"),
                          cur_comparisons = list(c("Base Model", "Base Model + LDS")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_vs_LDS_SplitByBoth_Comparison_ViolinPlot.pdf",
       height = 8)

## Split By Drug Scaffold ====
# Bar plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("Base Model", "Base Model + LDS"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.9",
                                                "Target Between 0.7 & 0.9",
                                                "Target Below 0.7"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = p, filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_vs_LDS_SplitByDrug_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("Base Model", "Base Model + LDS"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = c("Target Between 0.7 & 0.9", "Target Above 0.9"),
                          cur_comparisons = list(c("Base Model", "Base Model + LDS")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_vs_LDS_SplitByDrug_Comparison_BoxPlot.pdf",
       height = 8)
## Split By Cell Line ====
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "bottleneck", "TargetRange", "Targeted")

# Bar plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("Base Model", "Base Model + LDS"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "Targeted"),
                          facet_level_order = list(c("Target Above 0.9", "Target Between 0.7 & 0.9","Target Below 0.7"),
                                                   c("Targeted Drug", "Untargeted Drug")),
                          facet_nrow = 3,
                          legend_title = "Model Type:",
                          plot_type = "bar_plot",
                          calculate_avg_mae = F, y_lab = "Total RMSE Loss",
                          y_lim = 0.1)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_LDS_Upper_SplitByCellLine_Comparison_BarPlot.pdf",
       height = 12)

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("Base Model", "Base Model + LDS"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = c("Target Between 0.7 & 0.9", "Target Above 0.9"),
                          cur_comparisons = list(c("Base Model", "Base Model + LDS")),
                          test = "wilcox.test",
                          paired = T
)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))
ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_vs_LDS_SplitByCellLine_Comparison_BoxPlot.pdf",
       height = 8)

# Violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("Base Model", "Base Model + LDS"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "violin_plot",
                          target_sub_by = c("Target Between 0.7 & 0.9", "Target Above 0.9"),
                          cur_comparisons = list(c("Base Model", "Base Model + LDS")),
                          test = "ks.test",
                          paired = T
)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold")) +
  expand_limits(y = c(0, 1.5))
ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_LDS_SplitByCellLine_Comparison_ViolinPlot.pdf",
       height = 8)

## Split Comparison ====
# Bar plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(split_method),
                          bar_level_order = c("Split By Both Cell Line & Drug Scaffold", "Split By Cell Line", "Split By Drug Scaffold"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.9",
                                                "Target Between 0.7 & 0.9",
                                                "Target Below 0.7"),
                          legend_title = "Split Method:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_with_LDS_Split_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(split_method),
                          bar_level_order = c("Split By Cell Line", "Split By Drug Scaffold", "Split By Both Cell Line and Drug Scaffold"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Split Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = c("Target Between 0.7 & 0.9", "Target Above 0.9"),
                          cur_comparisons = list(c("Split By Cell Line", "Split By Drug Scaffold"),
                                                 c("Split By Cell Line", "Split By Both Cell Line and Drug Scaffold"),
                                                 c("Split By Drug Scaffold", "Split By Both Cell Line and Drug Scaffold")),
                          test = "t.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_with_LDS_Split_Comparison_BoxPlot.pdf",
       height = 8)

# Bi-modal Baseline vs LMF ====
all_results <- fread("Data/all_results.csv")
all_results <- all_results[nchar(data_types) <= 5]
all_results_copy <- all_results
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "bottleneck", "TargetRange")
all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

## Split By Both Cell Line & Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method != "Merge By Early Concat" &
                                                    drug_type == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Base Model", "Base Model + LMF", "Base Model + Sum"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Merge Method:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_vs_LMF_SplitByBoth_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method != "Merge By Early Concat" &
                                                    drug_type == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Base Model", "Base Model + Sum", "Base Model + LMF"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Base Model", "Base Model + Sum"),
                                                 c("Base Model + Sum", "Base Model + LMF"),
                                                 c("Base Model", "Base Model + LMF")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_LMF_SplitByBoth_Comparison_BoxPlot.pdf",
       height = 8)

## Split By Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method != "Merge By Early Concat" &
                                                    drug_type == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Base Model", "Base Model + LMF", "Base Model + Sum"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Merge Method:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_LMF_SplitByDrugScaffold_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method != "Merge By Early Concat" &
                                                    drug_type == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Base Model", "Base Model + Sum", "Base Model + LMF"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Base Model", "Base Model + Sum"),
                                                 c("Base Model + Sum", "Base Model + LMF"),
                                                 c("Base Model", "Base Model + LMF")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_LMF_SplitByDrugScaffold_Comparison_BoxPlot.pdf",
       height = 8)

## Split By Cell Line ====
# Bar plot
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "bottleneck", "TargetRange", "Targeted")

cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method != "Merge By Early Concat" &
                                                    drug_type == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Base Model", "Base Model + LMF", "Base Model + Sum"),
                          data_order = data_order,
                          facet_by = c("Targeted", "TargetRange"),
                          facet_level_order = list(c("Targeted Drug", "Untargeted Drug"),
                                                   c("Target Above 0.7","Target Below 0.7")),
                          legend_title = "Model Type:",
                          plot_type = "bar_plot",
                          calculate_avg_mae = F, y_lab = "Total RMSE Loss",
                          y_lim = 0.1)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_LMF_SplitByCellLine_Comparison_BarPlot.pdf",
       height = 8)

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method != "Merge By Early Concat" &
                                                    drug_type == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Base Model", "Base Model + Sum", "Base Model + LMF"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Base Model", "Base Model + Sum"),
                                                 c("Base Model + Sum", "Base Model + LMF"),
                                                 c("Base Model", "Base Model + LMF")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_LMF_SplitByCellLine_Comparison_BoxPlot.pdf",
       width = 15)

# Violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method != "Merge By Early Concat" &
                                                    drug_type == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("Base Model", "Base Model + Sum", "Base Model + LMF"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "violin_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Base Model", "Base Model + Sum"),
                                                 c("Base Model + Sum", "Base Model + LMF"),
                                                 c("Base Model", "Base Model + LMF")),
                          test = "ks.test",
                          paired = T
)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold")) + expand_limits(y = c(0, 1.5))

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_LMF_SplitByCellLine_Comparison_ViolinPlot.pdf",
       height = 10)

## Split Comparison ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    drug_type == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(split_method),
                          bar_level_order = c("Split By Both Cell Line & Drug Scaffold", "Split By Cell Line", "Split By Drug Scaffold"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Split Method:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_with_LMF_Split_Comparison.pdf")

# Bi-Modal Baseline vs GNN ====
all_results <- fread("Data/all_results.csv")
all_results <- all_results[nchar(data_types) <= 5]

all_results_copy <- all_results
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "bottleneck", "TargetRange")
# all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

## Split By Both Cell Line & Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("Base Model", "Base Model + GNN"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Drug Model:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_GNN_SplitByBoth_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("Base Model", "Base Model + GNN"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Base Model", "Base Model + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_GNN_SplitByBoth_Comparison_BoxPlot.pdf",
       height = 8)
## Split By Drug Scaffold ====
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "bottleneck", "TargetRange", "Targeted")

cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("Base Model", "Base Model + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "TargetRange"),
                          facet_level_order = list(c("Targeted Drug", "Untargeted Drug"),
                                                   c("Target Above 0.7", "Target Below 0.7")),
                          legend_title = "Model Type:",
                          plot_type = "bar_plot",
                          calculate_avg_mae = F,
                          y_lab = "Total RMSE Loss",
                          y_lim = 0.1)
cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_GNN_SplitByDrugScaffold_Comparison_BarPlot.pdf",
       height = 8)

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("Base Model", "Base Model + GNN"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Base Model", "Base Model + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_GNN_SplitByDrugScaffold_Comparison_BoxPlot.pdf",
       height = 8)

## Split By Cell Line ====
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "bottleneck", "TargetRange", "Targeted")

cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("Base Model", "Base Model + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "TargetRange"),
                          facet_level_order = list(c("Targeted Drug", "Untargeted Drug"),
                                                   c("Target Above 0.7","Target Below 0.7")),
                          legend_title = "Model Type:",
                          plot_type = "bar_plot",
                          calculate_avg_mae = F, y_lab = "Total RMSE Loss",
                          y_lim = 0.1)
cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_GNN_SplitByCellLine_Comparison_BarPlot.pdf",
       height = 8)

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("Base Model", "Base Model + GNN"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Base Model", "Base Model + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_GNN_SplitByCellLine_Comparison_BoxPlot.pdf",
       height = 8)

# Violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("Base Model", "Base Model + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted","data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "violin_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Base Model", "Base Model + GNN")),
                          test = "ks.test", 
                          paired = T
)
cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_GNN_SplitByCellLine_Comparison_ViolinPlot.pdf",
       height = 8)
## Split Comparison ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(split_method),
                          bar_level_order = c("Split By Both Cell Line & Drug Scaffold", "Split By Cell Line", "Split By Drug Scaffold"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Split Method:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_with_GNN_Split_Comparison.pdf")

## Targeted and Untargeted drugs in upper AAC range ====
all_results_copy <- all_results[TargetRange == "Target Above 0.7"]
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "bottleneck", "Targeted")
all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("Base Model", "Base Model + GNN"),
                          data_order = data_order,
                          facet_by = quote(Targeted),
                          facet_level_order = c("Targeted Drug", "Untargeted Drug"),
                          legend_title = "Drug Model:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_vs_GNN_Targeted_vs_Untargeted_SplitByBoth_Comparison.pdf")

# Bi-modal LMF + GNN without LDS (Split By Both Cell Line & Drug Scaffold) ====
all_results <- fread("Data/all_results.csv")
all_results <- all_results[nchar(data_types) <= 5]

all_results_copy <- all_results
# all_results_copy[target > 0.7 & target < 0.9]$TargetRange <- "Target Between 0.7 & 0.9"
# all_results_copy[target >= 0.9]$TargetRange <- "Target Above 0.9"

avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "bottleneck", "TargetRange", "Targeted")
# all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

# Must rename some columns to better distinguish differences on the plot
all_results_copy[loss_type == "Base Model", loss_type := "LMF + GNN"]
all_results_copy[loss_type == "Base Model + LDS", loss_type := "LDS + LMF + GNN"]

## Split By Both Cell Line & Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("LMF + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.9",
                                                "Target Between 0.7 & 0.9",
                                                "Target Below 0.7"),
                          legend_title = "Loss Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LDS_SplitByBoth_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("LMF + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = c("Target Between 0.7 & 0.9", "Target Above 0.9"),
                          cur_comparisons = list(c("LMF + GNN", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LDS_SplitByBoth_Comparison_BoxPlot.pdf",
       height = 8)

## Split By Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("LMF + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.9",
                                                "Target Between 0.7 & 0.9",
                                                "Target Below 0.7"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LDS_SplitByDrugScaffold_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("LMF + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = c("Target Between 0.7 & 0.9", "Target Above 0.9"),
                          cur_comparisons = list(c("LMF + GNN", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LDS_SplitByDrugScaffold_Comparison_BoxPlot.pdf",
       height = 8)

## Split By Cell Line ====

cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("LMF + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "TargetRange"), 
                          # facet_level_order = c("Target Above 0.9",
                          #                       "Target Between 0.7 & 0.9",
                          #                       "Target Below 0.7"),
                          facet_level_order = list(c("Targeted Drug", "Untargeted Drug"),
                                                   c("Target Above 0.7", "Target Below 0.7")),
                          # target_sub_by = c("Target Above 0.9", "Target Between 0.7 & 0.9"),
                          legend_title = "Model Type:",
                          calculate_avg_mae = F, y_lab = "Total RMSE Loss",
                          y_lim = 0.1)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))
ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LDS_SplitByCellLine_Comparison_BarPlot.pdf",
       height = 8)

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("LMF + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("TargetRange", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = c("Target Between 0.7 & 0.9", "Target Above 0.9"),
                          cur_comparisons = list(c("LMF + GNN", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LDS_SplitByCellLine_Comparison_BoxPlot.pdf",
       height = 8)

# Violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(loss_type),
                          bar_level_order = c("LMF + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "data_types"), 
                          plot_type = "violin_plot",
                          # facet_level_order = c("Target Above 0.9",
                          #                       "Target Between 0.7 & 0.9",
                          #                       "Target Below 0.7"),
                          # facet_level_order = list(c("Targeted Drug", "Untargeted Drug"),
                          #                          c("Target Above 0.7", "Target Below 0.7")),
                          cur_comparisons = list(c("LMF + GNN", "LDS + LMF + GNN")),
                          target_sub_by = c("Target Above 0.7"),
                          legend_title = "Model Type:",
                          calculate_avg_mae = F, y_lab = "Total RMSE Loss",
                          test = "ks.test", paired = T,
                          y_lim = 0.1)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))
ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LDS_SplitByCellLine_Comparison_ViolinPlot.pdf",
       height = 8)


## Split Comparison ====
# GNN + LMF - LDS
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    drug_type == "Base Model + GNN" &
                                                    loss_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(split_method),
                          bar_level_order = c("Split By Both Cell Line & Drug Scaffold", "Split By Cell Line", "Split By Drug Scaffold"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.9",
                                                "Target Between 0.7 & 0.9",
                                                "Target Below 0.7"),
                          legend_title = "Split Method:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_per_fold_Trifecta_without_LDS_Split_Comparison.pdf")

# Bi-modal LMF + LDS without GNN ====
all_results <- fread("Data/all_results.csv")
all_results <- all_results[nchar(data_types) <= 5]

## Upper vs Lower AAC Range ====
all_results_copy <- all_results
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "bottleneck", "TargetRange", "Targeted")
# all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

# Must rename some columns to better distinguish differences on the plot
all_results_copy[drug_type == "Base Model", drug_type := "LDS + LMF"]
all_results_copy[drug_type == "Base Model + GNN", drug_type := "LDS + LMF + GNN"]

table(all_results_copy[merge_method == "Base Model + LMF" &
                         loss_type == "Base Model + LDS" & nchar(data_types) <= 5 &
                         split_method == "Split By Both Cell Line & Drug Scaffold" &
                         bottleneck == "No Data Bottleneck"]$drug_type)
### Split By Both Cell Line & Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_GNN_Upper_vs_Lower_SplitByBoth_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("LDS + LMF", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_GNN_SplitByBoth_Comparison_BoxPlot.pdf")

### Split By Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    TargetRange == "Target Above 0.7" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "TargetRange"),
                          facet_level_order = list(c("Targeted Drug", "Untargeted Drug"),
                                                   c("Target Above 0.7", "Target Below 0.7")),
                          legend_title = "Model Type:",
                          calculate_avg_mae = F,
                          y_lab = "Total RMSE Loss",
                          add_mean = F,
                          y_lim = 0.05)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))
ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_GNN_Upper_vs_Lower_SplitByDrugScaffold_Comparison_BarPlot.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("LDS + LMF", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_GNN_SplitByDrugScaffold_Comparison_BoxPlot.pdf")

# Violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    TargetRange == "Target Above 0.7" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "data_types"),
                          facet_level_order = NULL,
                          # facet_level_order = list(c("Targeted Drug", "Untargeted Drug"),
                          #                          c("Target Above 0.7", "Target Below 0.7")),
                          legend_title = "Model Type:",
                          y_lab = "Total RMSE Loss",
                          add_mean = F,
                          plot_type = "violin_plot",
                          cur_comparisons = list(c("LDS + LMF", "LDS + LMF + GNN")),
                          test = "ks.test",
                          paired = T,
                          y_lim = 0.05)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))
ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_GNN_Upper_vs_Lower_SplitByDrugScaffold_Comparison_ViolinPlot.pdf")

### Split By Cell Line ====
table(all_results_copy[merge_method == "Base Model + LMF" &
                         loss_type == "Base Model + LDS" & nchar(data_types) <= 5 &
                         split_method == "Split By Cell Line" &
                         bottleneck == "No Data Bottleneck"]$drug_type)

cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    TargetRange == "Target Above 0.7" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "TargetRange"),
                          facet_level_order = list(c("Targeted Drug", "Untargeted Drug"),
                                                   c("Target Above 0.7", "Target Below 0.7")),
                          legend_title = "Model Type:",
                          calculate_avg_mae = F, y_lab = "Total RMSE Loss",
                          add_mean = T,
                          y_lim = 0.05)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))
ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_GNN_Upper_vs_Lower_SplitByCellLine_Comparison_BarPlot.pdf",
       height = 10)

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("LDS + LMF", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_GNN_SplitByCellLine_Comparison_BoxPlot.pdf")

# Violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "violin_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("LDS + LMF", "LDS + LMF + GNN")),
                          test = "ks.test",
                          paired = T
)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))
ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_GNN_SplitByCellLine_Comparison_ViolinPlot.pdf",
       height = 8)

### Split Comparison ====
# LDS + LMF - GNN
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(split_method),
                          bar_level_order = c("Split By Both Cell Line & Drug Scaffold", "Split By Cell Line", "Split By Drug Scaffold"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Split Method:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_per_fold_Trifecta_without_GNN_Upper_vs_Lower_Split_Comparison.pdf")

## Targeted vs Untargeted Drugs ==== 
all_results_copy <- all_results[TargetRange == "Target Above 0.7"]
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "bottleneck", "Targeted")
all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

### Split By Both Cell Line & Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = quote(Targeted),
                          facet_level_order = c("Targeted Drug", "Untargeted Drug"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_without_GNN_Targeted_vs_Untargeted_Upper_0.7_SplitByBoth_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("LDS + LMF", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_without_GNN_Targeted_vs_Untargeted_Upper_0.7_SplitByBoth_Comparison_BoxPlot.pdf",
       height = 8)

### Split By Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = quote(Targeted),
                          facet_level_order = c("Targeted Drug", "Untargeted Drug"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_without_GNN_Targeted_vs_Untargeted_Upper_0.7_SplitByDrugScaffold_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("LDS + LMF", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_without_GNN_Targeted_vs_Untargeted_Upper_0.7_SplitByDrugScaffold_Comparison_BoxPlot.pdf",
       height = 8)

### Split By Cell Line ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = quote(Targeted),
                          facet_level_order = c("Targeted Drug", "Untargeted Drug"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_without_GNN_Targeted_vs_Untargeted_Upper_0.7_SplitByCellLine_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(drug_type),
                          bar_level_order = c("LDS + LMF", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("LDS + LMF", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Trifecta_without_GNN_Targeted_vs_Untargeted_Upper_0.7_SplitByCellLine_Comparison_BoxPlot.pdf",
       height = 8)

### Split Comparison ====
# LDS + LMF - GNN, Upper Range, Targeted vs Untargeted
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((merge_method == "Base Model + LMF" &
                                                    loss_type == "Base Model + LDS" &
                                                    drug_type == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(split_method),
                          bar_level_order = c("Split By Both Cell Line & Drug Scaffold", "Split By Cell Line", "Split By Drug Scaffold"),
                          data_order = data_order,
                          facet_by = quote(Targeted),
                          facet_level_order = c("Targeted Drug", "Untargeted Drug"),
                          legend_title = "Drug Model:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_per_fold_Trifecta_without_GNN_Targeted_vs_Untargeted_Upper_0.7_Split_Comparison.pdf")


# Bi-modal LDS + GNN without LMF ====
all_results <- fread("Data/all_results.csv")
all_results <- all_results[nchar(data_types) <= 5]

all_results_copy <- all_results

avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "bottleneck", "TargetRange", "Targeted")
# all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

# Must rename some columns to better distinguish differences on the plot
all_results_copy[merge_method == "Base Model", merge_method := "LDS + GNN"]
all_results_copy[merge_method == "Base Model + LMF", merge_method := "LDS + LMF + GNN"]
all_results_copy[merge_method == "Base Model + Sum", merge_method := "LDS + Sum + GNN"]

table(all_results_copy[(loss_type == "Base Model + LDS" &
                          drug_type == "Base Model + GNN" &
                          nchar(data_types) <= 5 &
                          split_method == "Split By Drug Scaffold" &
                          bottleneck == "No Data Bottleneck")]$merge_method)

## Split By Both Cell Line & Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((loss_type == "Base Model + LDS" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("LDS + GNN", "LDS + Sum + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LMF_SplitByBoth_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((loss_type == "Base Model + LDS" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("LDS + GNN", "LDS + Sum + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("LDS + GNN", "LDS + Sum + GNN"),
                                                 c("LDS + Sum + GNN", "LDS + LMF + GNN"),
                                                 c("LDS + GNN", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LMF_SplitByBoth_Comparison_BoxPlot.pdf")

## Split By Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((loss_type == "Base Model + LDS" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("LDS + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LMF_SplitByDrugScaffold_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((loss_type == "Base Model + LDS" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Drug Scaffold" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("LDS + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("LDS + GNN", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LMF_SplitByDrugScaffold_Comparison_BoxPlot.pdf")

## Split By Cell Line ====
avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "bottleneck", "TargetRange", "Targeted")

cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((loss_type == "Base Model + LDS" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("LDS + GNN", "LDS + Sum + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          plot_type = "bar_plot",
                          facet_by = c("Targeted", "TargetRange"),
                          facet_level_order = list(c("Targeted Drug", "Untargeted Drug"),
                                                   c("Target Above 0.7", "Target Below 0.7")),
                          legend_title = "Model Type:",
                          calculate_avg_mae = F, y_lab = "Total RMSE Loss",
                          y_lim = 0.1)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LMF_SplitByCellLine_Comparison_BarPlot.pdf",
       height = 10)

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((loss_type == "Base Model + LDS" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("LDS + GNN", "LDS + Sum + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("LDS + GNN", "LDS + Sum + GNN"),
                                                 c("LDS + Sum + GNN", "LDS + LMF + GNN"),
                                                 c("LDS + GNN", "LDS + LMF + GNN")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LMF_SplitByCellLine_Comparison_BoxPlot.pdf")

# Violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((loss_type == "Base Model + LDS" &
                                                    drug_type == "Base Model + GNN" &
                                                    nchar(data_types) <= 5 &
                                                    split_method == "Split By Cell Line" &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(merge_method),
                          bar_level_order = c("LDS + GNN", "LDS + Sum + GNN", "LDS + LMF + GNN"),
                          data_order = data_order,
                          facet_by = c("Targeted", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "violin_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("LDS + GNN", "LDS + Sum + GNN"),
                                                 c("LDS + Sum + GNN", "LDS + LMF + GNN"),
                                                 c("LDS + GNN", "LDS + LMF + GNN")),
                          test = "ks.test", step_increase = 0.075,
                          paired = T
)
cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold")) +
  expand_limits(y = c(0, 1.7))

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_minus_LMF_SplitByCellLine_Comparison_ViolinPlot.pdf",
       height = 12)

## Split Comparison ====
# LDS + GNN - LMF
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote((loss_type == "Base Model + LDS" &
                                                    drug_type == "Base Model + GNN" &
                                                    merge_method == "Base Model" &
                                                    nchar(data_types) <= 5 &
                                                    bottleneck == "No Data Bottleneck")),
                          fill_by = quote(split_method),
                          bar_level_order = c("Split By Both Cell Line & Drug Scaffold", "Split By Cell Line", "Split By Drug Scaffold"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Split Method:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_per_fold_Trifecta_without_LMF_Split_Comparison.pdf")

# Bi-modal Baseline vs Trifecta ====
# all_results <- fread("Data/all_results.csv")
all_results_copy <- all_results

avg_loss_by <- c("data_types", "merge_method", "loss_type", "drug_type",
                 "split_method", "fold", "bottleneck", "TargetRange", "Targeted")
# all_results_copy[, loss_by_config := mean(RMSELoss), by = avg_loss_by]
data_order <- c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA")

all_results_copy[(merge_method == "Base Model + LMF" & drug_type == "Base Model + GNN" & loss_type == "Base Model + LDS"), config_type := "Trifecta"]
all_results_copy[(merge_method == "Base Model" & drug_type == "Base Model" & loss_type == "Base Model"), config_type := "Baseline"]
all_results_copy <- all_results_copy[config_type == "Trifecta" | config_type == "Baseline"]

avg_loss_by <- c(avg_loss_by, "config_type")

table(all_results_copy[split_method == "Split By Both Cell Line & Drug Scaffold" &
                   nchar(data_types) <= 5 &
                   bottleneck == "No Data Bottleneck"]$config_type)

## Split By Both Cell Line & Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote(split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                   nchar(data_types) <= 5 &
                                                   bottleneck == "No Data Bottleneck"),
                          fill_by = quote(config_type),
                          bar_level_order = c("Baseline", "Trifecta"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_Trifecta_SplitByBoth_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote(split_method == "Split By Both Cell Line & Drug Scaffold" &
                                                   nchar(data_types) <= 5 &
                                                   bottleneck == "No Data Bottleneck"),
                          fill_by = quote(config_type),
                          bar_level_order = c("Baseline", "Trifecta"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Baseline", "Trifecta")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_Trifecta_SplitByBoth_Comparison_BoxPlot.pdf")

## Split By Drug Scaffold ====
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote(split_method == "Split By Drug Scaffold" &
                                                   nchar(data_types) <= 5 &
                                                   bottleneck == "No Data Bottleneck"),
                          fill_by = quote(config_type),
                          bar_level_order = c("Baseline", "Trifecta"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_Trifecta_SplitByDrugScaffold_Comparison.pdf")

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote(split_method == "Split By Drug Scaffold" &
                                                   nchar(data_types) <= 5 &
                                                   bottleneck == "No Data Bottleneck"),
                          fill_by = quote(config_type),
                          bar_level_order = c("Baseline", "Trifecta"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Baseline", "Trifecta")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_Trifecta_SplitByDrugScaffold_Comparison_BoxPlot.pdf")

## Split By Cell Line ====
table(all_results_copy[split_method == "Split By Cell Line" &
                         nchar(data_types) <= 5 &
                         bottleneck == "No Data Bottleneck"]$config_type)

cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote(split_method == "Split By Cell Line" &
                                                   nchar(data_types) <= 5 &
                                                   bottleneck == "No Data Bottleneck"),
                          fill_by = quote(config_type),
                          bar_level_order = c("Baseline", "Trifecta"),
                          data_order = data_order,
                          facet_by = c("Targeted", "TargetRange"),
                          facet_level_order = list(c("Targeted Drug", "Untargeted Drug"),
                                                   c("Target Above 0.7", "Target Below 0.7")),
                          legend_title = "Model Type:",
                          plot_type = "bar_plot",
                          y_lab = "Total RMSE Loss", calculate_avg_mae = F,
                          y_lim = 0.05)

cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold"))

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_Trifecta_SplitByCellLine_Comparison_BarPlot.pdf",
       height = 10)

# Box plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote(split_method == "Split By Cell Line" &
                                                   nchar(data_types) <= 5 &
                                                   bottleneck == "No Data Bottleneck"),
                          fill_by = quote(config_type),
                          bar_level_order = c("Baseline", "Trifecta"),
                          data_order = data_order,
                          facet_by = "data_types",
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.05,
                          plot_type = "box_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Baseline", "Trifecta")),
                          test = "wilcox.test",
                          paired = F
)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_Trifecta_SplitByCellLine_Comparison_BoxPlot.pdf")

# Violin plot
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote(split_method == "Split By Cell Line" &
                                                   nchar(data_types) <= 5 &
                                                   bottleneck == "No Data Bottleneck"),
                          fill_by = quote(config_type),
                          bar_level_order = c("Baseline", "Trifecta"),
                          data_order = data_order,
                          facet_by = c("Targeted", "data_types"),
                          facet_level_order = NULL,
                          legend_title = "Model Type:",
                          y_lim = 0.1,
                          plot_type = "violin_plot",
                          target_sub_by = "Target Above 0.7",
                          cur_comparisons = list(c("Baseline", "Trifecta")),
                          test = "ks.test",
                          paired = T
)
cur_p <- cur_p + theme(text = element_text(size = 14, face = "bold")) + expand_limits(y = c(0, 1.3))

ggsave(plot = cur_p,
       filename = "Plots/CV_Results/Bimodal_CV_Baseline_vs_Trifecta_SplitByCellLine_Comparison_ViolinPlot.pdf",
       height = 8)

## Split Comparison ====
# Trifecta by splitting method
cur_p <- my_plot_function(avg_loss_by = avg_loss_by,
                          sub_results_by = quote(config_type == "Trio" &
                                                   nchar(data_types) <= 5 &
                                                   bottleneck == "No Data Bottleneck"),
                          fill_by = quote(split_method),
                          bar_level_order = c("Split By Both Cell Line & Drug Scaffold", "Split By Cell Line", "Split By Drug Scaffold"),
                          data_order = data_order,
                          facet_by = quote(TargetRange),
                          facet_level_order = c("Target Above 0.7",
                                                "Target Below 0.7"),
                          legend_title = "Model Type:",
                          y_lim = 0.05)

ggsave(plot = cur_p, filename = "Plots/CV_Results/Bimodal_CV_Trifecta_Split_Comparison.pdf")

# Trimodal Baseline vs Trifecta (Split By Both Cell Line & Drug Scaffold) ====
# install.packages("gt")
require(gt)
library(stringr)
all_results_copy <- all_results[str_count(data_types, "_") == 1]
all_results_copy[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange")]
# all_results_copy[, Targeted := ifelse(cpd_name %in% targeted_drugs, T, F)]

all_results_long_copy <- melt(unique(all_results_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config", "TargetRange")]),
                              id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange"))

all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange")]
all_results_long_copy[, cv_sd := sd(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange")]
length(unique(all_results_long_copy$data_types))  # 28 unique trimodal combinations

baseline_vs_trifecta <- all_results_long_copy[split_method == "Split By Both Cell Line & Drug Scaffold" & ((drug_type == "Base Model + GNN" &
                                                                                 merge_method == "Base Model + LMF" &
                                                                                 loss_type == "Base Model + LDS") | 
                                                                                (drug_type == "Base Model" &
                                                                                merge_method == "Base Model" &
                                                                                loss_type == "Base Model"))]

baseline_vs_trifecta[split_method == "Split By Both Cell Line & Drug Scaffold" & ((drug_type == "Base Model + GNN" &
                                                        merge_method == "Base Model + LMF" &
                                                        loss_type == "Base Model + LDS")), config_type := "Trio "]
baseline_vs_trifecta[split_method == "Split By Both Cell Line & Drug Scaffold" & ((drug_type == "Base Model" &
                                                        merge_method == "Base Model" &
                                                        loss_type == "Base Model")), config_type := "Baseline"]
# baseline_with_lmf <- all_results_long_copy[(nchar(data_types) > 5)]
dodge2 <- position_dodge2(width = 0.9, padding = 0)
cur_data <- unique(baseline_vs_trifecta[,-c("fold", "value")])
# Split data types column (cool function!)
cur_data[, c("data_1", "data_2") := tstrsplit(data_types, "_", fixed = T)]

gt(cur_data, rowname_col = "data_1") %>%
  tab_header(title = "Comparison of Baseline ANN and Trio of techniques in the tri-modal case",
                            subtitle = "5-fold validation RMSE loss using strict splitting")
  
  
p <- ggplot(cur_data) +
  geom_bar(mapping = aes(x = data_types, y = cv_mean, fill = config_type), stat = "identity", position='dodge') +
  facet_wrap(~TargetRange, ncol = 2) + 
  scale_fill_discrete(name = "CV Fold:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  geom_errorbar(aes(x=data_types,
                    y=cv_mean,
                    ymax=cv_mean + cv_sd, 
                    ymin=cv_mean - cv_sd, col='red'),
                linetype=1, show.legend = FALSE, position = dodge2, width = 0.9) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank()) +
  ylab("RMSE Loss") +
  ylim(0, max(cur_data$cv_mean) + max(cur_data$cv_sd) + 0.05) +
  ggtitle(label = tools::toTitleCase("Comparison of Baseline ANN and Trio of techniques in the tri-modal case"),
          subtitle = "5-fold validation RMSE loss using strict splitting") +
  geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean), vjust = -0.5)

ggsave(plot = p, filename = "Plots/CV_Results/Trimodal_CV_per_fold_Baseline_vs_Trifecta_SplitByBoth_Comparison.pdf",
       width = 24, height = 16, units = "in")



# Tri-modal Baseline Bottleneck Comparison (split by cell line) ====
all_results_copy <- all_results
# all_results_copy_sub <- all_results_copy[TargetRange == "TargetAbove 0.7"]
all_results_copy[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "bottleneck")]
# all_results_copy[, Targeted := ifelse(cpd_name %in% targeted_drugs, T, F)]

all_results_long_copy <- melt(unique(all_results_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config", "TargetRange", "bottleneck")]),
                              id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange", "bottleneck"))

all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange", "bottleneck")]
all_results_long_copy[, cv_sd := sd(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange", "bottleneck")]

baseline <- all_results_long_copy[(split_method == "Split By Cell Line" & merge_method == "Base Model" & loss_type == "Base Model" &
                                     drug_type == "Base Model" & nchar(data_types) > 6)]
dodge2 <- position_dodge2(width = 0.9, padding = 0)
cur_data <- unique(baseline[,-c("fold", "value")])

p <- ggplot(cur_data) +
  geom_bar(mapping = aes(x = data_types, y = cv_mean,
                         fill = factor(bottleneck,
                                       levels = c("With Data Bottleneck",
                                                  "No Data Bottleneck"))),
           stat = "identity", position='dodge') +
  facet_wrap(~factor(TargetRange,
                     levels = c("Target Above 0.7",
                                "Target Below 0.7")), ncol = 2) + 
  geom_errorbar(aes(x=data_types,
                    y=cv_mean,
                    ymax=cv_mean + cv_sd, 
                    ymin=cv_mean - cv_sd, col='red'),
                linetype=1, show.legend = FALSE, position = dodge2, width = 0.9) +
  scale_fill_discrete(name = "Loss Type:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = c(.9,.85)) +
  ylab("RMSE Loss") +
  ylim(0, max(cur_data$cv_mean) + max(cur_data$cv_sd) + 0.05)
  # ggtitle(label = tools::toTitleCase("Comparison of LDS Loss Weighting across three true AAC range groups"),
  #         subtitle = "5-fold validation RMSE loss using strict splitting by cell lines") +
  # geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean + cv_sd),
  #           vjust = 0.5, hjust = -0.25, angle = 90, position = position_dodge2(width = .9))

ggsave(plot = p, filename = "Plots/CV_Results/Trimodal_CV_Baseline_Bottleneck_Comparison.pdf")
# width = 24, height = 16, units = "in")

# Tri-modal Trifecta (Splitting Comparison) ====
all_results_copy <- all_results
# all_results_copy[target > 0.7 & target < 0.9]$TargetRange <- "Target Between 0.7 & 0.9"
# all_results_copy[target >= 0.9]$TargetRange <- "Target Above 0.9"
all_results_copy[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange")]
# all_results_copy[, Targeted := ifelse(cpd_name %in% targeted_drugs, T, F)]

all_results_long_copy <- melt(unique(all_results_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config", "TargetRange")]),
                              id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange"))

all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange")]
all_results_long_copy[, cv_sd := sd(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange")]


# trifecta_vs_baseline <- all_results_long_copy[((merge_method == "Base Model + LMF" & drug_type == "Base Model + GNN" & loss_type == "Base Model + LDS") |
#                                                  (merge_method == "Base Model" & drug_type == "Base Model" & loss_type == "Base Model")) &
#                                                 split_method == "Split By Both Cell Line & Drug Scaffold" & nchar(data_types) <= 5]
trifecta <- all_results_long_copy[(merge_method == "Base Model + LMF" & drug_type == "Base Model + GNN" &
                                     loss_type == "Base Model + LDS") & nchar(data_types) > 6]

dodge2 <- position_dodge2(width = 0.9, padding = 0)
# cur_data <- unique(trifecta_vs_baseline[,-c("fold", "value")])
cur_data <- unique(trifecta[,-c("fold", "value")])
# cur_data[(merge_method == "Base Model + LMF" & drug_type == "Base Model + GNN" & loss_type == "Base Model + LDS"), config_type := "Trio"]
# cur_data[(merge_method == "Base Model" & drug_type == "Base Model" & loss_type == "Base Model"), config_type := "Baseline"]

p <- ggplot(cur_data) +
  geom_bar(mapping = aes(x = data_types, y = cv_mean, fill = split_method),
           stat = "identity", position='dodge') +
  facet_wrap(~TargetRange, ncol = 2) + 
  scale_fill_discrete(name = "Configuration:") +
  scale_x_discrete() +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  geom_errorbar(aes(x=data_types,
                    y=cv_mean,
                    ymax=cv_mean + cv_sd, 
                    ymin=cv_mean - cv_sd, col='red'),
                linetype=1, show.legend = FALSE, position = dodge2, width = 0.9) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank(),
        legend.position = c(.9,.85)) +
  ylab("RMSE Loss") +
  ylim(0, max(cur_data$cv_mean) + max(cur_data$cv_sd) + 0.05)
  # ggtitle(label = tools::toTitleCase("Comparison of Baseline with LDS + LMF + GNN across two true AAC range groups"),
  #         subtitle = "5-fold validation RMSE loss using strict splitting by both drugs and cell lines") +
  # geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean + cv_sd),
  #           vjust = 0.5, hjust = -0.25, angle = 90, position = position_dodge2(width = .9))

ggsave(plot = p, filename = "Plots/CV_Results/Trimodal_CV_Trifecta_Split_Comparison.pdf")
# width = 24, height = 16, units = "in")




# Trimodal Heatmap for Best Combinations ====
library(stringr)
all_results_copy <- all_results[str_count(data_types, "_") == 1]
all_results_copy[, loss_by_config := rmse(target, predicted), by = c("data_types", "merge_method", "loss_type",
                                                                     "drug_type", "split_method", "bottleneck",
                                                                     "TargetRange")]

# No drug targetedness separation
all_results_copy <- unique(all_results_copy[, c("data_types", "merge_method", "loss_type",
                                                "drug_type", "split_method", "bottleneck",
                                                "TargetRange", "loss_by_config")])

all_results_copy <- all_results_copy[bottleneck == "No Data Bottleneck"]

# all_results_copy[, Targeted := ifelse(cpd_name %in% targeted_drugs, T, F)]

# all_results_long_copy <- melt(unique(all_results_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "bottleneck", "loss_by_config", "TargetRange")]),
#                               id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "bottleneck", "TargetRange"))

# all_results_long_copy[, loss_by_config := rmse(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "bottleneck", "TargetRange")]
# all_results_long_copy[, cv_sd := sd(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "bottleneck", "TargetRange")]
length(unique(all_results_long_copy$data_types))  # 28 unique trimodal combinations

save_pheatmap_pdf <- function(x, filename, width=7, height=7) {
  stopifnot(!missing(x))
  stopifnot(!missing(filename))
  pdf(filename, width=width, height=height)
  grid::grid.newpage()
  grid::grid.draw(x$gtable)
  dev.off()
}

require(pheatmap)
require(igraph)
## Split By Cell Line ====
baseline_trimodal <- all_results_copy[split_method == "Split By Cell Line" & (drug_type == "Base Model" &
                                                                                     merge_method == "Base Model" &
                                                                                     loss_type == "Base Model")]
trifectra_trimodal <- all_results_copy[split_method == "Split By Cell Line" & (drug_type == "Base Model + GNN" &
                                                                                      merge_method == "Base Model + LMF" &
                                                                                      loss_type == "Base Model + LDS")]
baseline_trimodal_cv <- unique(baseline_trimodal[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method",
                                                     "TargetRange", "loss_by_config")])
trifecta_trimodal_cv <- unique(trifectra_trimodal[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method",
                                                      "TargetRange", "loss_by_config")])

all_tri_omic_combos_el <- utils::combn(c("MUT", 'CNV', 'EXP', 'PROT', 'MIRNA', 'METAB', 'HIST', 'RPPA'), 2, simplify = T)
all_tri_omic_combos_el <- t(all_tri_omic_combos_el)

all_tri_omic_combos_el <- cbind(all_tri_omic_combos_el, rep(0.5, 28))
baseline_trimodal_cv[TargetRange == "Target Below 0.7"]
temp <- baseline_trimodal_cv[TargetRange == "Target Below 0.7"]
all_cv_means <- vector(mode = "numeric", length = nrow(temp))
for (i in 1:nrow(temp)) {
  cur_combo <- paste(all_tri_omic_combos_el[i, 1:2], collapse = "_")
  cur_cv_mean <- temp[data_types == cur_combo]$loss_by_config
  all_cv_means[i] <- cur_cv_mean
}

all_tri_omic_combos_el[,3] <- all_cv_means
colnames(all_tri_omic_combos_el) <-  c("first", "second", "Weight")
g=graph.data.frame(all_tri_omic_combos_el)
m <- get.adjacency(g,sparse=FALSE, attr = 'Weight')
storage.mode(m) <- "numeric"
m <- round(m, 4)
m2 <- m
m2[is.na(m)] <- ""

p <- pheatmap(t(m), cluster_rows = FALSE, cluster_cols = FALSE, display_numbers = t(m2), angle_col = "0", legend = F, 
              na_col = "white", border_color = NA, fontsize_number = 12)
save_pheatmap_pdf(p, "Plots/CV_Results/Trimodal_RMSE_Baseline_LowerAAC_SplitByCellLine_Heatmap.pdf", 8, 8)


all_tri_omic_combos_el <- utils::combn(c("MUT", 'CNV', 'EXP', 'PROT', 'MIRNA', 'METAB', 'HIST', 'RPPA'), 2, simplify = T)
all_tri_omic_combos_el <- t(all_tri_omic_combos_el)

all_tri_omic_combos_el <- cbind(all_tri_omic_combos_el, rep(0.5, 28))
baseline_trimodal_cv[TargetRange == "Target Above 0.7"]
temp <- baseline_trimodal_cv[TargetRange == "Target Above 0.7"]
all_cv_means <- vector(mode = "numeric", length = nrow(temp))
for (i in 1:nrow(temp)) {
  cur_combo <- paste(all_tri_omic_combos_el[i, 1:2], collapse = "_")
  cur_cv_mean <- temp[data_types == cur_combo]$loss_by_config
  all_cv_means[i] <- cur_cv_mean
}

all_tri_omic_combos_el[,3] <- all_cv_means
colnames(all_tri_omic_combos_el) <-  c("first", "second", "Weight")
g=graph.data.frame(all_tri_omic_combos_el)
m <- get.adjacency(g,sparse=FALSE, attr = 'Weight')
storage.mode(m) <- "numeric"
m <- round(m, 4)
m2 <- m
m2[is.na(m)] <- ""

p <- pheatmap(t(m), cluster_rows = FALSE, cluster_cols = FALSE, display_numbers = t(m2), angle_col = "0", legend = F, 
              na_col = "white", border_color = NA, fontsize_number = 12)
save_pheatmap_pdf(p, "Plots/CV_Results/Trimodal_RMSE_Baseline_UpperAAC_SplitByCellLine_Heatmap.pdf", 8, 8)


## Split By Both Cell Line & Drug Scaffold ====
baseline_trimodal <- all_results_long_copy[split_method == "Split By Both Cell Line & Drug Scaffold" & (drug_type == "Base Model" &
                                                                                   merge_method == "Base Model" &
                                                                                   loss_type == "Base Model")]
trifectra_trimodal <- all_results_long_copy[split_method == "Split By Both Cell Line & Drug Scaffold" & (drug_type == "Base Model + GNN" &
                                                                                 merge_method == "Base Model + LMF" &
                                                                                 loss_type == "Base Model + LDS")]
baseline_trimodal_cv <- unique(baseline_trimodal[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method",
                                              "TargetRange", "cv_mean")])
trifecta_trimodal_cv <- unique(trifectra_trimodal[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method",
                                              "TargetRange", "cv_mean")])

all_tri_omic_combos_el <- utils::combn(c("MUT", 'CNV', 'EXP', 'PROT', 'MIRNA', 'METAB', 'HIST', 'RPPA'), 2, simplify = T)
all_tri_omic_combos_el <- t(all_tri_omic_combos_el)
# 
# MUT   CNV  
# MUT   EXP  
# MUT   PROT 
# MUT   MIRNA
# MUT   METAB
# MUT   HIST 
# MUT   RPPA 
# CNV   EXP  
# CNV   PROT 
# CNV   MIRNA
# CNV   METAB
# CNV   HIST 
# CNV   RPPA 
# EXP   PROT 
# EXP   MIRNA
# EXP   METAB
# EXP   HIST 
# EXP   RPPA 
# PROT  MIRNA
# PROT  METAB
# PROT  HIST 
# PROT  RPPA 
# MIRNA METAB
# MIRNA HIST 
# MIRNA RPPA 
# METAB HIST 
# METAB RPPA 
# HIST  RPPA

all_tri_omic_combos_el <- cbind(all_tri_omic_combos_el, rep(0.5, 28))
baseline_trimodal_cv[TargetRange == "Target Above 0.7"]
temp <- baseline_trimodal_cv[TargetRange == "Target Above 0.7"]
all_cv_means <- vector(mode = "numeric", length = nrow(temp))
for (i in 1:nrow(temp)) {
  cur_combo <- paste(all_tri_omic_combos_el[i, 1:2], collapse = "_")
  cur_cv_mean <- temp[data_types == cur_combo]$cv_mean
  all_cv_means[i] <- cur_cv_mean
}

all_tri_omic_combos_el[,3] <- all_cv_means
colnames(all_tri_omic_combos_el) <-  c("first", "second", "Weight")

g=graph.data.frame(all_tri_omic_combos_el)
m <- get.adjacency(g,sparse=FALSE, attr = 'Weight')
storage.mode(m) <- "numeric"
m <- round(m, 4)
m2 <- m
m2[is.na(m)] <- ""

# install.packages("pheatmap")
require(pheatmap)
p <- pheatmap(t(m), cluster_rows = FALSE, cluster_cols = FALSE, display_numbers = t(m2), angle_col = "0", legend = F, 
         na_col = "white", border_color = NA, fontsize_number = 12)

save_pheatmap_pdf(p, "Plots/CV_Results/Trimodal_CV_Mean_Baseline_SplitByBoth_Heatmap.pdf", 8, 8)

## Split By Drug Scaffold ====
baseline_trimodal <- all_results_long_copy[split_method == "Split By Drug Scaffold" & (drug_type == "Base Model" &
                                                                                merge_method == "Base Model" &
                                                                                loss_type == "Base Model")]
trifectra_trimodal <- all_results_long_copy[split_method == "Split By Drug Scaffold" & (drug_type == "Base Model + GNN" &
                                                                                 merge_method == "Base Model + LMF" &
                                                                                 loss_type == "Base Model + LDS")]
baseline_trimodal_cv <- unique(baseline_trimodal[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method",
                                                     "TargetRange", "cv_mean")])
trifecta_trimodal_cv <- unique(trifectra_trimodal[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method",
                                                      "TargetRange", "cv_mean")])

all_tri_omic_combos_el <- utils::combn(c("MUT", 'CNV', 'EXP', 'PROT', 'MIRNA', 'METAB', 'HIST', 'RPPA'), 2, simplify = T)
all_tri_omic_combos_el <- t(all_tri_omic_combos_el)

all_tri_omic_combos_el <- cbind(all_tri_omic_combos_el, rep(0.5, 28))
baseline_trimodal_cv[TargetRange == "Target Above 0.7"]
temp <- baseline_trimodal_cv[TargetRange == "Target Above 0.7"]
all_cv_means <- vector(mode = "numeric", length = nrow(temp))
for (i in 1:nrow(temp)) {
  cur_combo <- paste(all_tri_omic_combos_el[i, 1:2], collapse = "_")
  cur_cv_mean <- temp[data_types == cur_combo]$cv_mean
  all_cv_means[i] <- cur_cv_mean
}

all_tri_omic_combos_el[,3] <- all_cv_means
colnames(all_tri_omic_combos_el) <-  c("first", "second", "Weight")
g=graph.data.frame(all_tri_omic_combos_el)
m <- get.adjacency(g,sparse=FALSE, attr = 'Weight')
storage.mode(m) <- "numeric"
m <- round(m, 4)
m2 <- m
m2[is.na(m)] <- ""

p <- pheatmap(t(m), cluster_rows = FALSE, cluster_cols = FALSE, display_numbers = t(m2), angle_col = "0", legend = F, 
              na_col = "white", border_color = NA, fontsize_number = 12)

save_pheatmap_pdf(p, "Plots/CV_Results/Trimodal_CV_Mean_Baseline_SplitByDrugScaffold_Heatmap.pdf", 8, 8)

# Trimodal Baseline vs Trifecta Bar Plot ====
require(ggplot2)
require(grid)
library(stringr)
require(data.table)
dodge2 <- position_dodge2(width = 0.9, padding = 0)
rmse <- function(x, y) sqrt(mean((x - y)^2))


all_results_copy <- fread("Data/all_results.csv")

# all_results_copy <- all_results_copy[str_count(data_types, "_") == 1]

unique_combos <- fread("Data/shared_unique_combinations.csv")
unique_combos[, unique_samples := paste0(cpd_name, "_", cell_name)]
all_results_copy[, unique_samples := paste0(cpd_name, "_", cell_name)]
all_results_copy <- all_results_copy[unique_samples %in% unique_combos$unique_samples]

all_results_copy[, loss_by_config := rmse(target, predicted),
                 by = c("data_types", "merge_method", "loss_type", "drug_type",
                        "split_method", "bottleneck", "TargetRange")]
# all_results_copy[, loss_by_config := rmse(target, predicted),
#                  by = c("data_types", "merge_method", "loss_type", "drug_type",
#                         "split_method", "bottleneck", "TargetRange", "Targeted")]
all_results_copy <- unique(all_results_copy[, c("data_types", "merge_method", "loss_type",
                                                "drug_type", "split_method", "bottleneck",
                                                "TargetRange", "loss_by_config")])
# all_results_copy <- unique(all_results_copy[, c("data_types", "merge_method", "loss_type",
#                                                 "drug_type", "split_method", "bottleneck",
#                                                 "TargetRange", "Targeted", "loss_by_config")])
length(unique(all_results_copy$data_types))  # 28 unique trimodal combinations

all_results_copy <- all_results_copy[bottleneck == "No Data Bottleneck"]

## Split By Both Cell Line ====
# Subset by splitting method and AAC range
all_results_long_copy <-
  all_results_copy[split_method == "Split By Cell Line" &
                     bottleneck == "No Data Bottleneck" &
                     TargetRange == "Target Above 0.7" &
                          ((
                            drug_type == "Base Model" &
                              merge_method == "Base Model" &
                              loss_type == "Base Model"
                          ) | (
                            drug_type == "Base Model + GNN" &
                              merge_method == "Base Model + LMF" &
                              loss_type == "Base Model + LDS"
                          ))]
# Assign model name
all_results_long_copy[(
  drug_type == "Base Model" &
    merge_method == "Base Model" &
    loss_type == "Base Model"
), model_type := "Baseline"]
all_results_long_copy[(
  drug_type == "Base Model + GNN" &
    merge_method == "Base Model + LMF" &
    loss_type == "Base Model + LDS"
), model_type := "Trifecta"]


all_results_long_copy <- unique(all_results_long_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "model_type",
                                                          "TargetRange", "loss_by_config")])
# all_results_long_copy <- unique(all_results_long_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "model_type",
#                                                           "TargetRange", "Targeted", "loss_by_config")])

all_results_long_copy[, first_data := strsplit(data_types, "_", fixed = T)[[1]][1], by = "data_types"]
all_results_long_copy[, second_data := strsplit(data_types, "_", fixed = T)[[1]][2], by = "data_types"]
all_results_long_copy$first_data <- factor(all_results_long_copy$first_data,
                                                levels = c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA"))
all_results_long_copy$second_data <- factor(all_results_long_copy$second_data,
                                                 levels = c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA"))

# all_results_long_copy[, max_config_cv_mean := max(loss_by_config), by = c("data_types")]

# all_top_trimodal[, data_types := factor(data_types, levels = data_order)]
all_results_long_copy[, model_type := factor(unlist(all_results_long_copy[, "model_type", with = F]),
                                                      levels = c("Baseline", "Trifecta"))]

p <- ggplot(all_results_long_copy) +
  geom_bar(mapping = aes(x = model_type,
                         y = loss_by_config,
                         # fill = factor(model_type,
                         #               levels = c("Baseline",
                         #                          "Trifecta"))),
                         fill = factor(Targeted,
                                       levels = c("Untargeted Drug",
                                                  "Targeted Drug"))),
                         # fill = c("Targeted", "model_type")),
           stat = "identity", position='dodge', width = 0.9) +
  scale_color_manual(values = c(NA, 'red'), guide='none') +
  # facet_geo(~ data_types, grid = mygrid,  scales = "free_x",
  #           strip.position = "left",
  #           drop = T
  #           # switch = "x"
  #           ) +
  facet_grid(rows = vars(second_data), cols = vars(first_data),
             scales = "free_x", switch = "both") +
  # scale_x_reordered() +
  # facet_wrap(~second_data + first_data,
  #            scales = "free_x", strip.position = "bottom") +
  scale_fill_discrete(name = "Drug Type:") +
  # scale_x_discrete(name = "Model Type") +
  # scale_x_discrete() +
  # scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
  #                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  # geom_errorbar(aes(x = model_type,
  #                   y=cv_mean,
  #                   ymax=cv_mean + cv_sd,
  #                   ymin=cv_mean - cv_sd, col='red'),
  #               linetype=1, show.legend = FALSE, position = dodge2, width = 0.9, colour = "black") +
  theme(
    text = element_text(size = 20, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    # axis.text.x = element_blank(),
    # axis.ticks = element_blank(),
    axis.title.x = element_blank(),
    legend.direction="horizontal",
    legend.position="top",
    legend.justification="right"
    # strip.background = element_blank(),
    # strip.text.x = element_blank(),
    # legend.position = c(.8,.75)
  ) +
  # legend.position = c(.9,.85)) +
  # ylab("Total RMSE Loss") +
  # ylim(0, max(all_results_long_copy$cv_mean) + max(all_results_long_copy$cv_sd) + 0.05) +
  # ylim(0, 1.2) +
  scale_y_continuous(name = "Total RMSE Loss", limits = c(0, 1.25), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  geom_text(aes(x=model_type, label = round(loss_by_config, 3), angle = 90,
                group = factor(Targeted,
                              levels = c("Untargeted Drug",
                                         "Targeted Drug")),
                y = loss_by_config), vjust = 0.5, hjust = -0.1, position = position_dodge(width = 0.9))

# p <- p + coord_flip()
# all_results_long_copy[data_types %like% "MUT"]

# Get ggplot grob
g = ggplotGrob(p)

# Get the layout dataframe. 
# Note the names.
# g$layout

# gtable::gtable_show_layout(g) # Might also be useful

# Replace the grobs with the nullGrob
cur_patterns <- c("panel-6-7", "panel-5-7", "panel-4-7", "panel-3-7", "panel-2-7", "panel-1-7",
                  "panel-5-6", "panel-4-6", "panel-3-6", "panel-2-6", "panel-1-6",
                  "panel-4-5", "panel-3-5", "panel-2-5", "panel-1-5",
                  "panel-3-4", "panel-2-4", "panel-1-4",
                  "panel-2-3", "panel-1-3",
                  "panel-1-2")
g = ggplotGrob(p)
for (pattern in cur_patterns) {
  pos <- grep(pattern = pattern, g$layout$name)
  g$grobs[[pos]] <- nullGrob()
}

# If you want, move the axis
# g$layout[g$layout$name == "axis-b-2", c("t", "b")] = c(8, 8)

# Draw the plot
grid.newpage()
grid.draw(g)
  
ggsave(filename = "Plots/CV_Results/Trimodal_CV_Baseline_vs_Trifecta_BarPlot_Comparison_Grid.pdf",
       plot = g,
       height = 12, units = "in")  


cur_func <- function(data_name) {
  if (!is.na(data_name)) {
    return(all_results_long_copy[first_data == data_name &
                                   is.na(second_data)]$loss_by_config)
  } else {
    return(NA)
  }
}

all_results_long_copy <- all_results_long_copy[str_count(data_types, "_") < 2]
all_results_long_copy <- all_results_long_copy[model_type == "Baseline"]
all_results_long_copy$first_loss <- sapply(all_results_long_copy$first_data, cur_func)
all_results_long_copy$second_loss <- sapply(all_results_long_copy$second_data, cur_func)

all_results_long_copy <- all_results_long_copy[!is.na(second_data)]

molten_results <- melt(all_results_long_copy[, c("first_data", "second_data",
                               "first_loss", "second_loss",
                               "loss_by_config")],
     id.vars = c("first_data", "second_data"),
     measure.vars = c("first_loss", "second_loss", "loss_by_config"))

molten_results[variable == "first_loss", variable := "Bimodal 1"]
molten_results[variable == "second_loss", variable := "Bimodal 2"]
molten_results[variable == "loss_by_config", variable := "Trimodal"]
# Compare BiModal and TriModal Performances
p <- ggplot(molten_results) +
  geom_bar(mapping = aes(x = variable,
                         y = value,
                         fill = factor(variable,
                                       levels = c("Bimodal 1",
                                                  "Bimodal 2",
                                                  "Trimodal"))),
           stat = "identity", position='dodge', width = 0.9) +
  scale_color_manual(values = c(NA, 'red'), guide='none') +
  facet_grid(rows = vars(second_data), cols = vars(first_data),
             scales = "free_x", switch = "both") +
  # scale_x_reordered() +
  # facet_wrap(~second_data + first_data,
  #            scales = "free_x", strip.position = "bottom") +
  scale_fill_discrete(name = "Drug Type:") +
  # scale_x_discrete(name = "Model Type") +
  # scale_x_discrete() +
  # scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
  #                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  # geom_errorbar(aes(x = model_type,
  #                   y=cv_mean,
  #                   ymax=cv_mean + cv_sd,
  #                   ymin=cv_mean - cv_sd, col='red'),
  #               linetype=1, show.legend = FALSE, position = dodge2, width = 0.9, colour = "black") +
  theme(
    text = element_text(size = 20, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    # axis.text.x = element_blank(),
    # axis.ticks = element_blank(),
    axis.title.x = element_blank(),
    # legend.direction="horizontal",
    # legend.position="top",
    # legend.justification="right"
    # strip.background = element_blank(),
    # strip.text.x = element_blank(),
    # legend.position = c(.8,.75)
    legend.position = "none"
  ) +
  # legend.position = c(.9,.85)) +
  # ylab("Total RMSE Loss") +
  # ylim(0, max(all_results_long_copy$cv_mean) + max(all_results_long_copy$cv_sd) + 0.05) +
  # ylim(0, 1.2) +
  # scale_y_continuous(name = "Total RMSE Loss", limits = c(0, .5), breaks = c(0, 0.15, 0.2, 0.25, 0.35, 0.45)) +
  scale_y_continuous(name = "Total RMSE Loss", limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  geom_text(aes(x=variable, label = round(value, 3), angle = 90,
                group = factor(variable,
                               levels = c("Bimodal 1",
                                          "Bimodal 2",
                                          "Trimodal")),
                y = value), vjust = 0.5, hjust = -0.1, position = position_dodge(width = 0.9))

g = ggplotGrob(p)

# Get the layout dataframe. 
# Note the names.
# g$layout

# gtable::gtable_show_layout(g) # Might also be useful

# Replace the grobs with the nullGrob
cur_patterns <- c("panel-6-7", "panel-5-7", "panel-4-7", "panel-3-7", "panel-2-7", "panel-1-7",
                  "panel-5-6", "panel-4-6", "panel-3-6", "panel-2-6", "panel-1-6",
                  "panel-4-5", "panel-3-5", "panel-2-5", "panel-1-5",
                  "panel-3-4", "panel-2-4", "panel-1-4",
                  "panel-2-3", "panel-1-3",
                  "panel-1-2")
g = ggplotGrob(p)
for (pattern in cur_patterns) {
  pos <- grep(pattern = pattern, g$layout$name)
  g$grobs[[pos]] <- nullGrob()
}

# If you want, move the axis
# g$layout[g$layout$name == "axis-b-2", c("t", "b")] = c(8, 8)

# Draw the plot
grid.newpage()
grid.draw(g)

ggsave(filename = "Plots/CV_Results/Trimodal_vs_Bimodal_Baseline_BarPlot_Comparison_Grid.pdf",
       plot = g,
       height = 12, units = "in")  

# Repeat for Trifecta Models
cur_func <- function(data_name) {
  if (!is.na(data_name)) {
    return(all_results_long_copy[first_data == data_name &
                                   is.na(second_data)]$loss_by_config)
  } else {
    return(NA)
  }
}

all_results_long_copy <- all_results_long_copy[str_count(data_types, "_") < 2]
all_results_long_copy <- all_results_long_copy[model_type == "Trifecta"]
all_results_long_copy$first_loss <- sapply(all_results_long_copy$first_data, cur_func)
all_results_long_copy$second_loss <- sapply(all_results_long_copy$second_data, cur_func)
cur_func("RPPA")
cur_func(NA)

all_results_long_copy <- all_results_long_copy[!is.na(second_data)]

molten_results <- melt(all_results_long_copy[, c("first_data", "second_data",
                                                 "first_loss", "second_loss",
                                                 "loss_by_config")],
                       id.vars = c("first_data", "second_data"),
                       measure.vars = c("first_loss", "second_loss", "loss_by_config"))

molten_results[variable == "first_loss", variable := "Bimodal 1"]
molten_results[variable == "second_loss", variable := "Bimodal 2"]
molten_results[variable == "loss_by_config", variable := "Trimodal"]
# Compare BiModal and TriModal Performances
p <- ggplot(molten_results) +
  geom_bar(mapping = aes(x = variable,
                         y = value,
                         fill = factor(variable,
                                       levels = c("Bimodal 1",
                                                  "Bimodal 2",
                                                  "Trimodal"))),
           # fill = factor(model_type,
           #               levels = c("Baseline",
           #                          "Trifecta"))),
           # fill = factor(Targeted,
           #               levels = c("Untargeted Drug",
           #                          "Targeted Drug"))),
           # fill = c("Targeted", "model_type")),
           stat = "identity", position='dodge', width = 0.9) +
  scale_color_manual(values = c(NA, 'red'), guide='none') +
  # facet_geo(~ data_types, grid = mygrid,  scales = "free_x",
  #           strip.position = "left",
  #           drop = T
  #           # switch = "x"
  #           ) +
  facet_grid(rows = vars(second_data), cols = vars(first_data),
             scales = "free_x", switch = "both") +
  # scale_x_reordered() +
  # facet_wrap(~second_data + first_data,
  #            scales = "free_x", strip.position = "bottom") +
  scale_fill_discrete(name = "Drug Type:") +
  # scale_x_discrete(name = "Model Type") +
  # scale_x_discrete() +
  # scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
  #                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  # geom_errorbar(aes(x = model_type,
  #                   y=cv_mean,
  #                   ymax=cv_mean + cv_sd,
  #                   ymin=cv_mean - cv_sd, col='red'),
  #               linetype=1, show.legend = FALSE, position = dodge2, width = 0.9, colour = "black") +
  theme(
    text = element_text(size = 20, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    # axis.text.x = element_blank(),
    # axis.ticks = element_blank(),
    axis.title.x = element_blank(),
    # legend.direction="horizontal",
    # legend.position="top",
    # legend.justification="right"
    # strip.background = element_blank(),
    # strip.text.x = element_blank(),
    # legend.position = c(.8,.75)
    legend.position = "none"
  ) +
  # legend.position = c(.9,.85)) +
  # ylab("Total RMSE Loss") +
  # ylim(0, max(all_results_long_copy$cv_mean) + max(all_results_long_copy$cv_sd) + 0.05) +
  # ylim(0, 1.2) +
  # scale_y_continuous(name = "Total RMSE Loss", limits = c(0, .5), breaks = c(0, 0.15, 0.2, 0.25, 0.35, 0.45)) + 
  scale_y_continuous(name = "Total RMSE Loss", limits = c(0, 1), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  geom_text(aes(x=variable, label = round(value, 3), angle = 90,
                group = factor(variable,
                               levels = c("Bimodal 1",
                                          "Bimodal 2",
                                          "Trimodal")),
                y = value), vjust = 0.5, hjust = -0.1, position = position_dodge(width = 0.9))

g = ggplotGrob(p)

# Get the layout dataframe. 
# Note the names.
# g$layout

# gtable::gtable_show_layout(g) # Might also be useful

# Replace the grobs with the nullGrob
cur_patterns <- c("panel-6-7", "panel-5-7", "panel-4-7", "panel-3-7", "panel-2-7", "panel-1-7",
                  "panel-5-6", "panel-4-6", "panel-3-6", "panel-2-6", "panel-1-6",
                  "panel-4-5", "panel-3-5", "panel-2-5", "panel-1-5",
                  "panel-3-4", "panel-2-4", "panel-1-4",
                  "panel-2-3", "panel-1-3",
                  "panel-1-2")
g = ggplotGrob(p)
for (pattern in cur_patterns) {
  pos <- grep(pattern = pattern, g$layout$name)
  g$grobs[[pos]] <- nullGrob()
}

# If you want, move the axis
# g$layout[g$layout$name == "axis-b-2", c("t", "b")] = c(8, 8)

# Draw the plot
grid.newpage()
grid.draw(g)

ggsave(filename = "Plots/CV_Results/Trimodal_vs_Bimodal_Trifecta_BarPlot_Comparison_Grid.pdf",
       plot = g,
       height = 12, units = "in")  

# Trimodal Trifecta Splitting Comparison ====
# install.packages("geofacet")
# require(geofacet)
# require(ggforce)
# require(tidytext)
require(ggplot2)
require(grid)
library(stringr)
require(data.table)
dodge2 <- position_dodge2(width = 0.9, padding = 0)
rmse <- function(x, y) sqrt(mean((x - y)^2))

all_results_copy <- fread("Data/all_results.csv")
all_results_copy <- all_results_copy[str_count(data_types, "_") == 1]

unique_combos <- fread("Data/shared_unique_combinations.csv")
unique_combos[, unique_samples := paste0(cpd_name, "_", cell_name)]
all_results_copy[, unique_samples := paste0(cpd_name, "_", cell_name)]
all_results_copy <- all_results_copy[unique_samples %in% unique_combos$unique_samples]

all_results_copy <- all_results_copy[bottleneck == "No Data Bottleneck"]

# grid_design()

# mygrid <- data.frame(
#   code = c("MUT_CNV", "MUT_EXP", "CNV_EXP", "CNV_PROT", "MUT_PROT", "EXP_PROT", "EXP_MIRNA", "CNV_MIRNA", "MUT_MIRNA", "PROT_MIRNA", "MIRNA_METAB", "PROT_METAB", "CNV_METAB", "EXP_METAB", "MUT_METAB", "MIRNA_HIST", "CNV_HIST", "EXP_HIST", "PROT_HIST", "MUT_HIST", "EXP_RPPA", "CNV_RPPA", "PROT_RPPA", "MIRNA_RPPA", "MUT_RPPA", "METAB_HIST", "METAB_RPPA", "HIST_RPPA"),
#   name = c("", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""),
#   row = c(1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 6, 7, 7),
#   col = c(1, 1, 2, 2, 1, 3, 3, 2, 1, 4, 5, 4, 2, 3, 1, 5, 2, 3, 4, 1, 3, 2, 4, 5, 1, 6, 6, 7),
#   stringsAsFactors = FALSE
# )
# geofacet::grid_preview(mygrid)


all_results_copy[, loss_by_config := rmse(target, predicted),
                 by = c("data_types", "merge_method", "loss_type", "drug_type",
                        "split_method", "bottleneck", "TargetRange", "Targeted")]

all_results_copy <- unique(all_results_copy[, c("data_types", "merge_method", "loss_type",
                                                "drug_type", "split_method", "bottleneck",
                                                "TargetRange", "Targeted", "loss_by_config")])
length(unique(all_results_copy$data_types))  # 28 unique trimodal combinations

# all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "bottleneck", "TargetRange")]
# all_results_long_copy[, cv_sd := sd(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "bottleneck", "TargetRange")]
length(unique(all_results_copy$data_types))  # 28 unique trimodal combinations


# Show only trifecta results
all_results_long_copy <-
  all_results_copy[bottleneck == "No Data Bottleneck" &
                     TargetRange == "Target Above 0.7" &
                     (
                       drug_type == "Base Model + GNN" &
                         merge_method == "Base Model + LMF" &
                         loss_type == "Base Model + LDS"
                     )]
# Assign model name
# all_results_long_copy[(
#   drug_type == "Base Model" &
#     merge_method == "Base Model" &
#     loss_type == "Base Model"
# ), model_type := "Baseline"]
all_results_long_copy[(
  drug_type == "Base Model + GNN" &
    merge_method == "Base Model + LMF" &
    loss_type == "Base Model + LDS"
), model_type := "Trifecta"]



all_results_long_copy <- unique(all_results_long_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "model_type",
                                                          "TargetRange", "Targeted", "loss_by_config")])

all_results_long_copy[, first_data := strsplit(data_types, "_", fixed = T)[[1]][1], by = "data_types"]
all_results_long_copy[, second_data := strsplit(data_types, "_", fixed = T)[[1]][2], by = "data_types"]
all_results_long_copy$first_data <- factor(all_results_long_copy$first_data,
                                           levels = c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA"))
all_results_long_copy$second_data <- factor(all_results_long_copy$second_data,
                                            levels = c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA"))

all_results_long_copy[, max_config_cv_mean := max(loss_by_config), by = c("data_types")]

# all_top_trimodal[, data_types := factor(data_types, levels = data_order)]
# all_results_long_copy[, model_type := factor(unlist(all_results_long_copy[, "model_type", with = F]),
#                                              levels = c("Baseline", "Trifecta"))]

table(all_results_long_copy[model_type == "Trifecta"]$data_types)

# baseline_trimodal <-
#   all_results_copy[(
#     drug_type == "Base Model" &
#       merge_method == "Base Model" &
#       loss_type == "Base Model"
#   )]
# trifectra_trimodal <-
#   all_results_copy[(
#     drug_type == "Base Model + GNN" &
#       merge_method == "Base Model + LMF" &
#       loss_type == "Base Model + LDS"
#   )]
# baseline_trimodal_cv <- unique(baseline_trimodal[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method",
#                                                      "TargetRange", "cv_mean")])
# trifecta_trimodal_cv <- unique(trifectra_trimodal[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method",
#                                                       "TargetRange", "cv_mean", "cv_sd")])
# 
# upper_trifecta_trimodal_cv <- trifecta_trimodal_cv[TargetRange == "Target Above 0.7"]
# 
# upper_trifecta_trimodal_cv[, first_data := strsplit(data_types, "_", fixed = T)[[1]][1], by = "data_types"]
# upper_trifecta_trimodal_cv[, second_data := strsplit(data_types, "_", fixed = T)[[1]][2], by = "data_types"]
# upper_trifecta_trimodal_cv$first_data <- factor(upper_trifecta_trimodal_cv$first_data,
#                                                 levels = c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA"))
# upper_trifecta_trimodal_cv$second_data <- factor(upper_trifecta_trimodal_cv$second_data,
#                                                 levels = c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA"))
# 
# upper_trifecta_trimodal_cv[, max_config_cv_mean := max(cv_mean), by = c("data_types")]

all_results_long_copy[split_method == "Split By Both Cell Line & Drug Scaffold",
                      split_method := "Cell Line & Drug Scaffold"]
all_results_long_copy[split_method == "Split By Cell Line",
                      split_method := "Cell Line"]
all_results_long_copy[split_method == "Split By Drug Scaffold",
                      split_method := "Drug Scaffold"]
all_results_long_copy[split_method == "Split By Cancer Type",
                      split_method := "Cancer Type"]
p <- ggplot(all_results_long_copy) +
  geom_bar(mapping = aes(x = split_method,
                         y = loss_by_config,
                         # fill = factor(split_method,
                         #               levels = c("Split By Cell Line",
                         #                          "Split By Drug Scaffold",
                         #                          "Split By Both Cell Line & Drug Scaffold",
                         #                          "Split By Cancer Type")),
                         fill = factor(Targeted,
                                       levels = c("Untargeted Drug",
                                                  "Targeted Drug")),
                         color = loss_by_config == max_config_cv_mean),
           stat = "identity", position='dodge', width = 0.9) +
  scale_color_manual(values = c(NA, 'red'), guide='none') +
  # facet_geo(~ data_types, grid = mygrid,  scales = "free_x",
  #           strip.position = "left",
  #           drop = T
  #           # switch = "x"
  #           ) +
  facet_grid(rows = vars(second_data), cols = vars(first_data),
             scales = "free_x", switch = "both") +
  # scale_x_reordered() +
  # facet_wrap(~second_data + first_data,
  #            scales = "free_x", strip.position = "bottom") +
  scale_fill_discrete(name = "Splitting Method:") +
  # scale_x_discrete() +
  # scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
  #                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  # geom_errorbar(aes(x = split_method,
  #                   y=cv_mean,
  #                   ymax=cv_mean + cv_sd,
  #                   ymin=cv_mean - cv_sd, col='red'),
  #               linetype=1, show.legend = FALSE, position = dodge2, width = 0.9, colour = "black") +
  theme(
    text = element_text(size = 20, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    # axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    # axis.ticks = element_blank(),
    legend.direction="horizontal",
    legend.position="top",
    legend.justification="right"
    # strip.background = element_blank(),
    # strip.text.x = element_blank(),
    # legend.position = c(.8,.75)
  ) +
        # legend.position = c(.9,.85)) +
  # ylab("RMSE Loss") +
  # ylim(0, max(all_results_long_copy$loss_by_config) + 0.1)
  # ylim(0, 1) +
  scale_y_continuous(name = "Total RMSE Loss", limits = c(0, 1.25), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  geom_text(aes(x=split_method, label = round(loss_by_config, 3), angle = 90,
                group = factor(Targeted,
                               levels = c("Untargeted Drug",
                                          "Targeted Drug")),
                y = loss_by_config), vjust = 0.5, hjust = -0.1, position = position_dodge(width = 0.9))


p
# Get ggplot grob
g = ggplotGrob(p)

# Get the layout dataframe. 
# Note the names.
# g$layout

# gtable::gtable_show_layout(g) # Might also be useful

# Replace the grobs with the nullGrob
cur_patterns <- c("panel-6-7", "panel-5-7", "panel-4-7", "panel-3-7", "panel-2-7", "panel-1-7",
                  "panel-5-6", "panel-4-6", "panel-3-6", "panel-2-6", "panel-1-6",
                  "panel-4-5", "panel-3-5", "panel-2-5", "panel-1-5",
                  "panel-3-4", "panel-2-4", "panel-1-4",
                  "panel-2-3", "panel-1-3",
                  "panel-1-2")
g = ggplotGrob(p)
for (pattern in cur_patterns) {
  pos <- grep(pattern = pattern, g$layout$name)
  g$grobs[[pos]] <- nullGrob()
}

# If you want, move the axis
# g$layout[g$layout$name == "axis-b-2", c("t", "b")] = c(8, 8)

# Draw the plot
grid.newpage()
grid.draw(g)


ggsave(filename = "Plots/CV_Results/Trimodal_CV_Trifecta_Split_Comparison_Grid.pdf",
       plot = g,
       height = 12, units = "in")


# ==== Show sample counts for each trimodal combination (DepMap + CTRPv2 overlap)
require(stringr)
line_info <- fread("Data/DRP_Training_Data/DepMap_21Q2_Line_Info.csv")
ctrp <- fread("Data/DRP_Training_Data/CTRP_AAC_SMILES.txt")

exp <- fread("Data/DRP_Training_Data/DepMap_21Q2_Expression.csv")
mut <- fread("Data/DRP_Training_Data/DepMap_21Q2_Mutations_by_Cell.csv")
cnv <- fread("Data/DRP_Training_Data/DepMap_21Q2_CopyNumber.csv")
prot <- fread("Data/DRP_Training_Data/DepMap_20Q2_No_NA_ProteinQuant.csv")

mirna <- fread("Data/DRP_Training_Data/DepMap_2019_miRNA.csv")
metab <- fread("Data/DRP_Training_Data/DepMap_2019_Metabolomics.csv")
hist <- fread("Data/DRP_Training_Data/DepMap_2019_ChromatinProfiling.csv")
rppa <- fread("Data/DRP_Training_Data/DepMap_2019_RPPA.csv")

mut$stripped_cell_line_name = str_replace(toupper(mut$stripped_cell_line_name), "-", "")
cnv$stripped_cell_line_name = str_replace(toupper(cnv$stripped_cell_line_name), "-", "")
exp$stripped_cell_line_name = str_replace(toupper(exp$stripped_cell_line_name), "-", "")
prot$stripped_cell_line_name = str_replace(toupper(prot$stripped_cell_line_name), "-", "")

mirna$stripped_cell_line_name = str_replace(toupper(mirna$stripped_cell_line_name), "-", "")
hist$stripped_cell_line_name = str_replace(toupper(hist$stripped_cell_line_name), "-", "")
metab$stripped_cell_line_name = str_replace(toupper(metab$stripped_cell_line_name), "-", "")
rppa$stripped_cell_line_name = str_replace(toupper(rppa$stripped_cell_line_name), "-", "")

ctrp$ccl_name = str_replace(toupper(ctrp$ccl_name), "-", "")

mut_line_info <- line_info[stripped_cell_line_name %in% unique(mut$stripped_cell_line_name)]  
cnv_line_info <- line_info[stripped_cell_line_name %in% unique(cnv$stripped_cell_line_name)]  
exp_line_info <- line_info[stripped_cell_line_name %in% unique(exp$stripped_cell_line_name)]  
prot_line_info <- line_info[stripped_cell_line_name %in% unique(prot$stripped_cell_line_name)]

mirna_line_info <- line_info[stripped_cell_line_name %in% unique(mirna$stripped_cell_line_name)]  
hist_line_info <- line_info[stripped_cell_line_name %in% unique(hist$stripped_cell_line_name)]  
metab_line_info <- line_info[stripped_cell_line_name %in% unique(metab$stripped_cell_line_name)]  
rppa_line_info <- line_info[stripped_cell_line_name %in% unique(rppa$stripped_cell_line_name)]

ctrp_line_info <- line_info[stripped_cell_line_name %in% unique(ctrp$ccl_name)]

mut_line_info <- mut_line_info[, c("stripped_cell_line_name", "primary_disease")]
mut_line_info$data_type <- "MUT"
cnv_line_info <- cnv_line_info[, c("stripped_cell_line_name", "primary_disease")]
cnv_line_info$data_type <- "CNV"
exp_line_info <- exp_line_info[, c("stripped_cell_line_name", "primary_disease")]
exp_line_info$data_type <- "EXP"
prot_line_info <- prot_line_info[, c("stripped_cell_line_name", "primary_disease")]
prot_line_info$data_type <- "PROT"

mirna_line_info <- mirna_line_info[, c("stripped_cell_line_name", "primary_disease")]
mirna_line_info$data_type <- "MIRNA"
hist_line_info <- hist_line_info[, c("stripped_cell_line_name", "primary_disease")]
hist_line_info$data_type <- "HIST"
metab_line_info <- metab_line_info[, c("stripped_cell_line_name", "primary_disease")]
metab_line_info$data_type <- "METAB"
rppa_line_info <- rppa_line_info[, c("stripped_cell_line_name", "primary_disease")]
rppa_line_info$data_type <- "RPPA"

ctrp_line_info <- ctrp_line_info[, c("stripped_cell_line_name", "primary_disease")]
ctrp_line_info$data_type <- "CTRP"

all_cells <- rbindlist(list(mut_line_info, cnv_line_info, exp_line_info, prot_line_info,
               mirna_line_info, metab_line_info, hist_line_info, rppa_line_info))
all_cells <- unique(all_cells)

rm(list = c("mut", "cnv", "exp", "prot", "mirna", "metab", "hist", "rppa"))
gc()

all_tri_omic_combos_el <- utils::combn(c("MUT", 'CNV', 'EXP', 'PROT', 'MIRNA', 'METAB', 'HIST', 'RPPA'), 2, simplify = T)
all_tri_omic_combos_el <- t(all_tri_omic_combos_el)
all_tri_omic_combos_el <- as.data.table(all_tri_omic_combos_el)

# all_sample_counts <- vector(mode = "numeric", length = nrow(temp))
ctrp_cells <- unique(ctrp_line_info$stripped_cell_line_name)
all_tri_omic_combos_el$sample_counts <- vector(mode = "integer")
for (i in 1:nrow(all_tri_omic_combos_el)) {
  first_cells <- all_cells[data_type == all_tri_omic_combos_el[i, 1]]$stripped_cell_line_name
  second_cells <- all_cells[data_type == all_tri_omic_combos_el[i, 2]]$stripped_cell_line_name
  cell_overlap <- Reduce(intersect, list(first_cells, second_cells, ctrp_cells))
  ctrp_overlap <- uniqueN(ctrp[ccl_name %in% cell_overlap])
  all_tri_omic_combos_el[i, 3] <- ctrp_overlap
}

temp <- trifecta_trimodal_cv[TargetRange == "Target Above 0.7"]

# ==== Trimodal Trifecta minus LMF (Split By Both Cell Line & Drug Scaffold) ====
library(stringr)
all_results_copy <- all_results[str_count(data_types, "_") == 1]
all_results_copy[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange")]
# all_results_copy[, Targeted := ifelse(cpd_name %in% targeted_drugs, T, F)]

all_results_long_copy <- melt(unique(all_results_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config", "TargetRange")]),
                              id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange"))

all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange")]
length(unique(all_results_long_copy$data_types))  # 28 unique trimodal combinations

baseline_with_lmf <- all_results_long_copy[split_method == "SplitByDrugScaffold"]
# baseline_with_lmf <- all_results_long_copy[(nchar(data_types) > 5)]
p <- ggplot(baseline_with_lmf) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = fold), stat = "identity", position='dodge') +
  facet_wrap(~drug_type+merge_method+loss_type+split_method+TargetRange, ncol = 2) + 
  scale_fill_discrete(name = "CV Fold:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of LMF Fusion across two true AAC range groups"),
          subtitle = "5-fold validation RMSE loss using strict splitting") +
  geom_errorbar(aes(x=data_types,
                    y=cv_mean,
                    ymax=cv_mean, 
                    ymin=cv_mean, col='red'), linetype=2, show.legend = FALSE) +
  geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean), vjust = -0.5)

ggsave(plot = p, filename = "Plots/CV_Results/Trimodal_CV_per_fold_Baseline_vs_Trifecta_SplitByBoth_Comparison.pdf",
       width = 24, height = 16, units = "in")


# Multi-modal Baseline vs Trifecta Bar Plot ====
require(ggplot2)
require(grid)
library(stringr)
require(data.table)
dodge2 <- position_dodge2(width = 0.9, padding = 0)
rmse <- function(x, y) sqrt(mean((x - y)^2))


# all_results_copy <- fread("Data/all_results.csv")

all_results_copy <- all_results_copy[str_count(data_types, "_") > 1]

unique_combos <- fread("Data/shared_unique_combinations.csv")
unique_combos[, unique_samples := paste0(cpd_name, "_", cell_name)]
all_results_copy[, unique_samples := paste0(cpd_name, "_", cell_name)]
all_results_copy <- all_results_copy[unique_samples %in% unique_combos$unique_samples]

all_results_copy[, loss_by_config := rmse(target, predicted),
                 by = c("data_types", "merge_method", "loss_type", "drug_type",
                        "split_method", "bottleneck", "TargetRange", "Targeted")]
all_results_copy <- unique(all_results_copy[, c("data_types", "merge_method", "loss_type",
                                                "drug_type", "split_method", "bottleneck",
                                                "TargetRange", "Targeted", "loss_by_config")])
length(unique(all_results_copy$data_types))  # 9 unique multimodal combinations

all_results_copy <- all_results_copy[bottleneck == "No Data Bottleneck"]

## Split By Both Cell Line ====
# Subset by splitting method and AAC range
all_results_long_copy <-
  all_results_copy[split_method == "Split By Cell Line" &
                     bottleneck == "No Data Bottleneck" &
                     TargetRange == "Target Above 0.7" &
                     ((
                       drug_type == "Base Model" &
                         merge_method == "Base Model" &
                         loss_type == "Base Model"
                     ) | (
                       drug_type == "Base Model + GNN" &
                         merge_method == "Base Model + LMF" &
                         loss_type == "Base Model + LDS"
                     ))]
# Assign model name
all_results_long_copy[(
  drug_type == "Base Model" &
    merge_method == "Base Model" &
    loss_type == "Base Model"
), model_type := "Baseline"]
all_results_long_copy[(
  drug_type == "Base Model + GNN" &
    merge_method == "Base Model + LMF" &
    loss_type == "Base Model + LDS"
), model_type := "Trifecta"]

# all_results_long_copy <- unique(all_results_long_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "model_type",
#                                                           "TargetRange", "Targeted", "loss_by_config")])

# all_results_long_copy[, first_data := strsplit(data_types, "_", fixed = T)[[1]][1], by = "data_types"]
# all_results_long_copy[, second_data := strsplit(data_types, "_", fixed = T)[[1]][2], by = "data_types"]
# all_results_long_copy$first_data <- factor(all_results_long_copy$first_data,
#                                            levels = c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA"))
# all_results_long_copy$second_data <- factor(all_results_long_copy$second_data,
#                                             levels = c("MUT", "CNV", "EXP", "PROT", "MIRNA", "METAB", "HIST", "RPPA"))

all_results_long_copy[, max_config_cv_mean := max(loss_by_config), by = c("data_types")]

# all_top_trimodal[, data_types := factor(data_types, levels = data_order)]
all_results_long_copy[, model_type := factor(unlist(all_results_long_copy[, "model_type", with = F]),
                                             levels = c("Baseline", "Trifecta"))]

all_results_long_copy[, data_types := gsub("_", "+", data_types, fixed = T)]
p <- ggplot(all_results_long_copy) +
  geom_bar(mapping = aes(x = model_type,
                         y = loss_by_config,
                         # fill = factor(model_type,
                         #               levels = c("Baseline",
                         #                          "Trifecta"))),
                         fill = factor(Targeted,
                                       levels = c("Untargeted Drug",
                                                  "Targeted Drug"))),
           # fill = c("Targeted", "model_type")),
           stat = "identity", position='dodge', width = 0.9) +
  scale_color_manual(values = c(NA, 'red'), guide='none') +
  # facet_geo(~ data_types, grid = mygrid,  scales = "free_x",
  #           strip.position = "left",
  #           drop = T
  #           # switch = "x"
  #           ) +
  # facet_grid(rows = vars(second_data), cols = vars(first_data),
  #            scales = "free_x", switch = "both") +
  # scale_x_reordered() +
  facet_wrap(~data_types,
             scales = "free_x", strip.position = "bottom") +
  scale_fill_discrete(name = "Drug Type:") +
  # scale_x_discrete(name = "Model Type") +
  # scale_x_discrete() +
  # scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
  #                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  # geom_errorbar(aes(x = model_type,
  #                   y=cv_mean,
  #                   ymax=cv_mean + cv_sd,
  #                   ymin=cv_mean - cv_sd, col='red'),
  #               linetype=1, show.legend = FALSE, position = dodge2, width = 0.9, colour = "black") +
  theme(
    text = element_text(size = 20, face = "bold"),
    # axis.text.x = element_text(angle = 0),
    # axis.text.x = element_blank(),
    # axis.ticks = element_blank(),
    axis.title.x = element_blank(),
    legend.direction="horizontal",
    legend.position="top",
    legend.justification="right"
    # strip.background = element_blank(),
    # strip.text.x = element_blank(),
    # legend.position = c(.8,.75)
  ) +
  # legend.position = c(.9,.85)) +
  # ylab("Total RMSE Loss") +
  # ylim(0, max(all_results_long_copy$cv_mean) + max(all_results_long_copy$cv_sd) + 0.05) +
  # ylim(0, 1.2) +
  scale_y_continuous(name = "Total RMSE Loss", limits = c(0, 1.25), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  geom_text(aes(x=model_type, label = round(loss_by_config, 3), angle = 90,
                group = factor(Targeted,
                               levels = c("Untargeted Drug",
                                          "Targeted Drug")),
                y = loss_by_config), vjust = 0.5, hjust = -0.1, position = position_dodge(width = 0.9))

ggsave(filename = "Plots/CV_Results/Multimodal_CV_Baseline_vs_Trifecta_BarPlot_Comparison_Grid.pdf",
       plot = p,
       height = 12, width = 14, units = "in")  

# p <- p + coord_flip()
# all_results_long_copy[data_types %like% "MUT"]

# Get ggplot grob
g = ggplotGrob(p)

# Get the layout dataframe. 
# Note the names.
# g$layout

# gtable::gtable_show_layout(g) # Might also be useful

# Replace the grobs with the nullGrob
cur_patterns <- c("panel-6-7", "panel-5-7", "panel-4-7", "panel-3-7", "panel-2-7", "panel-1-7",
                  "panel-5-6", "panel-4-6", "panel-3-6", "panel-2-6", "panel-1-6",
                  "panel-4-5", "panel-3-5", "panel-2-5", "panel-1-5",
                  "panel-3-4", "panel-2-4", "panel-1-4",
                  "panel-2-3", "panel-1-3",
                  "panel-1-2")
g = ggplotGrob(p)
for (pattern in cur_patterns) {
  pos <- grep(pattern = pattern, g$layout$name)
  g$grobs[[pos]] <- nullGrob()
}

# If you want, move the axis
# g$layout[g$layout$name == "axis-b-2", c("t", "b")] = c(8, 8)

# Draw the plot
grid.newpage()
grid.draw(g)

ggsave(filename = "Plots/CV_Results/Trimodal_CV_Baseline_vs_Trifecta_BarPlot_Comparison_Grid.pdf",
       plot = g,
       height = 12, units = "in")  




# ==== Multimodal Baseline vs LMF (Split By Both Cell Line & Drug Scaffold) ====
all_results_copy <- all_results
all_results_copy[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange")]
# all_results_copy[, Targeted := ifelse(cpd_name %in% targeted_drugs, T, F)]

all_results_long_copy <- melt(unique(all_results_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config", "TargetRange")]),
                              id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange"))

all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange")]

baseline_with_lmf <- all_results_long_copy[(drug_type == "Morgan" &
                                              split_method == "SplitByBoth" & nchar(data_types) > 5)]
baseline_with_lmf <- all_results_long_copy[(nchar(data_types) > 5)]
p <- ggplot(baseline_with_lmf) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = fold), stat = "identity", position='dodge') +
  facet_wrap(~merge_method+loss_type+split_method+TargetRange, ncol = 2) + 
  scale_fill_discrete(name = "CV Fold:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of LMF Fusion across two true AAC range groups"),
          subtitle = "5-fold validation RMSE loss using strict splitting") +
  geom_errorbar(aes(x=data_types,
                    y=cv_mean,
                    ymax=cv_mean, 
                    ymin=cv_mean, col='red'), linetype=2, show.legend = FALSE) +
  geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean), vjust = -0.5)

ggsave(plot = p, filename = "Plots/CV_Results/Multimodal_CV_per_fold_Baseline_vs_LMF_SplitByBoth_Comparison.pdf",
       width = 24, height = 16, units = "in")
# ggsave(filename = "Plots/CV_Results/Bimodal_CV_per_fold_Baseline_with_GNN_Upper_0.7_Comparison_long.pdf",
#        width = 24, height = 48, units = "in")

# ==== Multimodal Baseline vs LDS (Split By Both Cell Line & Drug Scaffold) ====
all_results_copy <- all_results
all_results_copy[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange")]
# all_results_copy[, Targeted := ifelse(cpd_name %in% targeted_drugs, T, F)]

all_results_long_copy <- melt(unique(all_results_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config", "TargetRange")]),
                              id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "TargetRange"))

all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "TargetRange")]

baseline_with_lmf <- all_results_long_copy[(drug_type == "Morgan" & merge_method == "MergeByConcat" &
                                              split_method == "SplitByBoth" & nchar(data_types) > 5)]
# baseline_with_lmf <- all_results_long_copy[(nchar(data_types) > 5)]
p <- ggplot(baseline_with_lmf) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = fold), stat = "identity", position='dodge') +
  facet_wrap(~merge_method+loss_type+split_method+TargetRange, ncol = 2) + 
  scale_fill_discrete(name = "CV Fold:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of LMF Fusion across two true AAC range groups"),
          subtitle = "5-fold validation RMSE loss using strict splitting") +
  geom_errorbar(aes(x=data_types,
                    y=cv_mean,
                    ymax=cv_mean, 
                    ymin=cv_mean, col='red'), linetype=2, show.legend = FALSE) +
  geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean), vjust = -0.5)

ggsave(plot = p, filename = "Plots/CV_Results/Multimodal_CV_per_fold_Baseline_vs_LMF_SplitByBoth_Comparison.pdf",
       width = 24, height = 16, units = "in")
# ==== Upper Range AAC Comparison ====
# targeted_drug_results <- all_results[cpd_name %in% targeted_drugs]
all_results_copy <- all_results
all_results_copy <- all_results_copy[target >= 0.7]
all_results_copy[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold")]
all_results_copy[, Targeted := ifelse(cpd_name %in% targeted_drugs, T, F)]

all_results_long_copy <- melt(unique(all_results_copy[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config", "Targeted")]),
                              id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "Targeted"))
all_results_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "split_method", "Targeted")]

baseline_with_lds <- all_results_long_copy[(merge_method == "Concat" & drug_type == "DRUG" & split_method == "DRUG")]

ggplot(baseline_with_lds) +
  geom_bar(mapping = aes(x = data_types, y = value, fill = fold), stat = "identity", position='dodge') +
  facet_wrap(~merge_method+loss_type+drug_type+split_method+Targeted, nrow = 2) + 
  scale_fill_discrete(name = "CV Fold:") +
  scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                               "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the bi-modal case"),
          subtitle = "Validation RMSE loss using strict splitting") +
  geom_errorbar(aes(x=data_types,
                    y=cv_mean,
                    ymax=cv_mean, 
                    ymin=cv_mean, col='red'), linetype=2, show.legend = FALSE) +
  geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean), vjust = -0.5)


# ==== 4 targeted drugs ("Gefitinib", "Tamoxifen", "MK-2206", "PLX-4720") ====
temp <- all_results[cpd_name %in% c("Gefitinib", "Tamoxifen", "MK-2206", "PLX-4720", "Imatinib")]
temp[, loss_by_config := mean(RMSELoss), by = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold")]
# temp[, Targeted := ifelse(cpd_name %in% targeted_drugs, T, F)]

# temp_long_copy <- melt(unique(temp[, c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "loss_by_config", "Targeted")]),
#                               id.vars = c("data_types", "merge_method", "loss_type", "drug_type", "split_method", "fold", "Targeted"))
# temp_long_copy[, cv_mean := mean(value), by = c("data_types", "merge_method", "loss_type", "split_method", "Targeted")]
# 
# baseline_with_lds <- temp_long_copy[(merge_method == "Concat" & drug_type == "DRUG" & split_method == "DRUG")]
# se <- function(y) sd(y)/length(y)
temp_baseline_with_lds <- temp[(merge_method == "Concat" & drug_type == "DRUG" & split_method == "DRUG")]
ggplot(data = temp_baseline_with_lds, mapping = aes(x = cpd_name, y = RMSELoss)) +
  # geom_bar(stat = "identity", position='dodge') +
  facet_wrap(~loss_type+split_method+data_types, nrow = 2) + 
  scale_fill_discrete(name = "CV Fold:") +
  # stat_summary_bin(geom = "errorbar", fun.data=function(RMSELoss)c(ymin=mean(RMSELoss)-se(RMSELoss),ymax=mean(RMSELoss)+se(RMSELoss)), position = "dodge") +
  # stat_summary_bin(geom = "errorbar", fun.data='mean', position = "dodge") +
  stat_summary(fun = mean, geom = "bar") +
  stat_summary(fun.data = mean_se, geom = "errorbar") +

  
  # scale_colour_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
  #                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle(label = tools::toTitleCase("Comparison of Loss-weighting, fusion method and drug representation in the bi-modal case"),
          subtitle = "Validation RMSE loss using strict splitting")
  # geom_errorbar(aes(x=data_types,
  #                   y=cv_mean,
  #                   ymax=cv_mean, 
  #                   ymin=cv_mean, col='red'), linetype=2, show.legend = FALSE) +
  # geom_text(aes(x=data_types, label = round(cv_mean, 3), y = cv_mean), vjust = -0.5)
