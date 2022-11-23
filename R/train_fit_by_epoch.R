# train_fit_by_epoch.R
require(data.table)
require(ggplot2)
require(gganimate)
# devtools::install_github("thomasp85/transformr")
require(transformr)

all_epochs <- list.files(path = "Data/EpochResults/{'drp_first_layer_size': 998, 'drp_last_layer_size': 21, 'gnn_out_channels': 243}/", full.names = T)

all_results <- vector(mode = "list", length = length(all_epochs))
for (i in 1:length(all_epochs)) {
  cur_res <- fread(all_epochs[i])
  epoch <- gsub(".+Epoch_(\\d+)_.+", "\\1", all_epochs[i])
  cur_res$epoch <- as.integer(epoch)
  all_results[[i]] <- cur_res
}
all_results <- rbindlist(all_results)

theme_set(theme_bw())

p <- ggplot(data = all_results) +
  geom_freqpoly(alpha = 0.7, aes(x = predicted), binwidth = 0.01, colour = "red") +
  geom_freqpoly(alpha = 0.7, aes(x = target), binwidth = 0.01, colour = "black") + 
  xlab("Area Above Curve") + ylab("Frequency")
p

p + transition_time(epoch) +
  labs(title = "Epoch: {frame_time}")

dir.create("Plots/Train_Fit_Animations/")
anim_save("Plots/Train_Fit_Animations/gnndrug_exp_amsgrad_silu_standardization_nobatchnorm.gif")

rm(epoch)
p <- ggplot(data = all_results, aes(x = predicted, y = target)) +
  geom_point() +
  geom_abline(intercept = 0,
              slope = 1,
              color = "red",
              size = 2) +
  # geom_freqpoly(alpha = 0.7, aes(x = predicted), binwidth = 0.01, colour = "red") +
  # geom_freqpoly(alpha = 0.7, aes(x = target), binwidth = 0.01, colour = "black") + 
  xlab("Predicted") + ylab("Observed") +
  transition_time(epoch) +
  labs(title = "Epoch: {frame_time}")

dir.create("Plots/R2_Line/")
anim_save("Plots/R2_Line/gnndrug_exp_amsgrad_silu_standardization_nobatchnorm.gif")

