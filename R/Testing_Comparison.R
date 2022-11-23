# Testing_Comparison.R
require(data.table)
require(Metrics)
require(stringr)
require(ggplot2)
Metrics::
rsq <- function (x, y) {cor(x, y) ^ 2}

all_full_evals <- list.files("Full_Model_Testing_Results/", full.names = T, pattern = "GDSC2")

all_full_metrics <- data.table(data_used = NULL, training_type = NULL, length_of_testing_data = NULL, MSE = NULL,
                               RMSE = NULL, MSLE = NULL, RSQ = NULL, Pearson = NULL)

for (filename in all_full_evals) {
  cur_eval <- fread(filename)
  cur_data <- str_replace(strsplit(basename(filename), "GDSC2")[[1]][1], "-", "_")
  cur_results <- data.table(data_used = cur_data,
                            training_type = "Full",
                            length_of_testing_data = nrow(cur_eval),
                            MSE = mse(cur_eval$actual_target, cur_eval$predicted_target),
                            RMSE = rmse(cur_eval$actual_target, cur_eval$predicted_target),
                            MSLE = msle(cur_eval$actual_target, cur_eval$predicted_target),
                            RSQ = rsq(cur_eval$actual_target, cur_eval$predicted_target),
                            Pearson = cor(cur_eval$actual_target, cur_eval$predicted_target, method = "pearson"))
  all_full_metrics <- rbind(all_full_metrics, cur_results)
}

all_full_metrics_long = melt.data.table(all_full_metrics[, c(1,2,4)], id.vars = c("data_used", "training_type"), variable.name = "Metric", value.name = "Loss")
ggplot(data = all_full_metrics_long) +
  geom_bar(mapping = aes(x = reorder(data_used, -Loss), y = Loss, group = Metric, fill = Metric), position = "dodge", stat = "identity") +
  theme(axis.text.x = element_text(angle = -45, hjust = 0, size = 16))



all_bottleneck_evals <- list.files("BottleNeck_Model_Testing_Results/", full.names = T, pattern = "GDSC2")
all_bottleneck_metrics <- data.table(data_used = NULL, training_type = NULL, length_of_testing_data = NULL, MSE = NULL, RMSE = NULL, MSLE = NULL, RSQ = NULL)

for (filename in all_bottleneck_evals) {
  cur_eval <- fread(filename)
  cur_data <- str_replace(strsplit(basename(filename), "GDSC2")[[1]][1], "-", "_")
  cur_results <- data.table(data_used = cur_data,
                            training_type = "BottleNeck",
                            length_of_testing_data = nrow(cur_eval),
                            MSE = mse(cur_eval$actual_target, cur_eval$predicted_target),
                            RMSE = rmse(cur_eval$actual_target, cur_eval$predicted_target),
                            MSLE = msle(cur_eval$actual_target, cur_eval$predicted_target),
                            RSQ = rsq(cur_eval$actual_target, cur_eval$predicted_target))
  all_bottleneck_metrics <- rbind(all_bottleneck_metrics, cur_results)
}

all_bottleneck_metrics_long = melt.data.table(all_bottleneck_metrics[, c(1,2,4)], id.vars = c("data_used", "training_type"), variable.name = "Metric", value.name = "Loss")

# ggplot(data = all_bottleneck_metrics_long) +
#   geom_bar(mapping = aes(x = reorder(data_used, -Loss), y = Loss, group = Metric, fill = Metric), position = "dodge", stat = "identity") +
#   theme(axis.text.x = element_text(angle = -45, hjust = 0, size = 16))


all_tests <- rbind(all_full_metrics_long, all_bottleneck_metrics_long)
ggplot(data = all_tests) +
  geom_bar(mapping = aes(x = reorder(data_used, -Loss), y = Loss, group = Metric), position = "dodge", stat = "identity") +
  theme(axis.text.x = element_text(angle = -90, hjust = 0, size = 16)) +
  facet_wrap(vars(training_type))
