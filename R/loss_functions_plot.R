# loss_functions_plot.R
require(ggplot2)
require(cowplot)

plot_activation_function <- function(f, title, range, target){
  ggplot(data.frame(x=range), mapping=aes(x=x)) + 
    geom_hline(yintercept=0, colour='red', alpha=1/4) +
    geom_vline(xintercept=0, colour='red', alpha=1/4) +
    stat_function(fun=f, colour = "dodgerblue3",) +
    ggtitle(title) +
    theme(text = element_text(size = 16), plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle=45, hjust = 1)) +
    xlab("Prediction") + ylab("Loss") +
    scale_x_continuous(breaks=c(0, target, 0.25, 0.5, 0.75, 1))
}

plot_all_activation_functions <- function(f1, f2, f3, title, range, target){
  ggplot(data.frame(x=range), mapping=aes(x=x)) + 
    geom_hline(yintercept=0, color='red', alpha=1/4) +
    geom_vline(xintercept=0, color='red', alpha=1/4) +
    stat_function(fun=f1, aes(colour = "MSE")) +
    stat_function(fun=f2, aes(colour = "MAE")) +
    stat_function(fun=f3, aes(colour = "RMSE")) +
    ggtitle(title) +
    theme(text = element_text(size = 18, face = "bold"),
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle=45, hjust = 1),
          legend.position = "top") +
    xlab("Prediction") + ylab("Loss") +
    scale_x_continuous(breaks=c(0, target, 0.25, 0.5, 0.75, 1)) +
    scale_colour_manual("", 
                        breaks = c("MSE", "MAE", "RMSE"),
                        values = c("MSE"="dodgerblue3", "MAE"="red3", "RMSE"="orange1"))
    
}

set.seed(42)
yhats <- runif(32, min=0, max=1)
mse <- function(yhats, y=0) {((y - yhats) ** 2)/2}
p_mse <- plot_activation_function(mse, 'MSE', c(0,1), 0)
  
mae <- function(yhats, y=0) {abs(y - yhats)/2}
p_mae <- plot_activation_function(mae, 'MAE', c(0,1), 0)

rmse <- function(yhats, y=0) {sqrt(((y - yhats) ** 2)/2)}
p_rmse <- plot_activation_function(rmse, 'RMSE', c(0,1), 0)

plot_all_activation_functions(mse, mae, rmse, '', c(0,2), 0)
ggsave("Plots/All_Loss_Functions_Plot.pdf")

p_grid <- cowplot::plot_grid(p_mse, p_mae, p_rmse, ncol = 3)
  # theme(plot.margin = unit(c(1,31,3,3), "lines"))
ggsave("Plots/Loss_Function_Plots.pdf", plot = p_grid)
