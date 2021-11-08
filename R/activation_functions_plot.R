# activation_functions_plot.R
require(ggplot2)
require(cowplot)

plot_activation_function <- function(f, title, range){
  ggplot(data.frame(x=range), mapping=aes(x=x)) + 
    geom_hline(yintercept=0, color='red', alpha=1/4) +
    geom_vline(xintercept=0, color='red', alpha=1/4) +
    stat_function(fun=f, colour = "dodgerblue3") +
    ggtitle(title) +
    scale_x_continuous(name='x') +
    scale_y_continuous(name='f(x)') +
    theme(plot.title = element_text(hjust = 0.5))
}



relu <- function(x) {ifelse(x < 0 , 0, x )}
p_relu <- plot_activation_function(relu, 'ReLU', c(-8,8))

lrelu <- function(x){ ifelse(x < 0 , 0.01 *x , x )}
p_lrelu <- plot_activation_function(lrelu, 'LReLU', c(-8,8))

softplus <- function(x){ log(1 + exp(x))}
p_softplus <- plot_activation_function(softplus, 'SoftPlus', c(-8,8))

prelu <- function(x, a=0.5) {ifelse(x < 0 , a * x , x )}
p_prelu <- plot_activation_function(prelu, 'PReLU (a=0.5)', c(-8,8))

elu <- function(x, a=0.7) {ifelse(x < 0 , a*(exp(x)-1) , x )}
p_elu <- plot_activation_function(elu, 'ELU (a=0.5)', c(-8,8))

selu <- function(x, a=1.6732632423543772848170429916717, l=1.0507009873554804934193349852946) {ifelse(x < 0 , l*(a*(exp(x)-1)) , x )}
p_selu <- plot_activation_function(selu, 'SELU (a=1.67, l=1.05)', c(-8,8))

sigmoid = function(x) {
  1 / (1 + exp(-x))
}

swish <- function(x) {x*sigmoid(x)}
p_swish <- plot_activation_function(swish, 'Swish/SiLU', c(-8,8))


gelu <- function(x) {x*pnorm(x)}
p_gelu <- plot_activation_function(gelu, 'GELU', c(-8,8))

p_grid <- cowplot::plot_grid(p_relu, p_lrelu, p_prelu, p_elu, p_selu, p_softplus, p_swish, p_gelu, ncol = 4) +
  theme(plot.margin = unit(c(3,3,3,3), "lines"))
ggsave("Plots/ReLU_Variant_Plots.pdf", plot = p_grid)
