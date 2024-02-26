library(ggplot2)
library(boot)

theme_set(theme_bw(base_size = 14))

# models: Apache, LogisticRegression, RandomForestClassifier, XGBClassifier, latest_XGBClassifier
model_name <- "latest_XGBClassifier"
plot_color <- switch(
  model_name,
  LogisticRegression = "#1f77b4",
  XGBClassifier = "#ff7f0e",
  latest_XGBClassifier = "#ff7f0e",
  RandomForestClassifier = "#2ca02c",
  Apache = "#000000"
)
relative_file_path <- file.path(paste0(model_name, "_predictions.csv"))
absolute_file_path <- normalizePath(relative_file_path)
model_results <- read.csv(absolute_file_path)

set.seed(123)

brier_score <- function(predictions, observed) {
  mean((predictions - observed)^2)
}
bootstrap_brier <- function(data, indices) {
  d <- data[indices, ]
  return(brier_score(d$predictions, d$labels))
}
bootstrap_results <- boot(data = model_results, statistic = bootstrap_brier, R = 1000)
ci <- boot.ci(bootstrap_results, type = "perc")$percent[4:5]
brier_score_value <- brier_score(model_results$predictions, model_results$labels)
legend_text <- paste0(model_name, " (Brier Score = ", round(brier_score_value, 3), ", CI: ", round(ci[1], 3), "-", round(ci[2], 3), ")")

calibration_plot <- ggplot(model_results, aes(predictions, labels)) +
  ggtitle(paste0("Calibration Plot")) +
  xlab("Observed") +
  ylab("Predicted") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey") +
  geom_rug(data = model_results[model_results$labels == 0, ], aes(predictions, labels), sides = "b", linewidth = 0.1, alpha = 0.1, position = position_jitter(width = 0.01, height = 0), color = plot_color) +
  geom_rug(data = model_results[model_results$labels == 1, ], aes(predictions, labels), sides = "t", linewidth = 0.1, alpha = 0.1, position = position_jitter(width = 0.01, height = 0), color = plot_color) +
  geom_smooth(method = stats::loess, se = FALSE, aes(colour = legend_text)) +
  scale_x_continuous(breaks = seq(0, 1, 0.1), limits = c(0, 1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.1), limits = c(0, 1)) +
  scale_colour_manual(values=c(plot_color)) +
  theme(legend.position = "bottom", legend.text = element_text(size = 14), legend.title = element_blank(), legend.background = element_rect(colour = "black", size = 0.3, linetype = "solid"), plot.title = element_text(hjust = 0.5))

ggsave(paste0(model_name, "_calibration_plot.png"), plot = calibration_plot, width = 10, height = 10, dpi = 300)