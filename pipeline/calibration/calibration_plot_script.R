library(ggplot2)

theme_set(theme_bw(base_size = 14))

# models: Apache, LogisticRegression, RandomForestClassifier, XGBClassifier
model_name <- "LogisticRegression"
relative_file_path <- file.path(paste0(model_name, "_predictions.csv"))
absolute_file_path <- normalizePath(relative_file_path)
model_results <- read.csv(absolute_file_path)

calibration_plot <- ggplot(model_results, aes(predictions, labels)) +
  ggtitle(paste0(model_name, " Calibration Plot")) +
  xlab("Observed") +
  ylab("Predicted") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey") +
  geom_rug(data = model_results[model_results$labels == 0, ], aes(predictions, labels), sides = "b", linewidth = 0.1, alpha = 0.1, position = position_jitter(width = 0.01, height = 0)) +
  geom_rug(data = model_results[model_results$labels == 1, ], aes(predictions, labels), sides = "t", linewidth = 0.1, alpha = 0.1, position = position_jitter(width = 0.01, height = 0)) +
  geom_smooth(method = stats::loess, se = FALSE, colour = "black") +
  scale_x_continuous(breaks = seq(0, 1, 0.1), limits = c(0, 1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.1), limits = c(0, 1))

ggsave(paste0(model_name, "_calibration_plot.png"), plot = calibration_plot, width = 10, height = 10, dpi = 300)