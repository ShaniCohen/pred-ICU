library(ggplot2)

theme_set(theme_bw(base_size = 14))

# models: Apache, LogisticRegression, RandomForestClassifier, XGBClassifier
model_name <- "Apache"
relative_file_path <- file.path(paste0(model_name, "_predictions.csv"))
absolute_file_path <- normalizePath(relative_file_path)
model_results <- read.csv(absolute_file_path)

calibration_plot <- ggplot(model_results, aes(predictions, labels)) +
  ggtitle(paste0(model_name, " Calibration Plot")) +
  xlab("Observed") +
  ylab("Predicted") +
  scale_x_continuous(breaks = seq(0, 1, 0.1), limits = c(0, 1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.1), limits = c(0, 1)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey") +
  geom_rug(sides = "tb", linewidth = 0.1, alpha = 0.1, position = position_jitter(width = 0.01, height = 0)) +
  geom_smooth(method = stats::loess, se = FALSE, colour = "black")

ggsave(paste0(model_name, "_calibration_plot.png"), plot = calibration_plot, width = 10, height = 10, dpi = 300)