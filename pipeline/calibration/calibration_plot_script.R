library(ggplot2)

# models: apache, LogisticRegression, RandomForestClassifier, XGBClassifier
model_name <- "apache"
relative_file_path <- file.path(paste0(model_name, "_predictions.csv"))
absolute_file_path <- normalizePath(relative_file_path)
model_results <- read.csv(absolute_file_path)

calibration_plot <- ggplot(model_results, aes(preds, labels)) +
  geom_point(alpha = 0.1) +
  geom_smooth(se = F) +
  theme_bw() +
  coord_cartesian(xlim = c(0,1), ylim = c(0,1)) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  xlab("Observed") +
  ylab("Predicted")

ggsave(paste0(model_name, "_calibration_plot.png"), plot = calibration_plot, width = 10, height = 8, dpi = 300)