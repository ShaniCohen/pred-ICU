library(mice)
library(ggplot2)
library(naniar)
library(dplyr)
library(farver)
library(gridExtra)
library(VIM)
df <- read.csv("data/training_v2.csv")
downsampled_df <- df %>%
  slice_sample(n = 1000) # for example, to select 1000 random rows
par(op)

op <- par(mfrow = c(2, 1)
        ,cex.axis=2,
        cex.main=2,
        cex.sub=2.5,
        cex.lab=2.5) 
        
p=marginplot(downsampled_df[,c("age","d1_glucose_min")])
ggsave("plot_with_margins.png", plot = p, dpi = 300, width = 10, height = 8)
