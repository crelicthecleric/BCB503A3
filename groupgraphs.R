library(tidyverse)

graph_groups <- function(csv, output, title) {
  data <- read.csv(csv, header=TRUE)
  ggplot(data, aes(x=fraction, y=score)) +
    geom_line(aes(color=type)) + ggtitle(title)
  ggsave(output)
}

graph_groups("../Desktop/kfold.csv", "../Desktop/kfold.png", "Random Forest Scores vs Fraction of Data with 5-Fold Cross Validation")
graph_groups("../Desktop/single.csv", "../Desktop/single.png", "Random Forest Scores vs Fraction of Data")
graph_groups("../Desktop/svm.csv", "../Desktop/svm.png", "SVM Scores vs Fraction of Data")
