library(tidyverse)

graph_groups <- function(csv, output, title) {
  data <- read.csv(csv, header=TRUE)
  ggplot(data, aes(x=fraction, y=score)) + geom_line(aes(color=type)) +
	ggtitle(title) + ylab("f1 Score") + theme(plot.title=element_text(hjust=0.5))
  ggsave(output)
}

graph_groups("./scores/kfold.csv", "./img/kfold.png", "Random Forest Scores vs Fraction of Data\nwith 5-Fold Cross Validation")
graph_groups("./scores/single.csv", "./img/single.png", "Random Forest Scores vs Fraction of Data")
graph_groups("./scores/svm.csv", "./img/svm.png", "SVM Scores vs Fraction of Data")
graph_groups("./scores/zeros.csv", "./img/zeros.png", "Random Forest Scores vs Fraction of\n0.5 Data Subset Replaced with Zeros")
