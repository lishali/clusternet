library(dplyr)
library(ggplot2)
library(reshape)
setwd("~/Desktop/clusternet/GNN/imbalanced_block_data")

temp <- list.files(pattern = "*.csv")

temp 
temp <- temp[1:123]
temp
plot_r <- function(csv){
  name <- gsub(".csv", "", csv)
  a <- read.csv(csv)
  a <- as.data.frame(a)
  g <- ggplot(a)+
    geom_point(aes(x=X, y=loss))+
    labs(title=name)
  #setwd("~/Desktop/clusternet/GNN/plots")
  #ggsave(name, device="pdf")
  setwd("~/Desktop/clusternet/GNN/imbalanced_block_data")
  return(g)
}

plot <- function(csv){
  name <- gsub(".csv", "", csv)
  a <- read.csv(csv)
  a <- as.data.frame(a)
  a <- melt(a, id="X")
  g <- ggplot(filter(a))+
    geom_point(aes(x=X, y=value, group=variable, color=variable, alpha=0.2))+
    labs(title=name)
  setwd("~/Desktop/clusternet/GNN/data/dataConstant/plots")
  ggsave(name, device="pdf")
  setwd("~/Desktop/clusternet/GNN/data/dataConstant")
  return(g)
}

plot_list <- function(csv){
  name <- gsub(".csv", "", csv)
  a <- read.csv(csv)
  a <- as.data.frame(a)
  colnames(a) <- c("X", "lossA", "lossB", "loss")
  a <- melt(a, id="X")
  g <- ggplot(filter(a, X>100))+
    geom_line(aes(x=X, y=value, group=variable, color=variable, alpha=0.2))+
    labs(title=name)
  setwd("~/Desktop/clusternet/GNN/data/dataConstant/plots")
  #ggsave(name, device="pdf")
  setwd("~/Desktop/clusternet/GNN/data/dataConstant")
  return(g)
}

plot(temp[1])

lapply(temp, plot)

read_data <- function(csv){
  a <- read.csv(csv)
  a <- as.data.frame(a)
  return(a)
}

a <- read_data(temp[1])
a <- a%>%melt(a)



num <- 16
summary(read_data(temp[num])$avg_deg)
a <- read_data(temp[num])
name <- gsub(".csv", "", csv)
a$avg_deg_cat <- cut(a$avg_deg,seq(min(a$avg_deg), max(a$avg_deg), length.out = 5))
a <- a %>% mutate(running_avg_loss = cummean(loss))
ggplot(a, aes(x = X, y = loss))+geom_point(aes(color=avg_deg_cat), alpha=0.7)
ggplot(a, aes(x = X, y = running_avg_loss))+
  geom_point(aes(color=avg_deg_cat), alpha=0.7)+
  labs(title=name)
plot_r(temp[num])

plot_moving_average <- function(csv_name){
  a <- read_data(csv_name)
  name <- gsub(".csv", "", csv_name)
  a <- a %>% mutate(running_avg_loss = cummean(loss))
  #ggplot(a, aes(x = X, y = loss))+geom_point(aes(color=avg_deg_cat), alpha=0.7)
  g <- ggplot(a, aes(x = X, y = running_avg_loss))+
    geom_point()+#geom_point(aes(color=avg_deg_cat), alpha=0.7)+
    labs(title=name)
  
  setwd("~/Desktop/clusternet/GNN/data/dataConstant/plots")
  ggsave(name, device="pdf")
  setwd("~/Desktop/clusternet/GNN/data/dataConstant")
  return(g)
}

plot_moving_average(temp[95])
lapply(temp[1:10], plot_moving_average)

test <- data.frame(matrix(ncol=2, nrow=5000))
colnames(test) <- c("X", "Y")
test$Y <- c(rnorm(5000, mean=5.0, sd=2))
test <- test%>%mutate(cum_avg_Y = cummean(Y))
test$X <- c(1:5000)
ggplot(test, aes(x=X, y=Y))+geom_point()
ggplot(test, aes(x=X, y=cum_avg_Y))+geom_point()


summ <- function(csvname){
  a <- read.csv(csvname)
  return(summary(a))
}



################
setwd("~/Desktop/clusternet/GNN/imbalanced_block_data")

temp <- list.files(pattern = "*.csv")

temp 
temp <- temp[1:98]
temp

test <- temp[grep("p_max0.6", temp)]

test <- test[grep("Size10000l", test)]
#test <- test[grep("size20", test)]
#test <- test[grep("l_rate1e-05", test)]
test <- test[grep("batch_size1p", test)]
#test <- test[grep("DATApoints200", test)]
test

result <- read.csv(test[1])
name <- gsub("Size10000l_rate0.001batch_size1", "", test[1])
name <- gsub(".csv", "", name)
result$less_than_27 <- as.factor(result$loss<27)
result$group <- name

for (i in 2:length(test)){
  tmp <- read.csv(test[i])
  name <- gsub("Size10000l_rate0.001batch_size1", "", test[i])
   #name <- gsub("Size10000l_rate1e-05group_size0.0batch_size20p_min1", "", test[i])
  tmp$less_than_27 <- as.factor(tmp$loss<27)
  name <- gsub(".csv", "", name)
  tmp$group <- name
  result = full_join(result, tmp)
}

ggplot(filter(result, X>0), aes(x=X, y=log(loss), colour=less_than_27))+
  geom_jitter(alpha=0.01, aes(colour=less_than_27))+
  facet_wrap(~group)


