library(dplyr)
library(ggplot2)
library(reshape)
setwd("~/Desktop/clusternet/annealed_data")

temp <- list.files(pattern = "*.csv")

temp
plot_r <- function(csv){
  name <- gsub(".csv", "", csv)
  a <- read.csv(csv)
  
  a <- as.data.frame(a)
  a <- slice(a, 2:nrow(a))
  g <- ggplot(a)+
    geom_point(aes(x=X, y=loss))+
    labs(title=name)
  #setwd("~/Desktop/clusternet/annealed_data/plots")
  #ggsave(name, device="pdf")
  setwd("~/Desktop/clusternet/annealed_data")
  return(g)
}

plot_r(temp[9])

lapply(temp, plot_r)


plot_avgdeg <- function(csv){
  name <- gsub(".csv", "", csv)
  a <- read.csv(csv)
  a <- as.data.frame(a)
  a <- slice(a, 2:nrow(a))
  g <- ggplot(a)+
    geom_point(aes(x=X, y=avg_deg))+
    labs(title=name)
  #setwd("~/Desktop/clusternet/annealed_data/plots")
  #ggsave(name, device="pdf")
  setwd("~/Desktop/clusternet/annealed_data")
  return(g)
}

plot_avgdeg(temp[9])
plot_r(temp[9])

plot_avgdeg(temp[8])
plot_r(temp[8])

plot_avgdeg(temp[7])
plot_r(temp[7])

plot_avgdeg(temp[6])
plot_r(temp[6])

read_data <- function(csv){
  a <- read.csv(csv)
  a <- as.data.frame(a)
  a <- slice(a, 2:nrow(a))
  return(a)
}




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
  a$avg_deg_cat <- cut(a$avg_deg,seq(min(a$avg_deg), max(a$avg_deg), length.out = 5))
  a <- a %>% mutate(running_avg_loss = cummean(loss))
  #ggplot(a, aes(x = X, y = loss))+geom_point(aes(color=avg_deg_cat), alpha=0.7)
  g <- ggplot(a, aes(x = X, y = running_avg_loss))+
    geom_point()+#geom_point(aes(color=avg_deg_cat), alpha=0.7)+
    labs(title=name)
  setwd("~/Desktop/clusternet/annealed_data/plots")
  ggsave(name, device="pdf")
  setwd("~/Desktop/clusternet/annealed_data")
  return(g)
}

plot_moving_average(temp[23])
lapply(temp[5:23], plot_moving_average)

test <- data.frame(matrix(ncol=2, nrow=5000))
colnames(test) <- c("X", "Y")
test$Y <- c(rnorm(5000, mean=5.0, sd=2))
test <- test%>%mutate(cum_avg_Y = cummean(Y))
test$X <- c(1:5000)
ggplot(test, aes(x=X, y=Y))+geom_point()
ggplot(test, aes(x=X, y=cum_avg_Y))+geom_point()
