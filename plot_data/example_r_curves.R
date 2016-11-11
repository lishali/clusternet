library(dplyr)
library(ggplot2)
library(reshape)
setwd("~/Desktop/clusternet/plot_data/r_optimization")

temp <- list.files(pattern = "*.csv")


plot_r <- function(csv){
  name <- gsub(".csv", "", csv)
  a <- read.csv(csv, header=FALSE)
  a <- t(a)
  colnames(a) <- c("r_value", "loss")
  a <- as.data.frame(a)
  a <- slice(a, 2:nrow(a))
  g <- ggplot(a)+
    geom_point(aes(x=r_value, y=loss))+
    labs(title=name)
  setwd("~/Desktop/clusternet/plot_data/r_optimization/plots")
  ggsave(name, device="pdf")
  setwd("~/Desktop/clusternet/plot_data/r_optimization")
  return(g)
}

plot_r(temp[30])

lapply(temp, plot_r)

extract_r <- function(csv){
  name <- gsub(".csv", "", csv)
  a <- read.csv(csv, header=FALSE)
  a <- t(a)
  colnames(a) <- c("r_value", "loss")
  a <- as.data.frame(a)
  a <- slice(a, 2:nrow(a))
  return(a)
}

temp <- list.files(pattern = "*.csv")

temp_seed <- temp[grep("seed", temp)]
temp_seed <- temp[grep("min-2", temp)]

start = extract_r(temp_seed[1])
start$pairs <- rep(gsub(".csv", "", temp_seed[1]), each=nrow(start))
for (i in 2:length(temp_seed)){
  tmp <- extract_r(temp_seed[i])
  tmp$pairs <- rep(gsub(".csv", "", temp_seed[i]), each=nrow(tmp))
  start = full_join(start, tmp)
}

ggplot(start, aes(x=r_value, y=loss, group=pairs, color=pairs))+geom_point()+geom_line()

ggplot(filter(start, pairs=="p0.4q0.3r_min-2seed0"), aes(x=r_value, y=loss, group=pairs, color=pairs))+geom_point()+geom_line()
ggplot(filter(start, pairs=="p0.4q0.3r_min-2seed10"), aes(x=r_value, y=loss, group=pairs, color=pairs))+geom_point()+geom_line()
ggplot(filter(start, pairs=="p0.4q0.3r_min-2seed2"), aes(x=r_value, y=loss, group=pairs, color=pairs))+geom_point()+geom_line()
ggplot(filter(start, pairs=="p0.4q0.3r_min-2seed4"), aes(x=r_value, y=loss, group=pairs, color=pairs))+geom_point()+geom_line()
ggplot(filter(start, pairs=="p0.4q0.3r_min-2seed"), aes(x=r_value, y=loss, group=pairs, color=pairs))+geom_point()+geom_line()


for (i in 1:10){
  ggplot(filter(start, pairs==paste0("p0.4q0.3r_min-2seed", i)), aes(x=r_value, y=loss, group=pairs, color=pairs))+geom_point()+geom_line()
}


head(start)
