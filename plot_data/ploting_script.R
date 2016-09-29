library(dplyr)
library(ggplot2)


setwd("~/Desktop/clusternet/plot_data/")

test <- read.csv("r-6rate0.0333333333333p0.4q0.3iterations50000step1000.csv")
colnames(test)[1] <- "timestep"
class(test$loss)
class(test$timestep)
ggplot(test)+geom_line(aes(x=timestep, y=loss))


plot_points <- function(csv_file){
  
  a <-gsub(".csv", "_", csv_file)
    
  tmp <- read.csv(csv_file)
  colnames(tmp)[1] <- "timestep"
  
  g1 <- ggplot(test)+geom_line(aes(x=timestep, y=loss))
  ggsave(paste(a, "loss"), device = "png")
  
  g2 <- ggplot(test)+geom_line(aes(x=timestep, y=gradient_loss_r))
  ggsave(paste(a, "gradient_loss"), device = "png")
  
  g3 <- ggplot(test)+geom_line(aes(x=timestep, y=r_value))
  ggsave(paste(a, "r_value"), device = "png")
  
  return(g1)
}

temp <- list.files(pattern = "*.csv_file")
temp

testing <- lapply(temp, plot_points)

plot_points(temp[1])
