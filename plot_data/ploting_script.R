library(dplyr)
library(ggplot2)

setwd("~/Desktop/clusternet/plot_data/")

plot_points_save <- function(csv_file){
  setwd("~/Desktop/clusternet/plot_data/")
  a <-gsub(".csv", "_", csv_file)
  
  tmp <- read.csv(csv_file)
  colnames(tmp)[1] <- "timestep"
  tmp$gradient_loss_r <- gsub("[^[:alnum:][:blank:].]", "", tmp$gradient_loss_r)
  tmp$gradient_loss_r <- gsub("array", "", tmp$gradient_loss_r)
  tmp$gradient_loss_r <- as.numeric(tmp$gradient_loss_r)
  tmp$r_value <- gsub("[^[:alnum:][:blank:].]", "", tmp$r_value)
  tmp$r_value <- as.numeric(tmp$r_value)
  
  loss_r_op <- tmp$gradient_loss_r[1]
  g1 <- ggplot(filter(tmp, timestep > 0), 
               aes(x=timestep, y=loss))+
    geom_point()+geom_line()+
    geom_hline(yintercept = loss_r_op, color = "blue")+
    geom_text(aes(0,loss_r_op,label = signif(loss_r_op, 4), vjust = -1))+
    labs(title = a)
  setwd("~/Desktop/clusternet/plot_data/loss_plots")
  ggsave(paste(a, "loss"), device = "pdf")
  
  
  g2 <- ggplot(filter(tmp, timestep > 0), aes(x=timestep, y=gradient_loss_r))+
    geom_point()+geom_line()+
    labs(title = a)
  setwd("~/Desktop/clusternet/plot_data/gradient_plots")
  ggsave(paste(a, "gradient_loss"), device = "pdf")
  
  r_op <- tmp$r_value[1]
  g3 <- ggplot(filter(tmp, timestep > 0), aes(x=timestep, y=r_value))+
    geom_point()+geom_line()+
    geom_hline(yintercept = r_op, color = "blue")+
    geom_text(aes(0,r_op,label = signif(r_op, 4), vjust = 1))+
    labs(title = a)
  setwd("~/Desktop/clusternet/plot_data/r_value_plots")
  ggsave(paste(a, "r_value"), device = "pdf")
  
  return()
}
setwd("~/Desktop/clusternet/plot_data/")
temp <- list.files(pattern = "*.csv")

temp <- temp[grep("*0.3q0.05*", temp)]

lapply(temp, plot_points_save)



try <- read.csv(temp[6])
try
#######################
plot_loss <- function(csv_file){
  
  a <-gsub(".csv", "_", csv_file)
  
  tmp <- read.csv(csv_file)
  colnames(tmp)[1] <- "timestep"
  tmp$gradient_loss_r <- gsub("[^[:alnum:][:blank:].]", "", tmp$gradient_loss_r)
  tmp$gradient_loss_r <- gsub("array", "", tmp$gradient_loss_r)
  tmp$gradient_loss_r <- as.numeric(tmp$gradient_loss_r)
  loss_r_op <- tmp$gradient_loss_r[1]

  g1 <- ggplot(filter(tmp, timestep > 0), 
               aes(x=timestep, y=loss))+
    geom_point()+geom_line()+
    geom_hline(yintercept = loss_r_op, color = "blue")+
    geom_text(aes(0,loss_r_op,label = signif(loss_r_op, 4), vjust = -1))+
    labs(title = a)
 
  return(g1)
}

plot_gradient <- function(csv_file){
  
  a <-gsub(".csv", "_", csv_file)
  
  tmp <- read.csv(csv_file)
  colnames(tmp)[1] <- "timestep"
  tmp$gradient_loss_r <- gsub("[^[:alnum:][:blank:].]", "", tmp$gradient_loss_r)
  tmp$gradient_loss_r <- gsub("array", "", tmp$gradient_loss_r)
  tmp$gradient_loss_r <- as.numeric(tmp$gradient_loss_r)
  
  g2 <- ggplot(filter(tmp, timestep > 0), 
               aes(x=timestep, y=gradient_loss_r))+
    geom_point()+geom_line()+
    labs(title = a)
  #  ggsave(paste(a, "gradient_loss"), device = "jpg")
  
  return(g2)
}

plot_r <- function(csv_file){
  
  a <-gsub(".csv", "_", csv_file)
  
  tmp <- read.csv(csv_file)
  colnames(tmp)[1] <- "timestep"
  tmp$gradient_loss_r <- gsub("[^[:alnum:][:blank:].]", "", tmp$gradient_loss_r)
  tmp$gradient_loss_r <- gsub("array", "", tmp$gradient_loss_r)
  tmp$gradient_loss_r <- as.numeric(tmp$gradient_loss_r)
  tmp$r_value <- gsub("[^[:alnum:][:blank:].]", "", tmp$r_value)
  tmp$r_value <- as.numeric(tmp$r_value)

  r_op <- tmp$r_value[1]
  g3 <- ggplot(filter(tmp, timestep > 0), aes(x=timestep, y=r_value))+
    geom_point()+geom_line()+
    geom_hline(yintercept = r_op, color = "blue")+
    geom_text(aes(0,r_op,label = signif(r_op, 4), vjust = 1))+
    labs(title = a)
  #  ggsave(paste(a, "r_value"), device = "jpg")
  
  return(g3)
}
#tmp <- read.csv(temp[11])
#colnames(tmp)[1] <- "timestep"

#g1 <- ggplot(filter(test, timestep > 5))+geom_line(aes(x=timestep, y=loss))



temp_new <- temp[grep("0.0003", temp)]

a <- read.csv(temp_new[2])
a

plot_loss(temp_new[2])
plot_gradient(temp_new[2])
plot_r(temp_new[2])

