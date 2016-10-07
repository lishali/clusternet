library(dplyr)
library(ggplot2)

setwd("~/Desktop/clusternet/r_array_op/")

plot_points_save <- function(csv_file){
  setwd("~/Desktop/clusternet/r_array_op/")
  a <-gsub(".csv", "_", csv_file)
  
  tmp <- read.csv(csv_file)
  colnames(tmp)[1] <- "timestep"
  tmp$r_param <- gsub("[^[:alnum:][:blank:].]", "", tmp$r_param)
  tmp$r_param <- gsub("array", "", tmp$r_param)
  tmp$r_param <- as.numeric(tmp$r_param)
  
  
  loss_r_op <- tmp$loss[1]
  
  g1 <- ggplot(filter(tmp, timestep > 0), 
               aes(x=timestep, y=loss))+
    geom_point()+geom_line()+
    geom_hline(yintercept = loss_r_op, color = "blue")+
    geom_text(aes(0,loss_r_op,label = signif(loss_r_op, 4), vjust = -1))+
    labs(title = a)
  setwd("~/Desktop/clusternet/r_array_op/plots")
  ggsave(paste(a, "loss"), device = "pdf")
  
  
  r_op <- tmp$gradient_loss_v[1]
  g3 <- ggplot(filter(tmp, timestep > 0), aes(x=timestep, y=r_param))+
    geom_point()+geom_line()+
    #geom_hline(yintercept = r_op, color = "blue")+
    #geom_text(aes(0,r_op,label = signif(r_op, 4), vjust = -1))+
    labs(title = a)
  setwd("~/Desktop/clusternet/r_array_op/r_plots")
  ggsave(paste(a, "r_value"), device = "pdf")
  
  return()
}
setwd("~/Desktop/clusternet/r_array_op/")
temp <- list.files(pattern = "*.csv")
temp
lapply(temp, plot_points_save)



try <- read.csv(temp[6])
try
#######################
plot_loss <- function(csv_file){
  
  a <-gsub(".csv", "_", csv_file)
  
  tmp <- read.csv(csv_file)
  colnames(tmp)[1] <- "timestep"
  tmp$r_param <- gsub("[^[:alnum:][:blank:].]", "", tmp$r_param)
  tmp$r_param <- as.numeric(tmp$r_param)
  loss_r_op <- tmp$loss[1]
  
  g1 <- ggplot(filter(tmp, timestep > 0), 
               aes(x=timestep, y=loss))+
    geom_point()+geom_line()+
    geom_hline(yintercept = loss_r_op, color = "blue")+
    geom_text(aes(0,loss_r_op,label = signif(loss_r_op, 4), vjust = -1))+
    labs(title = a)
  
  return(g1)
}

plot_r <- function(csv_file){
  
  a <-gsub(".csv", "_", csv_file)
  
  tmp <- read.csv(csv_file)
  colnames(tmp)[1] <- "timestep"
  tmp$r_param <- gsub("[^[:alnum:][:blank:].]", "", tmp$r_param)
  tmp$r_param <- gsub("array", "", tmp$r_param)
  tmp$r_param <- as.numeric(tmp$r_param)
  
  r_op <- tmp$gradient_loss_v[1]
  
  g3 <- ggplot(filter(tmp, timestep > 0), aes(x=timestep, y=r_param))+
    geom_point()+geom_line()+
    geom_hline(yintercept = r_op, color = "blue")+
    geom_text(aes(0,r_op,label = signif(r_op, 4), vjust = -1))+
    labs(title = a)
  #  ggsave(paste(a, "r_value"), device = "jpg")
  
  return(g3)
}
#tmp <- read.csv(temp[11])
#colnames(tmp)[1] <- "timestep"

#g1 <- ggplot(filter(test, timestep > 5))+geom_line(aes(x=timestep, y=loss))



temp_new <- temp[grep("*p0.25q0.15*", temp)]
temp_new <- temp_new[grep("*rate3.33333333333e-05*", temp_new)]

a <- read.csv(temp_new[1])
a

plot_loss(temp[7])

plot_r(temp[1])

