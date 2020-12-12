# try: optigrid, \url{https://github.com/mexxexx/Optigrid}
# denclue, \url{https://github.com/mgarrett57/DENCLUE}
# chameleon \url{https://github.com/Moonpuck/chameleon_cluster}
# and some pyClustering: \url{https://pypi.org/project/pyclustering/}



combo <- data.frame(read.csv(file = '/Users/byrdsmyth/Documents/School/Classes/CPTS575/Project/codebase/combo.csv'))
combo
# Installing Packages 
install.packages("fpc") 
combo$in_key_states 

# Loading package 
library(fpc) 

colnames(combo)

# Remove label form dataset 
drop<-c('agent','keyNum', 'key_state', 'pacman_direction', 'context_state','orange_ghost_direction','red_ghost_direction', 'blue_ghost_direction', 'pink_ghost_direction')
# , 'in_key_states', 
combo_stripped<-combo[,!names(combo) %in% drop]
summary(combo_stripped)
combo_stripped$end_of_epoch<-as.numeric(as.logical(combo_stripped$end_of_epoch))
combo_stripped$end_of_episode<-as.numeric(as.logical(combo_stripped$end_of_episode))
combo_stripped$beforeLifeLoss<-as.numeric(as.logical(combo_stripped$beforeLifeLoss))
sapply(combo_stripped, class)
for (column in colnames(combo_stripped)) {
    print(column)
    print(which(is.na(column), arr.ind=TRUE))
}
combo_stripped[is.na(combo_stripped)] <- 0

comboS_norm <- as.data.frame(apply(combo_stripped, 2, function(x) (x - min(x))/(max(x)-min(x))))
summary(comboS_norm)

library(tidyverse)
#colnames(combo_stripped %>% select_if(negate(is.numeric)))

# from here: https://www.r-statistics.com/2013/08/k-means-clustering-from-r-in-action/
wssplot <- function(data, nc=15, seed=1234){
    wss <- (nrow(data)-1)*sum(apply(data,2,var))
    for (i in 2:nc){
        set.seed(seed)
        wss[i] <- sum(kmeans(data, centers=i)$withinss)}
    plot(1:nc, wss, type="b", xlab="Number of Clusters",
         ylab="Within groups sum of squares")}
wssplot(combo_stripped)                                                #2
library(NbClust)
set.seed(1234)
nc <- NbClust(combo_stripped, min.nc=2, max.nc=15, method="kmeans")
table(nc$Best.n[1,])



# Fitting DBScan clustering Model  
# to training dataset 
set.seed(1)  # Setting seed 
kmeans.result <- kmeans(combo_stripped, 10)
combo$kMeans_cluster <- as.character(kmeans.result$cluster)
combo
table(combo$keyNum, kmeans.result$cluster)
kmeans.result$centers

# Plot the data point
library(ggplot2)
x_string = combo$to_blue_ghost
y_string = combo$to_db1
ggplot(data = combo, 
       mapping = aes(x = x_string, 
                     y = y_string,
                     colour = kMeans_cluster)) +
    geom_point(alpha = 0.35)
# Add centroid
ggplot() +
    geom_point(data = combo, 
               mapping =  aes(x = episode_step, 
                              y = 'action.3.episode.sum', 
                              colour = 'kMeans_cluster')) +
    geom_point(mapping = aes_string(x = kmeans.result$centers[, "episode_step"], 
                                    y = kmeans.result$centers[, "total_reward"]),
               color = "red", size = 4)
# Add text as label
ggplot() +
    geom_point(data = combo, 
               mapping =  aes(x = keyNum, 
                              y = state, 
                              colour = kMeans_cluster)) +
    geom_point(mapping = aes_string(x = kmeans.result$centers[, "keyNum"], 
                                    y = kmeans.result$centers[, "state"]),
               color = "red", size = 4) +
    geom_text(mapping = aes_string(x = kmeans.result$centers[, "keyNum"], 
                                   y = kmeans.result$centers[, "state"],
                                   label = 1:10),
              color = "black", size = 4) +
    theme_light()


plot(combo[, c("state", "keyNum")],
     col = kmeans.result$cluster)

# initializes the k-means algorithm several times with random
# points from the data set as means
kmeansruns.result <- kmeansruns(combo_stripped)
kmeansruns.result


library(cluster)
# group into 3 clusters
pam.result <- pam(combo_stripped, 8)
# check against actual class label
table(pam.result$clustering, combo$keyNum)

plot(pam.result)


# don't have to choose k
pamk.result <- pamk(combo_stripped)
# number of clusters
pamk.result$nc

# check clustering against actual class label
table(pamk.result$pamobject$clustering, combo$in_key_states)

# check clustering against actual class label
table(pamk.result$pamobject$clustering, combo$keyNum)

## clustering with DIANA - Divisive clustering works in an opposite way, which puts all
# objects in a single cluster and then divides the cluster into
# smaller and smaller ones.
# I DIANA [Kaufman and Rousseeuw, 1990]
# I BIRCH [Zhang et al., 1996]
# I CURE [Guha et al., 1998]
# I ROCK [Guha et al., 1999]
# I Chameleon [Karypis et al., 1999]
library(cluster)
diana.result <- diana(combo_stripped)
plot(diana.result, which.plots = 2, labels = combo$keyNum[idx])



# Density cluster
Dbscan_cl <- dbscan(combo_stripped, eps = 0.9, MinPts = 25) 
Dbscan_cl 

# Checking cluster 
Dbscan_cl$cluster 

# Table 
table(Dbscan_cl$cluster, iris$Species) 

# Plotting Cluster 
plot(Dbscan_cl, iris_1, main = "DBScan") 
plot(Dbscan_cl, iris_1, main = "Petal Width vs Sepal Length")



# try just clustering the key states
combo_key1 <- combo %>% filter(keyNum > 0)
combo_key2 <- combo_key1[,!names(combo) %in% drop]
combo_key2$end_of_epoch<-as.numeric(as.logical(combo_key2$end_of_epoch))
combo_key2$end_of_episode<-as.numeric(as.logical(combo_key2$end_of_episode))
combo_key2$beforeLifeLoss<-as.numeric(as.logical(combo_key2$beforeLifeLoss))
combo_key2[is.na(combo_key2)] <- 0

comboKey_S_norm <- as.data.frame(apply(combo_key2, 2, function(x) (x - min(x))/(max(x)-min(x))))
summary(comboKey_S_norm)

library(cluster)
diana.result <- diana(combo_key2)
plot(diana.result, which.plots = 2, labels = combo_key1$keyNum)


# group into 3 clusters
pam.result <- pam(combo_key2, 7)
# check against actual class label
table(pam.result$clustering, combo_key1$keyNum)

plot(pam.result)

wssplot(combo_key2)    

# KEEP THS - TELLS WHICH AGENT STRATEGIES ARE DISTINGUISHABLE
kmeans.result <- kmeans(combo_key2, 5)
combo_key1$kMeans_cluster <- as.numeric(kmeans.result$cluster)
combo_key1.columns
table(combo_key1$agentNum, kmeans.result$cluster)
kmeans.result$centers

ggplot() +
    geom_point(data = combo_key1, 
               mapping =  aes(x = mean_reward, 
                              y = X, 
                              colour = as.factor(kMeans_cluster))) +
    scale_colour_brewer(palette = "Paired") +
    theme_minimal() +
    theme(legend.position = "bottom")

ggplot() +
    geom_point(data = combo_key1, 
               mapping =  aes(x = to_ghosts_mean, 
                              y = keyNum, 
                              colour = as.factor(kMeans_cluster))) +
    scale_colour_brewer(palette = "Paired") +
    theme_minimal() +
    theme(legend.position = "bottom")







# initializes the k-means algorithm several times with random
# points from the data set as means
kmeansruns.result <- kmeansruns(combo_key2)
kmeansruns.result
combo_key1$kRun_Cluster <- as.numeric(kmeansruns.result$cluster)
combo_key1$kRun_Cluster


# Load RColorBrewer
library(RColorBrewer)
library(viridis)

# Define the number of colors you want
nb.cols <- 10
mycolors <- colorRampPalette(brewer.pal(8, "Set2"))(nb.cols)

ggplot() +
    geom_point(data = combo_key1, 
               mapping =  aes(x = mean_reward, 
                              y = X, 
                              colour = as.factor(kMeans_cluster))) +
    scale_colour_brewer(palette = "Paired") +
    theme_minimal() +
    theme(legend.position = "bottom")

ggplot() +
    geom_point(data = combo_key1, 
               mapping =  aes(x = diff_to_pill2, 
                              y = to_ghosts_mean, 
                              colour = as.factor(kMeans_cluster))) +
    scale_colour_brewer(palette = "Paired") +
    theme_minimal() +
    theme(legend.position = "bottom")





