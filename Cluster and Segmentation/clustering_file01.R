# install.packages("caret", dependencies = TRUE, repos="http://cran.us.r-project.org")
# install.packages("ggplot2", dependencies = TRUE, repos="http://cran.us.r-project.org")
# install.packages("psych", dependencies = TRUE, repos="http://cran.us.r-project.org")
# install.packages("car", dependencies = TRUE, repos="http://cran.us.r-project.org")
# install.packages("cluster", dependencies = TRUE, repos="http://cran.us.r-project.org")
# install.packages("factoextra", dependencies = TRUE, repos="http://cran.us.r-project.org")
# install.packages("tidyr", dependencies = TRUE, repos="http://cran.us.r-project.org")
# install.packages("alluvial", dependencies = TRUE, repos="http://cran.us.r-project.org")
# install.packages("rospca", dependencies = TRUE, repos="http://cran.us.r-project.org")
# install.packages("stringr", dependencies = TRUE, repos="http://cran.us.r-project.org")


# Where are the libraries
.libPaths()

# Use only the following packages
require(caret) # preProcess
require(ggplot2) # ggplot
require(cluster) # pam, silhouette
require(factoextra) # fviz_nbclust, fviz_pca_ind, fviz_pca_var, fviz_add
require(alluvial) # alluvial
require(stringr) # str_replace_all

'/home/jacob/R/x86_64-pc-linux-gnu-library/3.5'
'/home/jacob/anaconda3/envs/rstat/lib/R/library'

# Source utilities
source("clustering_101_util.r")

## Pre-load all models developed in training

impute.model <- readRDS("Models/rimpute_model.rds")
std.model <- readRDS("Models/rstd_model.rds")
cluster.model <- readRDS("Models/rcluster_model.rds")
robpca.model <- readRDS("Models/rpca_model.rds")
cat("All models have been recovered")

## Acquire Test Data


# Read twst data from the CSV file
test.houses <- read.csv(file = "Data/AmesHousing-Test.csv", header = TRUE)
cat(paste("Test Data: ", nrow(test.houses), " observations"))


## And prepare it in exactly the same way as in testing.

# Deal with ID attributes
cat(ifelse(length(test.housesPID)== length(unique(test.housesPID)), 
      "All IDs are unique", "Some IDs are not unique"))
rownames(test.houses) <- test.houses$PID
test.houses <- subset(test.houses, select=-c(Order, PID, X))


## Selection and transformation of variables into numeric

## No encoding of categorical variables in this care


# Select only numeric variables
test.houses <- test.houses[,sapply(test.houses, is.numeric)]
colnames(test.houses)

## Elimination of missing values
## Either missing values in test data should not be allowed or they must be eliminated using the model applied in training

# Impute missing values and check is any still lift 
cat("Percentage of missing values before imputation: ", round(mean(is.na(test.houses)), 3)*100, "%\n")
test.houses <- predict(impute.model, test.houses)
cat("Percentage of missing values after imputation: ", round(mean(is.na(test.houses)), 3)*100, "%")


## Variable Standardization
# Scale and center all variables using pre-loaded model
cat("Average SD before standardisation: ", mean(apply(test.houses, 2, sd)), "%\n")
std.houses <- predict(std.model, test.houses)
cat("Average SD after standardisation: ", mean(apply(std.houses, 2, sd)), "%\n\n")
cat('Standardised house properties: ', colnames(std.houses))


## USE PCA MODEL to select and transform variables

# Find the variable names from PCA loadings
sel.test.houses <- subset(std.houses, select=rownames(robpca.model$loadings)) cat('Selected house properties: ', colnames(sel.test.houses))
head(sel.test.houses, 3)

# Apply PCA to the selected columns
trans.new.data <- data.frame(predict(robpca.model, newdata = sel.test.houses))
head(trans.new.data, 3)

## Cluster Prediction / Allocation in test data 
# Function which applies a pam cluster model to new data
apply.pam <- function(clus, newdata) apply(newdata, 1, 
                   function(x, c=clus) which.min(colSums((t(c$medoids)-x)^2)))
pred.clusters <- apply.pam(cluster.model, newdata=trans.new.data)
head(pred.clusters, 5)
cat('Number of clustered observations: ', length(pred.clusters), '\n')
kNo <- nrow(cluster.model$medoids)
cat('Number of clusters: ', kNo)

## Plot clustered test Data
## Once clustered test data in PCA coordinates it can be plotted using any scatter plot
## Plot clustered test data in PC1 x PC2 an PC1 x PC3 dimensions

# Plot PC1 x PC2
sel.cols=c("limegreen", "royalblue", "cyan3", "magenta3", "gold3", "red2")
set_plot_dimensions(1.5, 1)
plot.clus.rpca(robpca.model, cluster.model, data=trans.new.data, cluster=pred.clusters,
               title="Test Data Clustering", col=sel.cols, alpha=0.1, level=0.7,
               comp1=1, comp2=2, xlim=c(-6, 6), ylim=c(-5, 5))

# Plot PC1 and PC3
sel.cols=c("limegreen", "royalblue", "cyan3", "magenta3", "gold3", "red2")
set_plot_dimensions(1.5, 1)
plot.clus.rpca(robpca.model, cluster.model, data=trans.new.data, cluster=pred.clusters,
               title="Test Data Clustering", col=sel.cols, alpha=0.1, level=0.7,
               comp1=1, comp2=3, xlim=c(-6, 6), ylim=c(-5, 5))

## Outlier Detection I (Global Outlier)

### By using robust methods, we have managed to construct a clustering system and perform its diagnostics, in the presence of outliers.
### However, outliers may or may not be welcome in the new data, especialy when their cluster allocation may need to be questioned. So we 
### must be able to detect outliers in new data! This can be achieved using a clustering model itself and identifying observations, which are further furthest away from all mediods (GSS - global sum-of-squares of distance to medoids)

# Function which applies pam model to new data and calculates medoid GSS distance
apply.pam <- function(clus, newdata) apply(newdata, 1, 
                   function(x, c=clus) sum(colSums((t(c$medoids)-x)^2)))
pred.clust.dist <- scale(apply.pam(cluster.model, newdata=trans.new.data))
head(pred.clust.dist, 5)
cat('Number of clustered observations: ', length(pred.clust.dist), '\n')

# Distribution of sums of squared distances to medoids
set_plot_dimensions(0.3, 0.5)
ggplot(data.frame(ss.dist=pred.clust.dist), aes(y=ss.dist)) + 
  geom_boxplot(width=0.5, col="black", fill="red", alpha=0.3) +
  ggtitle("z-SS Dist Medoids")



# Plot outliers

# Outcless = 1 (normal), 2 (outlier)
outliers <- data.frame(outclass=as.integer(abs(pred.clust.dist) > 2.5)+1)

set_plot_dimensions(1.5, 1)
plot.outl.in.pca(robpca.model, data=trans.new.data, outl=outliers$outclass,
               col=c("royalblue", "red2"), alpha=0.2, level=0.95, size=1.2,
               title="Boundary for Housing Outliers (0.95)", comp1=1, comp2=2)


## Note the outliers in the "middle" of the data boundary - checking higher PC dimensions (e.g. PC1xPC3) will ensure that these data points are indeed outside the boundary.
## Another way of dealing with outliers would be to use the "local outlier detection". For example we could declare all cluster members far from the medoid or outside the cluster boundary as outliers. It should be noted that such an approach would generate outliers which are "visually" in the middle of the dataset boundaries.            