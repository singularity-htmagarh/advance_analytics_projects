### Data clustering and Segmentation Analysis

### Robust K-medoids in Action - Part 2 Model Development


# Required Libraries

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

# ip <- as.data.frame(installed.packages()[,c(1,3:4)])
# rownames(ip) <- NULL
# ip <- ip[is.na(ip$Priority),1:2,drop=FALSE]
# print(ip, row.names=FALSE)
     

# Where are the libraries
.libPaths()

# Use only the following packages
require(caret) # preProcess
require(ggplot2) # ggplot
require(cluster) # pam, silhouette
require(factoextra) # fviz_nbclust, fviz_pca_ind, fviz_pca_var, fviz_add
require(alluvial) # alluvial
require(stringr) # str_replace_all

# Utilities to plot clustered data (using ggplot)
source("clustering_101_util.r")



# Read the data from the CSV file
train.houses <- read.csv(file = "Data/AmesHousing-Train.csv", header = TRUE)
cat(paste("Train Data: ", nrow(train.houses), " observations"))

# Deal with ID attributes
cat(ifelse(length(test.housesPID)== length(unique(test.housesPID)), 
      "All IDs are unique", "Some IDs are not unique"))
rownames(test.houses) <- test.houses$PID
test.houses <- subset(test.houses, select=-c(Order, PID, X))


## Consideration of Categorical Variables

## Optional: More is not necessarily be better!
## Dummy variables wil generate zero IQR may will not be accepted by robust PCA!

# Add the selected factors as dummy vars and deal with spaces in column names
# Note: Regionname was excluded because of Lat and long vars
# Warning: this could add NAs

# dummies.model <- dummyVars(~ MS_Zoning + Street + Sale_Condition, data = train.houses, 
#   fullRank = TRUE, drop2nd = TRUE)
# train.houses <- cbind(train.houses, predict(dummies.model, newdata = train.houses))
# names(train.houses)<-str_replace_all(names(train.houses), c(" " = ".", "-" = "."))
# train.houses <- subset(train.houses, select=-c(MS_Zoning, Street, Sale_Condition))


# Select only numerical variables
train.houses <- train.houses[,sapply(train.houses, is.numeric)]
str(train.houses)

# Find the percentage of missing values in each column
cat("Percentage of missing values in selected variables\n")
head(sort(colMeans(is.na(train.houses)), decreasing = TRUE), 7)


# Drop columns with the large number of missing values
# train.houses <- subset(train.houses, select=-c())

# Impute missing values for the remaining columns
# impute.model <- preProcess(std.houses, method = c("knnImpute"))
impute.model <- preProcess(train.houses, method = c("medianImpute"))
train.houses <- predict(impute.model, train.houses)

# Check that imputation worked
cat("Percentage of missing values in selected variables after imputation\n")
head(sort(colMeans(is.na(train.houses)), decreasing = TRUE), 7)


# Start working with a copy
std.houses <- train.houses

# Standardise variables
# Warning: this may generate NAs
set_plot_dimensions(2, 0.9)
ggplot(stack(std.houses), aes(x = ind, y = values)) + 
  geom_boxplot() +
  ggtitle("Original distribution of values in selected variables") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))



# Scale and center all variables 
# This is required for k-means and k-median clustering
std.model <- preProcess(std.houses, method=c("center", "scale"))
std.houses <- predict(std.model, std.houses)

# Note that scaling may produce NAs or NaNs, in which case we'd need treatment
cat("Percentage of missing values in variables after scaling\n")
head(sort(colMeans(is.na(std.houses)), decreasing = TRUE), 7)

set_plot_dimensions(2, 0.9)
ggplot(stack(std.houses), aes(x = ind, y = values)) + 
  geom_boxplot() +
  ggtitle("Distribution of values in scaled variables") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
cat('Number of vars: ', ncol(std.houses))

## Robust PCA

# Remove all columns with IQR=0, i.e. insufficient variance in IQR
# Note: IQR(x) = quantile(x, 3/4) - quantile(x, 1/4) (not Tukey's IQR)
cat('Variables with IQR = 0 (to be removed)\n')
sapply(subset(sel.train.houses, select=sapply(sel.train.houses, IQR)==0), IQR)
sel.train.houses <- subset(sel.train.houses, select=sapply(sel.train.houses, IQR)>0)
cat('Variables with IQR > 0 (to be retained)\n')
sapply(sel.train.houses, IQR)
cat('Number of vars: ', ncol(sel.train.houses))


# Create a robust PCA (with SVD and cov.rob)
robpca.model <- princomp(sel.train.houses, cor=FALSE, covmat= MASS::cov.rob(sel.train.houses))
cat("Loadings of the selected PCs:\n\n")
prmatrix(round(robpca.model$loadings[,1:9], 3))
cat("\nNumber of PCs: ", ncol(robpca.model$loadings))


# Show the projection of original dimensions into principal components (loadings)
set_plot_dimensions(0.9, 0.7)
fviz_pca_var(robpca.model, col.var = "contrib", axes=c(1, 2),
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), repel = TRUE)

# Ensure you can "predict" scores
head(robpca.model$scores)
trans.train.houses <- data.frame(predict(robpca.model, newdata=sel.train.houses))
head(trans.train.houses)        


# Scree plot of eigenvalues
plot.eigenv <- function(rpca) {
    pc.var = rpca$sdev^2
    pc.nos = seq(1:length(pc.var))
    plot(x = pc.nos, y = pc.var, type = "b", axes=FALSE,
        main = "Scree Plot of Component Variance", col="blue",
        xlab = "Principal Component", ylab = "Variance")
    axis(side=1, at=pc.nos)
    axis(side=2, at=seq(0, 10, by=2))
    box()
}
set_plot_dimensions(1.5, 0.8, margin=c(5,5,5,5))
plot.eigenv(robpca.model)


# Cumulative variance plot (eigenvalues)
plot.cumvar <- function(rpca) {
    pc.var = append(0, rpca$sdev^2 / sum(rpca$sdev^2))
    pc.nos = seq(0, length(pc.var)-1)
    plot(x=pc.nos, y=cumsum(pc.var), type="b", col="red", axes=FALSE,
       xlab="Principal Component", ylab="Cumulative Variance", 
       main="Cumulative Variance Plot", panel.first=grid(), 
       xlim = c(0, length(pc.var)-1), ylim = c(0, 1))
    axis(side=1, at=pc.nos)
    axis(side=2, at=seq(0, 1, by=0.2))
    box()
}

set_plot_dimensions(1.5, 0.8, margin=c(5,5,5,5))
plot.cumvar(robpca.model)


# Try different methods: "gap_stat", "wss", "silhouette"
set.seed(2019)
small.train.size <- 100
small.train.idx <- sample(seq_len(nrow(trans.train.houses)), size = small.train.size)

# Investigate silhouette measurements (explained further)
set_plot_dimensions(1.5, 0.7)
sel.subset <- trans.train.houses[small.train.idx,]
fviz_nbclust(sel.subset, FUNcluster=cluster::pam, method = "silhouette",  k.max = 70) +
  ggtitle("Silhouettes") +
  theme(text = element_text(size=9), axis.text.x = element_text(angle = 90, hjust = 1))

  # Investigate silhouette measurements (explained further)
set_plot_dimensions(1.5, 0.7)
sel.subset <- trans.train.houses[small.train.idx,]
fviz_nbclust(sel.subset, FUNcluster=cluster::pam, method = "gap_stat",  k.max = 50) +
  ggtitle("Gap Statistics") +
  theme(text = element_text(size=9), axis.text.x = element_text(angle = 90, hjust = 1))



# Investigate the optimum using the "elbow method"
set_plot_dimensions(1.5, 0.7)
sel.subset <- trans.train.houses[small.train.idx,]
fviz_nbclust(sel.subset, FUNcluster=cluster::pam, method = "wss",  k.max = 70) +
  ggtitle("WSS Elbow Method") +
  theme(text = element_text(size=9), axis.text.x = element_text(angle = 90, hjust = 1))

## Creation of Medoid Clusters

# k-medoid settings
kNo <- 6 # Try: 2, 6 and 11 and 21
std <- FALSE

# Create a cluster model
cluster.model <- pam(x=trans.train.houses, k=kNo, stand = std)


# Cluster statistics for individual data points
sil.widths <- cluster.model
widths
head(sil.widths[order(as.numeric(row.names(sil.widths))),], 6)
print(paste("Sizes: silhouettes = ", length(cluster.model
widths),
        "; clustered set = ", length(cluster.model$clustering),
        "; original data set = ", nrow(sel.train.houses)))
print(paste("Average cluster silhouette width: ", round(cluster.model
avg.width, 2)))


# Enter cluster details into the standardised data in PCA coordinates
dataClusters <- cluster.model$clustering
clusteredDataset <- data.frame(cluster=dataClusters, trans.train.houses)
head(clusteredDataset)

## Cluster Diagnostics


# Plot PC1 x PC2
sel.cols=c("limegreen", "royalblue", "cyan3", "magenta3", "gold3", "red2")
set_plot_dimensions(1.5, 1)
plot.clus.rpca(robpca.model, cluster.model, data=trans.train.houses, cluster=dataClusters,
    title="Train Data Clustering", col=sel.cols, alpha=0.1, comp1=1, comp2=2, xlim=c(-6, 6), ylim=c(-5, 5))

    # Plot PC1 and PC3
sel.cols=c("limegreen", "royalblue", "cyan3", "magenta3", "gold3", "red2")
set_plot_dimensions(1.5, 1)
plot.clus.rpca(robpca.model, cluster.model, data=trans.train.houses, cluster=dataClusters,
    title="Train Data Clustering", col=sel.cols, alpha=0.1, comp1=1, comp2=3, xlim=c(-6, 6), ylim=c(-5, 5))


    # Distribution of silhouete widths within clusters
set_plot_dimensions(1.5, 0.7)
cols <- colorRampPalette(c("limegreen", "royalblue", "cyan3", "magenta3", "gold3", "red2"))(kNo)
ggplot(data.frame(cluster.model
widths), aes(group = cluster, x = cluster, y = sil_width)) + 
  geom_boxplot(width=0.5, fill=cols, alpha=0.3) +
  ggtitle("Distribution of Silhouette Widths per Cluster") +
  scale_x_discrete(limits=1:kNo) +
  theme(axis.text.x = element_text(angle = 60, hjust = 1, size=18))

  # Silhouete chart of data point widths within clusters
set_plot_dimensions(1.5, 0.7)
colsRamp <- colorRampPalette(c("limegreen", "royalblue", "cyan3", "magenta3", "gold3", "red2"))(kNo)
colsAlpha <- function(..., n, alpha) {
   colors <- colorRampPalette(...)(n)
   paste(colors, sprintf("%x", ceiling(255*alpha)), sep="")
}
cols <- colsAlpha(colsRamp, n=kNo, alpha=0.2)

fviz_silhouette(cluster.model, print.summary = FALSE, ylim=c(-0.2, 0.4)) +
    scale_color_manual(breaks = c("1", "2", "3", "4", "5"), values=cols) +
    scale_fill_manual(values=cols) +
    theme(legend.background = element_rect(color = "red", linetype = "solid")) +
    ggtitle("Silhouette Widths within Clusters")



# Adding cluster column to the original data
clus.train.houses <- merge(data.frame(cluster.model$clustering), train.houses, by=0, all=FALSE)
colnames(clus.train.houses)[2] <- "Cluster"
rownames(clus.train.houses) <- clus.train.houses
Cluster <- as.factor(clus.train.houses$Cluster)
clus.train.houses <- subset(clus.train.houses, select=-c(Row.names))
head(clus.train.houses)
cat("Number of clustered observations: ", nrow(clus.train.houses))


# Retrieving the medoid original observations
clus.train.medoid.ids <- row.names(cluster.model$medoids)
cat(clus.train.medoid.ids)
clus.train.medoids <- clus.train.houses[row.names(clus.train.houses) %in% clus.train.medoid.ids,]
clus.train.medoids <- subset(clus.train.medoids, 
     select=c(Cluster, Lot_Frontage, Lot_Area, Overall_Qual,
              Bedroom_AbvGr, Garage_Area, Year_Built, 
              Yr_Sold, SalePrice))
clus.train.medoids <- clus.train.medoids[with(clus.train.medoids, order(Cluster)),]
clus.train.medoids


## SEGMENTATION ANALYSIS / CLUSTER INTERPRETATION

# Plotting cluster information in the original units - using cluster medoids
colfunc <- colorRampPalette(c("navy", "red", "orange"))
cols <- colfunc(kNo)
set_plot_dimensions(2, 0.5, c(2, 7, 7, 2))
par(oma=c(0,2,2,0))
alluvial(clus.train.medoids, freq=clus.aggr.freq$Cluster, col=cols, cex=0.8, cex.axis = 0.7)
mtext("Segmentation Analysis of Houses / Original Units with Cluster Medoids", line = 6.5, side=3, cex=1.2)
mtext("Low - Medium - High", line = 6.5, side=2, cex=1.2)



saveRDS(impute.model, "Models/rimpute_model.rds")
saveRDS(std.model, "Models/rstd_model.rds")
saveRDS(robpca.model, "Models/rpca_model.rds")
saveRDS(cluster.model, "Models/rcluster_model.rds")


