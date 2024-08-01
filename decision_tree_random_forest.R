

library(ggplot2)
library(ISLR)
library(dplyr)
library(caTools)
library(rpart)

df <- College
head(College)

## EDA
### Some basic data exploration before we being building our models
### ** Scatterplot of graduation rate vs room and boarding costs

ggplot(df, aes(Room.Board, Grad.Rate)) + geom_point(aes(color=Private))

## It makes sense to see that Private Colleges have higher Room and Boarding Costs
## ** Histogram of full time undergrad students, colored by Private. **

ggplot(df, aes(F.Undergrad))+geom_histogram(aes(fill=Private),color="black",bins=50)

## ** histogram of graduation rate, colored by Private **

ggplot(df, aes(Grad.Rate)) + geom_histogram(aes(fill=Private),color="black", bin=50)

## There is a college with a graduation rate above 100%. Lets find that out and fix it

subset(df, Grad.Rate>100)

## Fix:

df['Cazenovia College','Grad.Rate'] <- 100

## Decision Tree

tree = rpart(Private ~ .,method="class",data=df)
tree.preds = predict(tree,df)
head(tree.preds)

## Creating a column "Private", with the variables Yes/No to indicate if the college is private or not, to match our original dataframe so that we can compare the results easily

tree.preds = as.data.frame(tree.preds)
tree.preds$Private = ifelse(tree.preds$Yes > 0.5,"Yes","No")

## Checking out our confusion Matrix

table(tree.preds$Private, df$Private)

library(rpart.plot)
prp(tree)

## RANDOM FOREST

library(randomForest)

rf.model = randomForest(Private ~ .,data=df, importance = TRUE)

p <- predict(rf.model, test)
table(p,test$Private)



