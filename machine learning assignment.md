Machine Learning Assignment
========================================================

The report use data from accelerometers on the belt, forearm, arm and dumbell from 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways, which is denoted as the "classe" variable in the data set.The goal of the project is to predict the manner in which they did the exercise. Here load the data as following:


```r
originaldata <- read.csv("C:\\Users\\lenovo\\Desktop\\Practical Machine Learning\\pml-training.csv", 
    header = T)
selcol1 <- grep("accel", colnames(originaldata))
tempdata <- originaldata[, c(selcol1, 160)]
selcol2 <- grep("var", colnames(tempdata))
smalldata <- tempdata[, -selcol2]
str(smalldata)
```

```
## 'data.frame':	19622 obs. of  17 variables:
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```


The function **createDataPartition** can be used to spit the data into the training and test sets:


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(lattice)
library(ggplot2)
inTrain <- createDataPartition(y = smalldata$classe, p = 0.7, list = FALSE)
training <- smalldata[inTrain, ]
testing <- smalldata[-inTrain, ]
```


First, linear discrimation analysis is used and fitting this model using repeated cross-validation having 5 repeats is shown below:


```r
library(MASS)
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
modFit1 <- train(classe ~ ., method = "lda", trControl = fitControl, data = training)
print(modFit1)
```

```
## Linear Discriminant Analysis 
## 
## 13737 samples
##    16 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 5 times) 
## 
## Summary of sample sizes: 12364, 12362, 12362, 12365, 12364, 12364, ... 
## 
## Resampling results
## 
##   Accuracy  Kappa   Accuracy SD  Kappa SD
##   0.5176    0.3819  0.01214      0.01556 
## 
## 
```

```r
pred <- predict(modFit1, testing)
table(pred, testing$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1160  342  476  196  170
##    B   67  464   92   58  204
##    C  126  167  336   76   97
##    D  295  116  112  570  126
##    E   26   50   10   64  485
```


Accuracy is one metric describing model in predicting and we got about 0.5 of the method.Principal component analysis and normalization did little improvement in model establishing here.

Second,predicting outcomes with trees was tried, parameters of cross validation is the same as LDA.


```r
library(rpart)
modFit2 <- train(classe ~ ., method = "rpart", trControl = fitControl, data = training)
print(modFit2)
```

```
## CART 
## 
## 13737 samples
##    16 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 5 times) 
## 
## Summary of sample sizes: 12364, 12362, 12363, 12365, 12362, 12363, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp       Accuracy  Kappa    Accuracy SD  Kappa SD
##   0.02421  0.4369    0.26676  0.02265      0.03351 
##   0.03869  0.4103    0.21862  0.04424      0.07978 
##   0.07151  0.3076    0.03568  0.02353      0.03625 
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.02421.
```


Predicting outcomes with trees is easy to interpret and has better performance in nonlinear settings theoretically. But the results shows smaller  accuracy in the project. A prettier version of this plot can be made with the rattle package.


```r
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## XXXX 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
fancyRpartPlot(modFit2$finalModel)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5.png) 






