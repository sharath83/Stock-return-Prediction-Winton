# Winton stock market challenge - Predict every minute return value of 120000 stocks.
rm(list=ls())

setwd("/Users/homw/Documents/petp/winton/")
library(DMwR)
suppressMessages(library(dplyr))
library(xgboost)

train <- read.csv("train.csv", header = T)
test <- read.csv("test_2.csv", header = T)
sub <- read.csv("sample_submission_2.csv", header = T)
str(train)
train$Ret_121 
# Create train data
# features are 1 - 25, -2, -1 returns and average intra day return of 10 mins (39 features)
# For each stock we have to predict every minute returns for 60 minutes and +1, +2 returns (62 return values)
tr_features <- train[,2:28]
tr_features <- centralImputation(tr_features)

train[is.na(train)] <- 0 #Understand why missing values in returns. might not be imputed with a 0


m <- as.vector(0)
s <- as.vector(0)
ret <- matrix(0,nrow = nrow(train),ncol = 24)
for (n in 1:nrow(train)){
  stock1 <- train[n,29:(119+28)]
  for (i in 0:11){
    start = i*10+1
    end = i*10+9
    m[i+1] <- median(as.numeric(stock1[start:end]), na.rm = T)
    s[i+1] <- sd(as.numeric(stock1[start:end]), na.rm = T)
  }
  ret[n,] <- cbind(t(m), t(s))
}

tr_features <- cbind(tr_features, ret)
#Now create 62 records from each record
response <- train[,148:209]
tr.data1 <- as.data.frame(matrix(nrow = 0,ncol = 53))
tr.data2 <- as.data.frame(matrix(nrow = 0,ncol = 53))
tm <- proc.time()
for (n in 20001:30000){
  vec <- tr_features[n,]
  t <- matrix(rep(vec,62),nrow = 62, byrow = T)
  V53 <- as.vector(t(response[n,])) #target variable
  t <- as.data.frame(cbind(t, 1:62, V53))
  tr.data1 <- rbind_all(list(tr.data1,t))
  if(n%%500 == 0){
    tr.data2 <- rbind_all(list(tr.data2,tr.data1))
    tr.data1 <- as.data.frame(matrix(nrow = 0,ncol = 53))
    
    print(n/500)
    
  }
}
proc.time()-tm
tr1_10 <- tr.data2
tr10_20 <- tr.data2
tr20_30 <- tr.data2
tr30_40 <- tr.data2

tr30_40 <- data.frame(lapply(tr30_40, as.numeric))

tr20 <- rbind_all(list(tr1_10,tr10_20))
tr.data2 <- rbind_all(list(tr20,tr20_30,tr30_40))
names(tr.data2)
tr.data2 <- as.data.frame(tr.data2)
rm(tr1_10,tr10_20,tr20_30,tr30_40,tr20)
write.csv(tr.data2, "trdata.csv")


#Prepare test data
ts_features <- test[,2:28]
ts_features <- centralImputation(ts_features)

m <- as.vector(0)
s <- as.vector(0)
ret <- matrix(0,nrow = nrow(test),ncol = 24)
for (n in 1:nrow(test)){
  stock1 <- test[n,29:(119+28)]
  for (i in 0:11){
    start = i*10+1
    end = i*10+9
    m[i+1] <- median(as.numeric(stock1[start:end]), na.rm = T)
    s[i+1] <- sd(as.numeric(stock1[start:end]), na.rm = T)
  }
  ret[n,] <- cbind(t(m), t(s))
  
  if(n%%5000 == 0){
    print(n)
  }
}
ts_features <- cbind(ts_features, ret)

ts.data2[is.na(ts.data2)] <- 0

# create 62 records from each record for prediction for test data
#Now create 62 records from each record
ts.data1 <- as.data.frame(matrix(nrow = 0,ncol = 52))
ts.data2 <- as.data.frame(matrix(nrow = 0,ncol = 52))
tm <- proc.time()
for (a in 1:12){
  for (n in (1+(a-1)*10000):(a*10000)){
    vec <- ts_features[n,]
    t <- matrix(rep(vec,62),nrow = 62, byrow = T)
    #V53 <- as.vector(t(response[n,])) #target variable
    t <- as.data.frame(cbind(t, 1:62))
    ts.data1 <- rbind_all(list(ts.data1,t))
    if(n%%500 == 0){
      ts.data2 <- rbind_all(list(ts.data2,ts.data1))
      ts.data1 <- as.data.frame(matrix(nrow = 0,ncol = 52))
      print(n/1000)
    }
  }
  name <- paste0("tsd_",a)
  ts.data2 <- data.frame(lapply(ts.data2, as.numeric))
  assign(name, ts.data2)
  ts.data2 <- as.data.frame(matrix(nrow = 0,ncol = 52))
  print(a)
}
proc.time()-tm

tsd_1 <- rbind_all(list(tsd_1,tsd_2,tsd_3))
tsd_2 <- rbind_all(list(tsd_4,tsd_5,tsd_6))
tsd_3 <- rbind_all(list(tsd_7,tsd_8,tsd_9))
tsd_4 <- rbind_all(list(tsd_10,tsd_11,tsd_12))
rm(tsd_5, tsd_6, tsd_7,tsd_8,tsd_9,tsd_10,tsd_11,tsd_12)

tsd_1 <- rbind_all(list(tsd_1,tsd_2))
tsd_2 <- rbind_all(list(tsd_3,tsd_4))
rm(tsd_3, tsd_4)

ts.data2 <- rbind_all(list(tsd_1,tsd_2))
rm(tsd_1,tsd_2)
write.csv(ts.data2, "tsdata.csv")

# Now fit xgboost model on train data
names(tr.data2)[53]
label <- tr.data2$V53
tr <- sample(nrow(tr.data2), 20e05)
dtrain <- xgb.DMatrix(data = data.matrix(tr.data2[tr,-53]), label = label[tr])
dval <- xgb.DMatrix(data = data.matrix(tr.data2[-tr,-53]), label = label[-tr])
watchlist<-list(val=dval,train=dtrain)

param <- list("objective" = "reg:linear",    # multiclass classification 
              #"num_class" = num.class,    # number of classes 
              "eval_metric" = "rmse",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 15,    # maximum depth of tree 
              "eta" = 0.35,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 0.7,    # part of data instances to grow tree 
              "colsample_bytree" = 0.6  # subsample ratio of columns when constructing each tree 
              # "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

fit <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 500, #300, #280
                    verbose             = 1,
                    early.stop.round    = 20,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

trpred <- predict(fit, data.matrix(tr.data2[,-53]))
#(abs((trpred-tr_features[,52]))/sum(abs(mean(tr_features[,52])-tr_features[,52])))
sum(abs(trpred - label))/nrow(train)

tspred <- predict(fit, data.matrix(ts.data2))

#Create submission file
sub <- read.csv("sample_submission_2.csv")
sub$Predicted <- tspred
write.csv(sub, "sub1.csv", row.names = FALSE)
#------------- Not required -----------------
#Second level prediction - train data preparation
trows <- 40000*62
tr.data <- as.data.frame(matrix(nrow = 0,ncol = 3))
names(tr.data)[3] <- "target2" 
tm <- proc.time()
j <- 1
for (n in 1:nrow(train)){
  vec <- tr_features[n,"target"]
  t <- matrix(rep(vec,62),nrow = 62, byrow = T)
  target2 <- as.vector(t(response[n,]))
  t <- as.data.frame(cbind(t, 1:62, target2))
  tr.data <- rbind_all(list(tr.data,t))
#   tr.data[j:(j+61),] <- t
#   j <- n*62 + 1
  if(n%%5000 == 0){
    print(n/1000)
  }
}
proc.time()-tm
#-----------------------------------
sub1$Predicted<-as.numeric(replicate(60000,apply(train[,148:209],FUN=median,MARGIN=2)))
median_62 <- apply(train[,148:209],2,FUN = median)
median_sub <- median_62*(.86)
#tr.data2 <- read.csv("trdata.csv")
median_sub <- rep(median_sub,120000)
sub$Predicted <- median_sub
write.csv(sub, "sub4.csv", row.names = FALSE)
# predictors - median value of return for a given minute, sd of returns from 1 to 120 minutes
tr_features$target <- train$Weight_Intraday

stdev <- apply(train[,29:(119+28)],1,FUN = sd)

tr_features$returnSD <- stdev
#build a model for ret_plusone
lm1 <- lm(target ~ ., data = tr_features[,-c(26,27)])
summary(lm1)
train$Weight_Daily[1:10]
#------- New Idea -------------
# fit a model for every return as one observation. for each stock there will be 62 observations per stock. ret121 to re180, ret_PlusOne and ret_PlusTwo.
# 4 Predictors - median value of the corresponding return, sd & median of the corresponding stock, corresponding weight value (two types of weights - Intra day and daily). For ret_121 to ret_180 intra day weight value and for ret_+1 and ret_+2 will have daily weight value
# Weight values are not available for test. We need to predict the weights for test data first

#
tr_features$returnSD <- apply(train[,29:(119+28)],1,FUN = sd)
tr_features$returnMD <- apply(train[,29:(119+28)],1,FUN = median)
tr_features$target <- train$Weight_Intraday
#predict intraday weight
label <- tr_features$target
matrx <- tr_features[,-30]
tr <- sample(nrow(tr_features), 32000)
dtrain <- xgb.DMatrix(data = data.matrix(matrx[tr,]), label = label[tr])
dval <- xgb.DMatrix(data = data.matrix(matrx[-tr,]), label = label[-tr])
watchlist = list(val = dval, train = dtrain)

param <- list("objective" = "reg:linear",    # multiclass classification 
              #"num_class" = num.class,    # number of classes 
              "eval_metric" = "rmse",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 5,    # maximum depth of tree 
              "eta" = 0.25,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 0.6,    # part of data instances to grow tree 
              "colsample_bytree" = 1  # subsample ratio of columns when constructing each tree 
              # "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

fit <- xgb.train(params = param, 
                 data = dtrain, 
                 nrounds = 300, 
                 watchlist = watchlist, 
                 maximize = FALSE, 
                 verbose = 1, 
                 early.stop.round = 30)
summary(fit)
tr_intraWt <- predict(fit, data.matrix(matrx))

#prepare test data
ts_features <- test[,2:28]
ts_features <- centralImputation(ts_features)
fun <- function(x){
  x[is.na(x)] <- median(x,na.rm = TRUE)
  x
}
t <- as.data.frame(apply(test[,29:147],2,fun))

ts_features$returnSD <- apply(t,1,FUN = sd)
ts_features$returnMD <- apply(t,1,FUN = median)
sum(is.na(ts_features))
#predict intraday weight for test
ts_intraDayWt <- predict(fit, data.matrix(ts_features))

#-----Predict daily weight value for test data
tr_features$target <- train$Weight_Daily
#predict intraday weight
label <- tr_features$target
matrx <- tr_features[,-30]
tr <- sample(nrow(tr_features), 32000)
dtrain <- xgb.DMatrix(data = data.matrix(matrx[tr,]), label = label[tr])
dval <- xgb.DMatrix(data = data.matrix(matrx[-tr,]), label = label[-tr])
watchlist = list(val = dval, train = dtrain)

param <- list("objective" = "reg:linear",    # multiclass classification 
              #"num_class" = num.class,    # number of classes 
              "eval_metric" = "rmse",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 5,    # maximum depth of tree 
              "eta" = 0.25,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 0.6,    # part of data instances to grow tree 
              "colsample_bytree" = 1  # subsample ratio of columns when constructing each tree 
              # "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

fit <- xgb.train(params = param, 
                 data = dtrain, 
                 nrounds = 500, 
                 watchlist = watchlist, 
                 maximize = FALSE, 
                 verbose = 1, 
                 early.stop.round = 30)
summary(fit)
#predict intraday weight for test
ts_DailyWt <- predict(fit, data.matrix(ts_features))
# create train data for model fitting
#median_62, sdst, medst, wt
response <- train[,148:209]
tr.data1 <- as.data.frame(matrix(nrow = 0,ncol = 5))
names(tr.data1) <- c("median_62","V2","V3","V4","target")
tr.data2 <- tr.data1
tm <- proc.time()
for (n in 30001:40000){
  target <- as.vector(t(response[n,]))
  t <- as.data.frame(cbind(median_62, rep(tr_features$returnSD[n],62), rep(tr_features$returnMD[n],62),c(rep(train$Weight_Intraday[n],60),rep(train$Weight_Daily[n],2)),target))
  tr.data1 <- rbind_all(list(tr.data1,t))
  if(n%%500 == 0){
    tr.data2 <- rbind_all(list(tr.data2, tr.data1))
    tr.data1 <- as.data.frame(matrix(nrow = 0,ncol = 5))
    names(tr.data1) <- c("median_62","V2","V3","V4","target")
    print(n/1000)
  }
}
proc.time() - tm

tr1_10 <- tr.data2
tr10_20 <- tr.data2
tr20_30 <- tr.data2
tr30_40 <- tr.data2

tr20 <- rbind_all(list(tr1_10,tr10_20))
tr.data2 <- rbind_all(list(tr20,tr20_30,tr30_40))
rm(tr1_10,tr10_20,tr20_30,tr30_40,tr20)

#prepare test data
# 4 predictors as in tr.data2
ts.data1 <- as.data.frame(matrix(nrow = 0,ncol = 4))
names(ts.data1) <- c("median_62","V2","V3","V4")
ts.data2 <- ts.data1
tm <- proc.time()
for (a in 1:12){
  for (n in (1+(a-1)*10000):(a*10000)){
    t <- as.data.frame(cbind(median_62, rep(ts_features$returnSD[n],62), rep(ts_features$returnMD[n],62),c(rep(ts_intraDayWt[n],60),rep(ts_DailyWt[n],2))))
    ts.data1 <- rbind_all(list(ts.data1,t))
    if(n%%500 == 0){
      ts.data2 <- rbind_all(list(ts.data2,ts.data1))
      ts.data1 <- as.data.frame(matrix(nrow = 0,ncol = 4))
      names(ts.data1) <- c("median_62","V2","V3","V4")
      print(n/1000)
    }
  }
  name <- paste0("tsd_",a)
  ts.data2 <- data.frame(lapply(ts.data2, as.numeric))
  assign(name, ts.data2)
  ts.data2 <- ts.data1
  print(a)
}
proc.time()-tm

tsd_1 <- rbind_all(list(tsd_1,tsd_2,tsd_3))
tsd_2 <- rbind_all(list(tsd_4,tsd_5,tsd_6))
tsd_3 <- rbind_all(list(tsd_7,tsd_8,tsd_9))
tsd_4 <- rbind_all(list(tsd_10,tsd_11,tsd_12))
rm(tsd_5, tsd_6, tsd_7,tsd_8,tsd_9,tsd_10,tsd_11,tsd_12)

tsd_1 <- rbind_all(list(tsd_1,tsd_2))
tsd_2 <- rbind_all(list(tsd_3,tsd_4))
rm(tsd_3, tsd_4)

ts.data2 <- rbind_all(list(tsd_1,tsd_2))
rm(tsd_1,tsd_2)

# fit linear model first
label <- tr.data2$target
matrx <- tr.data2[,-c(4:6)]
tr <- sample(nrow(tr.data2), 2e06)
dtrain <- xgb.DMatrix(data = data.matrix(matrx[tr,]), label = label[tr])
dval <- xgb.DMatrix(data = data.matrix(matrx[-tr,]), label = label[-tr])
watchlist = list(val = dval, train = dtrain)

param <- list("objective" = "reg:linear",    # multiclass classification 
              #"num_class" = num.class,    # number of classes 
              "eval_metric" = "rmse",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 5,    # maximum depth of tree 
              "eta" = 0.2,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 0.6,    # part of data instances to grow tree 
              "colsample_bytree" = 1  # subsample ratio of columns when constructing each tree 
              # "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)

fit <- xgb.train(params = param, 
                 data = dtrain, 
                 nrounds = 500, 
                 watchlist = watchlist, 
                 maximize = FALSE, 
                 verbose = 1, 
                 early.stop.round = 30)

tspred <- predict(fit, data.matrix(ts.data2[,-4]))
sub$pred <- tspred
sub$pred2 <- sub$Predicted*0.96+sub$pred*.04
sub1 <- sub[,-c(2,3)]
names(sub1)[2] <- "Predicted"
write.csv(sub1, "sub7.csv", row.names = FALSE)
#0.92*.96*median + 0.04*xgb model -> 83rd position
#----------------- New Approach - Clustering ----------
# cluster the train records
tr_scaled <- scale(tr_features)

# Check where the cluster partitioning is good
wss <- 1:10
for (i in 1:10){
  fit.i <- kmeans(tr_scaled, i)
  wss[i] <- sum(fit.i$withinss)
}
plot(x = wss,lty = 1, type = "l")
line(wss)
centers <- which.min(wss)

centers <- 10
tm <- proc.time()
km.train <- kmeans(tr_scaled,centers)
proc.time()-tm


#get target values of train
target <- train[,148:209]
# Get median return value for each cluster in train
target$cluster <- km.train$cluster
median_62 <- matrix(0,nrow = 62, ncol =centers)
for (i in 1:centers)
  median_62[1:62,i] <- apply(target[target$cluster==i,1:62],2,median)
#partition test according to train clusters
# prepare test data
ts_features <- test[,2:28]
ts_features <- centralImputation(ts_features)
ts_scaled <- scale(ts_features)
# for each record in test, find the closest cluster center and assign that cluster number
#ts.clust <- predict(km.train, ts_scaled)
centers <- km.train$centers
clusterme <- function(x, centers){
  dist <- apply(centers,1, function(c) {sum((x-c)^2)})
  return(which.min(dist))
}
ts.clust <- apply(ts_scaled,1,function(x) clusterme(x, centers))
#read latest best sub
sub7 <- read.csv("sub7.csv")

a= NULL
pred = a
for (c in ts.clust){
  a <- c(a,median_62[,c])
  if ((length(a)/62)%%2000 == 0) {
    pred <- c(pred,a)
    a = NULL
    print(length(pred)/1e05)
    }
}

sub8 <- sub7
sub8$Predicted <- pred
write.csv(sub8,"sub8.csv", row.names = FALSE)
sub8 <- read.csv("sub8.csv")
# try ensemble
pred <- sub8$Predicted * 0.04 + sub7$Predicted*0.96
sub8$Predicted <- sub8$Predicted*.91
write.csv(sub8,"sub11.csv", row.names = FALSE)

g <- function(x){
  return(1/(1+exp(-x)))
}
g(5)
g(-5)
# all the features and extracted variable with median return of a stock as target.
target1 <- as.data.frame(apply(train[,148:207],1,median))
target2 <- as.data.frame(apply(train[,208:209],1,mean))
#train data
tr_features$returnSD <- apply(train[,29:(119+28)],1,FUN = sd)
tr_features$returnMD <- apply(train[,29:(119+28)],1,FUN = median)

#test data
ts_features$returnSD <- apply(test[,29:(119+28)],1,function(x) sd(x,na.rm = T))
ts_features$returnMD <- apply(test[,29:(119+28)],1,function(x) median(x, na.rm = T))

#fit xgb on target1
tr <- sample(nrow(tr_features),32000)
dtrain <- xgb.DMatrix(data.matrix(tr_features[tr,]), label = target1[tr,])
dval <- xgb.DMatrix(data.matrix(tr_features[-tr,]), label = target1[-tr,])
watchlist <- list(train=dtrain, val=dval)

param <- list("objective" = "reg:linear",    # multiclass classification 
              #"num_class" = num.class,    # number of classes 
              "eval_metric" = "rmse",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 5,    # maximum depth of tree 
              "eta" = 0.01,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 0.5,    # part of data instances to grow tree 
              "colsample_bytree" = 0.5  # subsample ratio of columns when constructing each tree 
              # "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)
fit <- xgb.train(params = param, 
                 data= dtrain, 2000, 
                 watchlist = watchlist, 
                 #obj = "minimize", 
                 verbose = 1,
                 early.stop.round = 30, 
                 maximize = FALSE)
#predict median value of minute wise return for test
sum(is.na(ts_features[,]))
ts_features[is.na(ts_features)] <- 0
ts_pred1 <- predict(fit, data.matrix(ts_features)) #Minute wise return

#Predict daywise return
#fit xgb on target1
tr <- sample(nrow(tr_features),32000)
dtrain <- xgb.DMatrix(data.matrix(tr_features[tr,]), label = target2[tr,])
dval <- xgb.DMatrix(data.matrix(tr_features[-tr,]), label = target2[-tr,])
watchlist <- list(train=dtrain, val=dval)

param <- list("objective" = "reg:linear",    # multiclass classification 
              #"num_class" = num.class,    # number of classes 
              "eval_metric" = "rmse",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 5,    # maximum depth of tree 
              "eta" = 0.01,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 0.5,    # part of data instances to grow tree 
              "colsample_bytree" = 0.5  # subsample ratio of columns when constructing each tree 
              # "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)
fit2 <- xgb.train(params = param, 
                 data= dtrain, 2000, 
                 watchlist = watchlist, 
                 #obj = "minimize", 
                 verbose = 1,
                 early.stop.round = 30, 
                 maximize = FALSE)
#predict median value of minute wise return for test
sum(is.na(ts_features[,]))
ts_features[is.na(ts_features)] <- 0
ts_pred2 <- predict(fit, data.matrix(ts_features)) #day wise return
#Validate on train
tr_pred1 <- predict(fit, data.matrix(tr_features))
tr_pred2 <- predict(fit2, data.matrix(tr_features))
#Actual returns
s <- lapply(1:nrow(train), function(n) {c(rep(target1[n,],60), rep(target2[n,],2))})
target <- unlist(s)
# Predicted returns on train
s <- lapply(1:nrow(train), function(n) {c(rep(tr_pred1[n],60), rep(tr_pred2[n],2))})
tr_pred <- unlist(s)
#Create weights vector
w <- lapply(1:nrow(train), function(n){c(rep(train[n,210],60), rep(train[n,211],2))})
w <- unlist(w)
names(train)[211]
mean(abs(target-tr_pred)*w)


#make a file for submission
s <- lapply(1:nrow(test), function(n) {c(rep(ts_pred1[n],60), rep(ts_pred2[n],2))})
pred <- unlist(s)
sub13 <- read.csv("sub13.csv")
sub12 <- read.csv("sub12.csv")

predicted <- 0.96*sub13$Predicted + 0.04*pred
sub12$Predicted <- predicted
write.csv(sub12, "sub15.csv", row.names = FALSE)
