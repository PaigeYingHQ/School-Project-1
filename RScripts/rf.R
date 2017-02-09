train1 <- read.csv("binaryfeat.csv", header = T, stringsAsFactors = F)
train2 <- read.csv("numfeat_stand.csv", header = T, stringsAsFactors = F) #去过量纲
train3 <- read.csv("numfeat_rank.csv", header = T, stringsAsFactors = F)
train4 <- read.csv("numfeat_transform.csv", header = T, stringsAsFactors = F)
train5 <- read.csv("raw_train_rank_transform.csv", header = T, stringsAsFactors = F)
train11 <- read.csv("WhyNotTry.csv",header = T,stringsAsFactors = F)

#-----------自定义函数(MinMaxScaler)-------------#
MinMaxScaler <- function(Matrix)              
{
  for(i in 1:NCOL(Matrix))
  {
    Min <- min(Matrix[,i])
    Max <- max(Matrix[,i])
    Matrix[,i] <- (Matrix[,i]-Min)/(Max-Min)
  }
  return(Matrix)
}
#------------------------------------------------#

#对train3-5进行去量纲
train3 <- MinMaxScaler(train3)
train4 <- MinMaxScaler(train4)
train5 <- MinMaxScaler(train5)
#train11 <- MinMaxScaler(train11)

#合并所有特征集
train <- cbind(train1,train2,train3,train4,train5,train11)


#--------------自定义函数(disorder)--------------#
disorder <- function(Matrix,seed)
{
  set.seed(seed)
  index <- round(runif(NCOL(Matrix))*NCOL(Matrix))
  Matrix <- Matrix[,index]
  return(Matrix)
}
#------------------------------------------------#

#对train的特征进行随机排序
train_disorder1 <- disorder(train,101)
train_disorder2 <- disorder(train,169)
train_disorder3 <- disorder(train,2689)
train_disorder4 <- disorder(train,8008)
train_disorder5 <- disorder(train,518)

#将flag与train_disorder进行捆绑
flag <- read.csv("flag.csv",header = T,stringsAsFactors = F)
flag <- as.vector(flag$x)
train_disorder1 <- cbind(train_disorder1,flag)
train_disorder2 <- cbind(train_disorder2,flag)
train_disorder3 <- cbind(train_disorder3,flag)
train_disorder4 <- cbind(train_disorder4,flag)
train_disorder5 <- cbind(train_disorder5,flag)

#-------------------------自定义函数(lars_feat)-----------------------------------#
lars_feat <- function(Matrix)
{
  index <- which(colnames(Matrix) == 'flag')
  lar <- lars(as.matrix(Matrix[,-index]),as.matrix(Matrix[,index]),type = 'lasso')
  index_Cp <- as.integer(names(lar$Cp[which.min(lar$Cp)]))+1
  coeff <- coef.lars(lar, mode = "step", s = index_Cp)
  feat <- names(coeff[coeff!=0])
  TMatrix <- cbind(Matrix[,feat],flag = Matrix[,'flag'])
  return(TMatrix)
}
#---------------------------------------------------------------------------------#

#使用lasso回归对每个train_disorder挑选特征
library(lars)
train_disorder1.2 <- lars_feat(train_disorder1)
train_disorder2.2 <- lars_feat(train_disorder2)
train_disorder3.2 <- lars_feat(train_disorder3)
train_disorder4.2 <- lars_feat(train_disorder4)
train_disorder5.2 <- lars_feat(train_disorder5)

#分离train和test，test之间的区别为特征排序的不同和特征选择的不同
train1 <- train_disorder1.2[1:5000,]
test1 <- train_disorder1.2[5001:5449,]
prop.table(table(train1[,'flag']))
prop.table(table(test1[,'flag']))

train2 <- train_disorder2.2[1:5000,]
test2 <- train_disorder2.2[5001:5449,]
prop.table(table(train2[,'flag']))
prop.table(table(test2[,'flag']))

train3 <- train_disorder3.2[1:5000,]
test3 <- train_disorder3.2[5001:5449,]
prop.table(table(train3[,'flag']))
prop.table(table(test3[,'flag']))

train4 <- train_disorder4.2[1:5000,]
test4 <- train_disorder4.2[5001:5449,]
prop.table(table(train4[,'flag']))
prop.table(table(test4[,'flag']))

train5 <- train_disorder5.2[1:5000,]
test5 <- train_disorder5.2[5001:5449,]
prop.table(table(train5[,'flag']))
prop.table(table(test5[,'flag']))

#-------------------------自定义函数(model_rf)--------------------------------#
model_rf <- function(train,test,ntree)
{
  Ntree <- ntree
  index <- which(colnames(train) == 'flag')
  rf <- randomForest(x = train[,-index],
                      y = as.factor(train[,'flag']),
                      importance = T,
                      ntree = Ntree)
  
  index <- which(colnames(test) == 'flag')
  pred <- predict(rf,test[,-index],type = 'prob')[,2]
  
  library(pROC)
  auc2 <- auc(test[,'flag'],pred)
  print(auc2)
  
  pred.t <- predict(rf,train[,-index],type = 'prob')[,2]
  pred.t <- c(pred.t,pred)
  
  return(pred = pred.t)
}
#-----------------------------------------------------------------------------#
#建立模型
library(randomForest)
Ntree <- 100
pred1 <- model_rf(train1,test1,Ntree)
pred2 <- model_rf(train2,test2,Ntree)
pred3 <- model_rf(train3,test3,Ntree)
pred4 <- model_rf(train4,test4,Ntree)
pred5 <- model_rf(train5,test5,Ntree)

PRED <- cbind(pred1,pred2,pred3,pred4,pred5)
pred <- apply(PRED, 1, mean)
auc(flag,pred)

write.csv(PRED,file = "RF_PRED.csv",row.names = F)

#----------------------------------自定义函数(rank2group)-------------------------------#
rank2group <- function(vect)
{
    Q <- quantile(vect,probs = seq(from = 0.1, to = 0.9, by = 0.1))
    
    Tvector <- vect
    index1 <- which(Tvector <= Q[1])
    index2 <- which(Tvector > Q[1] & Tvector <= Q[2])
    index3 <- which(Tvector > Q[2] & Tvector <= Q[3])
    index4 <- which(Tvector > Q[3] & Tvector <= Q[4])
    index5 <- which(Tvector > Q[4] & Tvector <= Q[5])
    index6 <- which(Tvector > Q[5] & Tvector <= Q[6])
    index7 <- which(Tvector > Q[6] & Tvector <= Q[7])
    index8 <- which(Tvector > Q[7] & Tvector <= Q[8])
    index9 <- which(Tvector > Q[8] & Tvector <= Q[9])
    index10 <- which(Tvector > Q[9])
    
    vect[index1] <- 1
    vect[index2] <- 2
    vect[index3] <- 3
    vect[index4] <- 4
    vect[index5] <- 5
    vect[index6] <- 6
    vect[index7] <- 7
    vect[index8] <- 8
    vect[index9] <- 9
    vect[index10] <- 10
    return(vect)
}
#--------------------------------------------------------------------------------------#  
#将pred划分为分组数据
pred_group <- apply(PRED, 2, rank2group)
pred_group <- apply(pred_group, 1, mean)
write.csv(pred_group,file = "RF_PRED_GROUP.csv",row.names = F)