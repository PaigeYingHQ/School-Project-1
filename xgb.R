#在模型1的基础上重新设置train和test

train1 <- read.csv("binaryfeat.csv", header = T, stringsAsFactors = F)
train2 <- read.csv("numfeat_stand.csv", header = T, stringsAsFactors = F) #去过量纲
train3 <- read.csv("numfeat_rank.csv", header = T, stringsAsFactors = F)
train4 <- read.csv("numfeat_transform.csv", header = T, stringsAsFactors = F)
train5 <- read.csv("raw_train_rank_transform.csv", header = T, stringsAsFactors = F)
train10 <- read.csv("RF_PRED_GROUP.csv",header = T,stringsAsFactors = F)


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

#合并所有特征集
train <- cbind(train1,train2,train3,train4,train5)


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
train_disorder5 <- disorder(train,522)

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

#加入新生成的特征
train_disorder1.2 <- cbind(train_disorder1.2,train10)
train_disorder2.2 <- cbind(train_disorder2.2,train10)
train_disorder3.2 <- cbind(train_disorder3.2,train10)
train_disorder4.2 <- cbind(train_disorder4.2,train10)
train_disorder5.2 <- cbind(train_disorder5.2,train10)

#--------------自定义函数(disorder2)--------------#
disorder2 <- function(Matrix,seed)
{
  set.seed(seed)
  index <- round(runif(NROW(Matrix))*NROW(Matrix))
  Matrix <- Matrix[index,]
  return(Matrix)
}
#-------------------------------------------------#
#生成新样本排序
train1.3 <- disorder2(train_disorder1.2,165)
train2.3 <- disorder2(train_disorder2.2,2016)
train3.3 <- disorder2(train_disorder3.2,331)
train4.3 <- disorder2(train_disorder4.2,10086)
train5.3 <- disorder2(train_disorder5.2,999)

#分离train和test
train1 <- train1.3[1:5000,]
test1 <- train1.3[5001:5449,]
prop.table(table(train1[,'flag']))
prop.table(table(test1[,'flag']))

train2 <- train2.3[1:5000,]
test2 <- train2.3[5001:5449,]
prop.table(table(train2[,'flag']))
prop.table(table(test2[,'flag']))

train3 <- train3.3[1:5000,]
test3 <- train3.3[5001:5449,]
prop.table(table(train3[,'flag']))
prop.table(table(test3[,'flag']))

train4 <- train4.3[1:5000,]
test4 <- train4.3[5001:5449,]
prop.table(table(train4[,'flag']))
prop.table(table(test4[,'flag']))

train5 <- train5.3[1:5000,]
test5 <- train5.3[5001:5449,]
prop.table(table(train5[,'flag']))
prop.table(table(test5[,'flag']))


#---------------自定义函数(where)-------------#
where <- function(Matrix,seed)
{
  set.seed(seed)
  index <- round(runif(NROW(Matrix))*NROW(Matrix))
  return(index)
}
#---------------------------------------------#
#提取test1-5在原训练集中的位置
coord1 <- where(train_disorder1.2,165)
coord2 <- where(train_disorder2.2,2016)
coord3 <- where(train_disorder3.2,331)
coord4 <- where(train_disorder4.2,10086)
coord5 <- where(train_disorder5.2,999)

#建立模型
#-------------------------自定义函数(model_xgb)-----------------------------------#
model_xgb <- function(train,test)
{
  
  train <- as.matrix(train)
  test <- as.matrix(test)

  index <- which(colnames(train) == 'flag')
  train <- xgb.DMatrix(train[,-index],label = train[,index])
  
  #Modeling
  Eta <- 0.1
  Gamma <- 0
  Max_depth <- 2
  Subsample <- 1
  Colsample_bytree <- 1
  
  Lambda <- 0.27
  Lambda_bias <- 0.33
  Alpha <- 0.25
  
  Nround <- 20
  Base_score <- 0.5
  
  Param <- list(
    booster = 'gblinear',
    objective = 'binary:logistic',
    eta = Eta,
    gamma = Gamma,
    base_score = Base_score,
    subsample <- Subsample,
    colsample_bytree <- Colsample_bytree,
    max_depth <- Max_depth,
    
    lambda <- Lambda,
    lambda_bias <- Lambda_bias,
    alpha <- Alpha
  )
  
  xgb <- xgb.train(data = train,
                    nrounds = Nround, 
                    params = Param)
  
  index <- which(colnames(test) == 'flag')
  pred1 <- xgboost::predict(xgb,test[,-index])
  library(pROC)
  auc2 <- auc(test[,'flag'],pred1)
  print(auc2)
  
  pred2 <- xgboost::predict(xgb,train)
  pred_t <- c(pred2,pred1)
  
  return(pred_t)
}
#---------------------------------------------------------------------------------#
#将数据代入模型
library(xgboost)
pred1 <- model_xgb(train1,test1)
pred2 <- model_xgb(train2,test2)
pred3 <- model_xgb(train3,test3)
pred4 <- model_xgb(train4,test4)
pred5 <- model_xgb(train5,test5)

#读入flag
flag <- read.csv("flag.csv",header = T,stringsAsFactors = F)
flag <- flag$x

#对所有pred恢复原始排序
pred1_order <- pred1[order(coord1)]
pred2_order <- pred2[order(coord2)]
pred3_order <- pred3[order(coord3)]
pred4_order <- pred4[order(coord4)]
pred5_order <- pred5[order(coord5)]

#----------自定义函数(orderagain)----------#
orderagain <- function(coord)
{
  x <- c(1:length(coord))
  x <- x[coord]
  x <- x[order(coord)]
  return(x)
}
#------------------------------------------#
index1 <- orderagain(coord1)
index2 <- orderagain(coord2)
index3 <- orderagain(coord3)
index4 <- orderagain(coord4)
index5 <- orderagain(coord5)

PRED1 <- cbind(index1,pred1_order)
PRED2 <- cbind(index2,pred2_order)
PRED3 <- cbind(index3,pred3_order)
PRED4 <- cbind(index4,pred4_order)
PRED5 <- cbind(index5,pred5_order)

target <- rbind(PRED1,PRED2,PRED3,PRED4,PRED5)
index <- order(target[,1])
pred <- target[index,2]
coord <- target[index,1]

#-----------自定义函数(shrinkage)--------------#
shrinkage <- function(pred,coord)
{
  index <- c()
  new_pred <- c()
  
  x <- c(1:5449)
  x <- x[coord]
  x <- x[order(coord)]
  i <- 1
  while (i <= length(coord))
  {
    j <- 1
    while (j <= (length(coord)-i))
    {
      if(x[i+j] != x[i])
      {
        break
      }
      else
      {
        j = j + 1
      }
    }
    
    p <- 0
    for (k in 1:j)
    {
      p <- p + pred[i+k-1]
    }
    p <- p/j
    
    index <- c(index,x[i])
    new_pred <- c(new_pred,p)
    i <- i + j
  }
  PRED <- cbind(index,new_pred)
  return(PRED)
}
#-------------------------------------------#
#压缩预测值
s1 <- shrinkage(pred,coord)
flag_t <- flag[s1[,1]]
auc(flag_t,s1[,2])


