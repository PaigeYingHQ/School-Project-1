#feature
#导入原始训练集
raw_train <- read.csv("train.csv",header = T,stringsAsFactors = F)

#移除缺失值
which(is.na(raw_train),arr.ind = T)
raw_train <- raw_train[complete.cases(raw_train),]
raw_train <- raw_train[,-1]
#raw_train中有98个特征，5449个样本

flag <- raw_train[,'flag']
write.csv(flag,file = "flag.csv",row.names = F)

#剔除标准差为0的特征
SD <- apply(raw_train, 2, sd)
index <- which(SD == 0)
print(colnames(raw_train[,index]))
raw_train <- raw_train[,-index]

#------------自定义函数(howMany)----------#
howMany <- function(matrix,method = 1,k)
{
  num <- c()
  for (i in 1:NROW(matrix))
  {
    if(method == 1)
    {
      n <- length(which(matrix[i,] == k))
    }
    else
    {
      n <- length(which(matrix[i,] < k))
    }
    num <- c(num,n)
  }
  return(num)
}
#-----------------------------------------#
#对每条样本进行条件计数，统计样本取值小于0的个数
index <- which(colnames(raw_train)=='flag')
Zero <- howMany(raw_train[,-index],1,0)
Negtive <- howMany(raw_train[,-index],0,0)
Count <- cbind(Zero,Negtive)
write.csv(Count,file = "WhyNotTry.csv",row.names = F)

#将card相关的分类特征转换成哑变量
#目标特征: 
#1.card_zh_cnt        2.card_xz_cnt   3.card_brand_cnt
#4.card_product_cnt   5.card_zh       6.card_xz
#7.card_brand         8.card_product
library(nnet)
index <- c('card_zh_cnt','card_xz_cnt','card_brand_cnt','card_product_cnt',
            'card_zh','card_xz','card_brand','card_product')
Tfeatures <- raw_train[,index]
Tfeatures_changed <- apply(Tfeatures, 2, class.ind)

card_feat <- matrix(nrow = NROW(raw_train),ncol = 1)
for(i in Tfeatures_changed)
{
  card_feat <- cbind(card_feat,i)
}
card_feat <- card_feat[,-1]

na <- paste("card_",1:NCOL(card_feat))
na <- gsub("([ ])",'',na)
colnames(card_feat) <- na
write.csv(card_feat,"card_feat.csv",row.names = F)

#从raw_train中删除card相关的分类特征
raw_train <- raw_train[,9:NCOL(raw_train)]
#raw_train中有79个特征，5449个样本


#列举除card之外的分类特征
library(lattice)
library(survival)
library(Formula)
library(ggplot2)
library(Hmisc)

index <- which(colnames(raw_train) == 'flag')
describeResult <- Hmisc::describe(raw_train[,-index])

Cfeatures <- c()         #Cfeatures means categorical features
for(i in describeResult)
{
  if(as.integer(i$counts['unique']) <= 10)
  {
    print(i$descript)
    Cfeatures <- c(Cfeatures,i$descript)
  }
}

#对其他分类特征（即上面打印的特征）进行哑变量处理
Tfeatures <- raw_train[,Cfeatures]
Tfeatures_changed <- apply(Tfeatures, 2, class.ind )

categ_feat <- matrix(nrow = NROW(raw_train),ncol = 1)
for(i in Tfeatures_changed)
{
  categ_feat <- cbind(categ_feat,i)
}
categ_feat <- categ_feat[,-1]
na <- paste("CategF_",1:34,seq = '')
na <- gsub("([ ])",'',na) 
colnames(categ_feat) <- na
write.csv(categ_feat,file = "categ_feat.csv",row.names = F)

#合并card_feat和categ_geat
binaryfeat <- cbind(card_feat,categ_feat)
write.csv(binaryfeat,file = "binaryfeat.csv",row.names = F)

#从raw_train中删除上面的分类特征
index <- c()
for(i in 1:length(Cfeatures))
{
  loc <- which(colnames(raw_train) == Cfeatures[i])
  index <- c(index,loc)
}
raw_train <- raw_train[,-index]
#raw_train中只剩下73个数值特征和一个flag，有样本5449个

#对数值特征进行排序
index <- which(colnames(raw_train) == 'flag')
numfeat_rank <- apply(raw_train[,-index], 2, rank)

na <- paste(colnames(raw_train[,-index]),'rank')
na <- gsub("([ ])",'_',na)
colnames(numfeat_rank) <- na
write.csv(numfeat_rank,file = "numfeat_rank.csv", row.names = F)

#---------------自定义函数(MinMaxScaler)-----------------#
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
#---------------------------------------------------------#

#对raw_train进行去量纲
index <- which(colnames(raw_train) == 'flag')
Tmatrix <- MinMaxScaler(raw_train[,-index])
write.csv(Tmatrix,file = "numfeat_stand.csv", row.names = F)

Tmatrix <- apply(raw_train[,-index], 2, rank)

#计算排序后的特征两两交叉运算的结果，并将结果作为新的特征
colname1 <- c()
colname2 <- c()
colname3 <- c()
Matrix1 <- matrix(nrow = NROW(Tmatrix),ncol = 1)
Matrix2 <- matrix(nrow = NROW(Tmatrix),ncol = 1)
Matrix3 <- matrix(nrow = NROW(Tmatrix),ncol = 1)
for(i in 1:(NCOL(Tmatrix)-1))
{
  for(j in (i+1):NCOL(Tmatrix))
  {
    Matrix1 <- cbind(Matrix1,Tmatrix[,i]*Tmatrix[,j])
    colname1 <- c(colname1,paste('X',as.character(i),'mu','X',as.character(j)))
    
    Matrix2 <- cbind(Matrix2,Tmatrix[,i]+Tmatrix[,j])
    colname2 <- c(colname2,paste('X',as.character(i),'p','X',as.character(j)))
    
    Matrix3 <- cbind(Matrix3,Tmatrix[,i]-Tmatrix[,j])
    colname3 <- c(colname3,paste('X',as.character(i),'mi','X',as.character(j)))
  }
}
Matrix1 <- Matrix1[,-1]
Matrix2 <- Matrix2[,-1]
Matrix3 <- Matrix3[,-1]

colname1 <- gsub("([ ])",'',colname1)
colname2 <- gsub("([ ])",'',colname2)
colname3 <- gsub("([ ])",'',colname3)
colnames(Matrix1) <- colname1
colnames(Matrix2) <- colname2
colnames(Matrix3) <- colname3

#----------------------------------自定义函数(rank2group)-------------------------------#
rank2group <- function(Matrix1)
{
  Quantile_Matrix <- matrix(nrow = 9,ncol = 1) 
  for(i in 1:NCOL(Matrix1))
  {
    Quantile_Vector <- quantile(Matrix1[,i],probs = seq(from = 0.1, to = 0.9, by = 0.1))
    Quantile_Matrix <- cbind(Quantile_Matrix,Quantile_Vector)
  }
  Quantile_Matrix <- Quantile_Matrix[,-1]
  colnames(Quantile_Matrix) <- colnames(Matrix1)
  
  for( i in 1:NCOL(Matrix1))
  {
    Q <- Quantile_Matrix[,i]
    
    Tvector <- Matrix1[,i]
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
    
    Matrix1[index1,i] <- 1
    Matrix1[index2,i] <- 2
    Matrix1[index3,i] <- 3
    Matrix1[index4,i] <- 4
    Matrix1[index5,i] <- 5
    Matrix1[index6,i] <- 6
    Matrix1[index7,i] <- 7
    Matrix1[index8,i] <- 8
    Matrix1[index9,i] <- 9
    Matrix1[index10,i] <- 10
  }
  return(Matrix1)
}
#--------------------------------------------------------------------------------------#
#对三个矩阵数据分组赋值
Matrix1 <- rank2group(Matrix1)
Matrix2 <- rank2group(Matrix2)
Matrix3 <- rank2group(Matrix3)

R_train <- rank2group(numfeat_rank)


#-------------------------自定义函数(condition_count)-----------------------------#
condition_count <- function(Matrix1)
{
  Count_Matrix1 <- matrix(nrow = 1,ncol = 10)
  for(i in 1:NROW(Matrix1))
  {
    Tvector <- c()
    for(j in 1:10)
    {
      Tvector[j] <- length(which(Matrix1[i,] == j))
    }
    Count_Matrix1 <- rbind(Count_Matrix1,Tvector)
  }
  Count_Matrix1 <- Count_Matrix1[-1,]
  row.names(Count_Matrix1) <- row.names(Matrix1)
  colname <- paste('N',1:10)
  colname <- gsub("([ ])",'',colname)
  colnames(Count_Matrix1) <- colname
  return(Count_Matrix1)
}
#---------------------------------------------------------------------------------#
#对每个样本进行条件计数
Count_Matrix1 <- condition_count(Matrix1)
Count_Matrix2 <- condition_count(Matrix2)
Count_Matrix3 <- condition_count(Matrix3)
Count_train <- condition_count(R_train)

colname <- paste('NA',1:10)
colname <- gsub("([ ])",'',colname)
colnames(Count_Matrix1) <- colname

colname <- paste('NB',1:10)
colname <- gsub("([ ])",'',colname)
colnames(Count_Matrix2) <- colname

colname <- paste('NC',1:10)
colname <- gsub("([ ])",'',colname)
colnames(Count_Matrix3) <- colname

colname <- paste('ND',1:10)
colname <- gsub("([ ])",'',colname)
colnames(Count_train) <- colname

#把合成特征组合成一个矩阵
Count_Matrix <- cbind(Count_Matrix1,Count_Matrix2,Count_Matrix3)
write.csv(Count_Matrix,file = "numfeat_transform.csv",row.names = F)
write.csv(Count_train,file = "raw_train_rank_transform.csv",row.names = F)

