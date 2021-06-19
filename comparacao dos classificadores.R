data(iris)

set.seed(1)
x <- sample(1:150,150,replace = F)
iris <- iris[x,]
train2 <- iris[1:100,]
test2 <- iris[101:150,]

# ---- KNN
saidaknn <- class::knn(train = train2[,-5], test = test2[,-5], cl = train2[,5], k = 3 )
table(saidaknn,test2[,5])

# ---- Fuzzy Gaussian Naive Bayes
saida <- fuzzy_gau_nb(train =  train2[,-5], test= test2[,-5], cl = train2[,5], metd=1, cores=2)
table(saida,test2[,5])

# ---- Naive Naive Bayes
library(e1071)
fit_NB <- naiveBayes(x = train2[,-5], y = train2[,5])
predictions <- predict(fit_NB,newdata=test2[,-5], type = "class")
table(predictions,test2[,5])
