# --------------------------
# Lendo um banco de dados
dados <- read.table(url("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"),
                    sep=",")
# -------------------------
# Renomeando as colunas
names <- c("Sex","Length","Diameter","Height","Whole",
           "Shucked weight","Viscera weight","Shell weight",
           "Rings")
# ----------------
colnames(dados) <- names
# ----------------------------------
# A coluna a ser classificada é a primeira no banco de dados
head(dados,3)
# ----------------------------------
# Separando em Treinamento e Teste (80% / 20%)
amostratreino <- sample(1:nrow(dados), round(0.8*nrow(dados)), replace = F)
# ----------------
Treinamento <- dados[amostratreino,]
# ----------------
Teste <- dados[-amostratreino,]
# ----------------------------------
# ----------------------------------
# Aplicando o Classificador Gaussian Naive Bayes
library(e1071)
fit_NB <- naiveBayes(x = Treinamento[,-1], y = Treinamento[,1])
predictions <- predict(fit_NB,newdata=Teste[,-1], type = "class")
# ---------------
source(url("https://raw.githubusercontent.com/Jodavid/FuzzyGaussianNaiveBayes/main/FuzzyGaussianNaiveBayes.R"))
fit_FGNB <- FuzzyGaussianNaiveBayes(train =  Treinamento[,-1], test= Teste[,-1], cl = Treinamento[,1], metd=1, cores=2)
# ----------------------------------
# ----------------------------------
# Verificando a Acurácia
# ---------------------
# Gaussian Naive Bayes
TabelaNB <- table(predictions,Teste[,1]) 
TabelaNB # Matriz de Confusão
sum(diag(TabelaNB))/sum(TabelaNB) # Acurácia
# ---------------
# Gaussian Naive Bayes com parâmetros Fuzzy
TabelaFGNB <- table(fit_FGNB,Teste[,1]) 
TabelaFGNB # Matriz de Confusão
sum(diag(TabelaFGNB))/sum(TabelaFGNB) # Acurácia



# ----------------------------------------------------------------------------
# Banco iris
# -----------
set.seed(1) # determinando uma semente
x <- sample(1:150,150,replace = F) # Gerando uma amostra para aleatorizar os dados
iris <- iris[x,] # aleatorizando os dados do iris
# ----------------------------------
# A coluna a ser classificada é a primeira no banco de dados
head(iris,3)
# ----------------------------------
# Separando em Treinamento e Teste (70% / 30%)
amostratreino <- sample(1:nrow(iris), round(0.8*nrow(iris)), replace = F)
# ----------------
Treinamento <- iris[amostratreino,]
# ----------------
Teste <- iris[-amostratreino,]
# ----------------------------------
source(url("https://raw.githubusercontent.com/Jodavid/FuzzyGaussianNaiveBayes/main/FuzzyGaussianNaiveBayes.R"))
fit_FGNB <- FuzzyGaussianNaiveBayes(train =  Treinamento[,-5], test= Teste[,-5], cl = Treinamento[,5], metd=1, cores=2)
# ----------------------------------
# ----------------------------------
# ---------------
# Gaussian Naive Bayes com parâmetros Fuzzy
TabelaFGNB <- table(fit_FGNB,Teste[,5]) 
TabelaFGNB # Matriz de Confusão
sum(diag(TabelaFGNB))/sum(TabelaFGNB) # Acurácia


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
