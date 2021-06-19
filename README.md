
# classificador Gaussian Naive Bayes com parâmetros Fuzzy

Última Atualização: 19-06-2021

## Visão geral

Venho compartilhar uma nova abordagem do algoritmo **Naive Bayes** que
junto com meu orientador [**Ronei Moraes**](mailto:ronei@de.ufpb.br)
propusemos no meu TCC de graduação em Estatística defendido no fim de
2014 na UFPB ([link para a documento do TCC intitulado como *‘Sistema de
avaliação de treinamento baseado em realidade virtual usando rede de
probabilidade fuzzy fundamentada na distribuição Normal
Fuzzy.’*](http://www.de.ufpb.br/graduacao/tcc/TCC2014p2Jodavid.pdf)), e
publicado em periódico esse ano (2021) com o título [*‘A New Bayesian
Network Based on Gaussian Naive Bayes with Fuzzy Parameters for Training
Assessment in Virtual
Simulators’*](https://link.springer.com/article/10.1007/s40815-020-00936-4)
(publicado na [International Journal of Fuzzy
Systems](https://www.springer.com/journal/40815)).

## Publicação:

Foi feito um post com detalhamentos teóricos na página:
<https://jodavid.github.io/post/>

## Utilização:

### Exemplo 1:

``` r
# --------------------------
# Banco iris
# -----------
set.seed(1) # determinando uma semente
x <- sample(1:150,150,replace = F) # Gerando uma amostra para aleatorizar os dados
iris <- iris[x,] # aleatorizando os dados do iris
# ----------------------------------
# A coluna a ser classificada é a primeira no banco de dados
head(iris,3)
#>     Sepal.Length Sepal.Width Petal.Length Petal.Width    Species
#> 68           5.8         2.7          4.1         1.0 versicolor
#> 129          6.4         2.8          5.6         2.1  virginica
#> 43           4.4         3.2          1.3         0.2     setosa
# ----------------------------------
# Separando em Treinamento e Teste (70% / 30%)
amostratreino <- sample(1:nrow(iris), round(0.8*nrow(iris)), replace = F)
# ----------------
Treinamento <- iris[amostratreino,]
# ----------------
Teste <- iris[-amostratreino,]
# ----------------------------------
source(url("https://raw.githubusercontent.com/Jodavid/FuzzyGaussianNaiveBayes/main/FuzzyGaussianNaiveBayes.R"))
#> Carregando pacotes exigidos: foreach
#> Carregando pacotes exigidos: iterators
#> Carregando pacotes exigidos: snow
#> 
#> Attaching package: 'snow'
#> The following objects are masked from 'package:parallel':
#> 
#>     clusterApply, clusterApplyLB, clusterCall, clusterEvalQ,
#>     clusterExport, clusterMap, clusterSplit, makeCluster, parApply,
#>     parCapply, parLapply, parRapply, parSapply, splitIndices,
#>     stopCluster
fit_FGNB <- FuzzyGaussianNaiveBayes(train =  Treinamento[,-5], test= Teste[,-5], cl = Treinamento[,5], metd=1, cores=2)
# ----------------------------------
# ----------------------------------
# Gaussian Naive Bayes com parâmetros Fuzzy
TabelaFGNB <- table(fit_FGNB,Teste[,5]) 
TabelaFGNB # Matriz de Confusão
#>             
#> fit_FGNB     setosa versicolor virginica
#>   setosa          9          0         0
#>   versicolor      0          9         2
#>   virginica       0          0        10
sum(diag(TabelaFGNB))/sum(TabelaFGNB) # Acurácia
#> [1] 0.9333333
```

### Exemplo 2:

Site do Banco de dados:
<https://archive.ics.uci.edu/ml/datasets/Abalone>

``` r
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
#>   Sex Length Diameter Height  Whole Shucked weight Viscera weight Shell weight
#> 1   M  0.455    0.365  0.095 0.5140         0.2245         0.1010         0.15
#> 2   M  0.350    0.265  0.090 0.2255         0.0995         0.0485         0.07
#> 3   F  0.530    0.420  0.135 0.6770         0.2565         0.1415         0.21
#>   Rings
#> 1    15
#> 2     7
#> 3     9
# ----------------------------------
# Separando em Treinamento e Teste (80% / 20%)
amostratreino <- sample(1:nrow(dados), round(0.8*nrow(dados)), replace = F)
# ----------------
Treinamento <- dados[amostratreino,]
# ----------------
Teste <- dados[-amostratreino,]
# ----------------------------------
# ----------------------------------
source(url("https://raw.githubusercontent.com/Jodavid/FuzzyGaussianNaiveBayes/main/FuzzyGaussianNaiveBayes.R"))
fit_FGNB <- FuzzyGaussianNaiveBayes(train =  Treinamento[,-1], test= Teste[,-1], cl = Treinamento[,1], metd=1, cores=2)
# ----------------------------------
# ----------------------------------
# Verificando a Acurácia
# ---------------------
# Gaussian Naive Bayes com parâmetros Fuzzy
TabelaFGNB <- table(fit_FGNB,Teste[,1]) 
TabelaFGNB # Matriz de Confusão
#>         
#> fit_FGNB   F   I   M
#>        F  58  10  70
#>        I  64 209  85
#>        M 132  38 169
sum(diag(TabelaFGNB))/sum(TabelaFGNB) # Acurácia
#> [1] 0.5221557
```
