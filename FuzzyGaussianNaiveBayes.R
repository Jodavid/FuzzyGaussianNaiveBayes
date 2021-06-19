#' --------------------------------------------------
#'             Fuzzy Gaussian Naive Bayes
#'
#' data: 18.06.2021
#' version: 0.1
#' author: Jodavid Ferreira; Ronei Moraes
#' e-mails: jodavid@protonmail.com; ronei@de.ufpb.br
#' 
#' obs.: por questao de codificacao nao esta sendo
#' considerado os acentos nas palavras
#' --------------------------------------------------
#' Pacotes Necessarios
#' -------------------
library(parallel) # para funcao makeCluster
library(doSNOW) # para funcao registerDoSnow

#' -----------------------------------------------
#'      Classificador Fuzzy Gaussian Naive Bayes
#' -----------------------------------------------
# Funcao utilizando distribuicao Normal Fuzzy Triangular
FuzzyGaussianNaiveBayes <- function(train,test,cl,metd=1,cores=2){

  #metd -> Método de transformar o triangulo em escalar
  #1 -> Baricentro
  #2 - > Q do test de uniformidade (artigo: Directional Statistics and Shape analysis)
  # --------------------------------------------------------
  # Estimando os parâmetros das classes
  cols <- ncol(train) # Quantidade de variáveis
  dados <- train; # matriz de dados
  M <- cl; # Classes verdadeiras
  # --------------------------------------------------------
  # Encontrando Mu e Sigma para cada classe
  medias <- lapply(1:length(unique(M)), function(i) colMeans( subset( dados, M == unique(M)[i] ) ) )
  varian <- lapply(1:length(unique(M)), function(i) diag( diag( cov( subset( dados, M==unique(M)[i] ) ) ), (cols), (cols) ) )
  # --------------------------------------------------------
  # --------------------------------------------------------
  # Estimando parametros triangulares
  alpha <- seq(0.0001,1.1,0.1)
  # -------------------------------
  N <- nrow(dados) # Quantidade de observacoes
  # -------------------------------
  #  Parametros Media
  # ------------------
  Parameters_media <- lapply(1:length(medias),function(i){ # laco para grupos
    lapply(1:length(medias[[1]]),function(k){ # laco das dimensoes
      round(t(sapply(1:length(alpha),function(j){
        c(medias[[i]][k]-(qnorm(1-alpha[j]/2)*(sqrt(varian[[i]][k,k]/N))),
          medias[[i]][k]+(qnorm(1-alpha[j]/2)*(sqrt(varian[[i]][k,k]/N))))
      }))
      ,3)
    })
  })
  # -------------------------------
  # Parametros Variancia
  # ------------------
  Parameters_varian <- lapply(1:length(medias),function(i){ # laco para grupos
    lapply(1:length(medias[[1]]),function(k){ #laco para dimensoes
      round(t(sapply(1:length(alpha),function(j){
        beta=0.05 # previamente fixo
        lambda=alpha[j]
        # ------
        L <- (1-lambda)*qchisq(p =1-(beta/2),N-1) + (lambda*(N-1))
        R <- (1-lambda)*qchisq(p =beta/2,N-1) + (lambda*(N-1))
        # ------
        c( ((N-1)*varian[[i]][k,k])/L,
           ((N-1)*varian[[i]][k,k])/R)
        # ------
      }))
      ,3)
    })
  })
  # --------------------------------------------------------
  # Calculo dos triangulos para cada observação de test
  # soma dos Logs e calculo dos Baricentro
  # --------------
  N_test <- nrow(test)
  # --------------
  # Definindo quantos nucleos do CPU utilizar
  core <- makeCluster(cores)
  registerDoSNOW(core)
  # --------------
  # Inicio do laco
  R_M_obs <- foreach(h=1:N_test,.combine = rbind) %dopar% {
    # ------------
    x <- test[h,]
    # ------------
    triangulos_obs <-
      lapply(1:length(medias),function(i){ #laco para os grupos
        trian <- lapply(1:length(medias[[1]]),function(k){ #laco para as dimensoes
          t(sapply(1:length(alpha),function(j){
            # ------------
            a <- dnorm(x = as.numeric(x[k]),mean = as.numeric(Parameters_media[[i]][[k]][j,1]),sd = sqrt(as.numeric(Parameters_varian[[i]][[k]][j,1])))
            b <- dnorm(x = as.numeric(x[k]),mean = as.numeric(Parameters_media[[i]][[k]][j,2]),sd = sqrt(as.numeric(Parameters_varian[[i]][[k]][j,2])))
            # ------------
            c(min(a,b),max(a,b))
            # ------------
          }))
        })
        if(length(trian)>1){return(Reduce('+',trian))}else{return(trian)}
      })
    # ------------
    # Calculo Centro de Massa
    vec_trian <- lapply(1:length(unique(M)), function(i) c(triangulos_obs[[i]][1,1],triangulos_obs[[i]][11,1],triangulos_obs[[i]][1,2]))
    # --------------------------------------------------------
    # Transformando Vetor em Escalar
    # ------------
    R_M <- switch(metd,
                  # ------------
                  # Baricentro
                  "1"={
                    # ------------
                    sapply(1:length(unique(M)),function(i) vec_trian[[i]][2] * (  ( (vec_trian[[i]][2] - vec_trian[[i]][1])*(vec_trian[[i]][3] - vec_trian[[i]][2]) + 1 ) / 3  ) )
                    # ------------
                  },
                  "2"={
                    # ------------
                    # Usando distancia Q
                    sapply(1:length(unique(M)),function(i){
                      # ------------
                      # Inicio 3 valores
                      y <- vec_trian[[i]]
                      # ------------
                      # Obtendo o produto zz*
                      S = y%*%t(Conj(y)) # matriz k x k
                      # ------------
                      # obtendo os autovalores
                      l <- eigen(S)$values
                      # Calculando Q
                      Q <- 3*(l[1]-l[2])^2
                      # ------------
                      return(Q)
                    })
                    # ------------
                  })
    # --------------------------------------------------------
    R_M_class <- which.max(R_M)
    # -------------------------
    return(R_M_class)
  }

  # -------------------------
  stopCluster(core)
  # -------------------------
  resultado <- unique(M)[R_M_obs]
  # ---------
  return(as.factor(c(resultado)))
}
# Fim da função
# -------------------------
