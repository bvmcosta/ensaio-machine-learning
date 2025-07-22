# Apresentação

  <div style="text-align: justify">Este projeto de ciência de dados foi desenvolvido para adquirir experiência quanto à utilização dos algoritmos de <em>machine learning</em> para resolução de problemas. Os conjuntos de dados foram disponibilizados no curso Fundamentos de <em>Machine Learning</em> ministrado pelo professor Meigarom na [Comunidade DS](https://comunidadeds.com/).</div><br>

# Objetivo Geral
>
>> <div style="text-align: justify">O objetivo desse projeto foi avaliar a performance de algoritmos de aprendizado supervisionado e não supervisionado na execução de tarefas de classificação, regressão e agrupamento (clusterização).</div>
>
>
>> ## Objetivos Específicos
>>
>> - <div style="text-align: justify">Avaliar a performance de algoritmos supervisionados (<em>Stochastic Gradient Descent Classifier</em> - SGD, <em>K Nearest Neighbor Classifier</em> - KNN, <em>Decision Tree Classifier</em>, <em>Random Forest Classifier</em> e <em>Logistic Regression</em>) em ensaio de classificação utilizando métricas diversas.</div><br>
>> - <div style="text-align: justify">Avaliar a performance de algoritmos supervisionados (<em>Linear Regression</em>, <em>Linear Regression Lasso</em>, <em>Linear Regression Ridge</em>, <em>Linear Regression Elastic Net</em>, <em>Polionomial Regression</em>, <em>Polinomial Regression Lasso</em>, <em>Polinomial Regression Ridge</em>, <em>Polinomial Regression Elastic Net</em>, <em>Decision Tree Regressor</em>, <em>Random Forest Regressor</em>) em ensaio de regressão utilizando métricas diversas.</div><br>
>> -  <div style="text-align: justify">Avaliar a performance de algoritmos não supervisionados (<em>K Means</em>, <em>Affinity Propagation</em>) em ensaio de clusterização utilizando métricas diversas.</div><br>

# Metodologia
>
>> ## <em>Datasets</em>
>>
>> <div style="text-align: justify">Três conjuntos de dados foram utilizados para condução dos ensaios de classificação (diretório datasets/classificacao/), regressão (diretório datasets/regressao/) e clusterização (diretório datasets/clusterizacao/). Conjuntos de dados diferentes foram utilizados para ajuste (X_training.csv e y_training.csv) e avaliação dos algoritmos supervisionados (X_validation, X_test, y_validation e y_test) e um conjunto de dados para os algoritmos não supervisionados (X_dataset.csv). Após ajuste dos modelos, as predições foram feitas utilizando os conjuntos de treinamento, validação e teste para avaliar as generalizações dos modelos.</div><br>
>
>> 
>> ## Métricas para avaliação da performance dos modelos
>>
>> <div style="text-align: justify">As métricas acurácia, precisão, revocação (<em>recall</em>) e pontuação F1 (<em>F1 score</em>) foram utilizadas para avaliar a performance dos algoritmos supervisionados para classificação. Elas foram calculadas a partir da construção da matriz de confusão (tabela abaixo). Além dessas métricas, foi utilizada a área sob a curva característica de operação do recebedor (<em>area under the receiver operating characteristic curve </em> - ROC AUC).</div><br>
>
>>
>>|                                 | <strong>Previsão (Classe 0)</strong> | <strong>Previsão (Classe 1)</strong> |
>>|---                              |---                                   |                                   ---|
>>| <strong>Real (Classe 0)</strong>| TN                                   |                                   FP |
>>| <strong>Real (Classe 1)</strong>| FN                                   |                                   TP |
>><br>
>>
>><div style="text-align: justify">onde, TP (<em>True Positive</em>) é a quantidade de instâncias corretamente classificadas como Classe 1; TN (<em>True Negative</em>) é a quantidade de instâncias corretamente classificadas como Classe 0; FN (<em>False Negative</em>) e FP (<em>False Positive</em>) são as quantidades de instâncias incorretamente classificadas como Classe 0 e Classe 1, respectivamente.</div><br>
>
>><div style="text-align: justify">As métricas coeficiente de determinação (R2), erro quadrático médio (MSE), raiz quadrada do erro quadrático médio (RMSE), erro absoluto médio (MAE) e erro absoluto médio percentual (MAPE) foram utilizadas para avaliar a performance dos algoritmos supervisionados para regressão</div><br>
>
>><div style="text-align: justify">As métricas coeficiente médio de silhueta e soma dos quadrados intra-custers (WCSS) foram utilizadas para avaliar a performance dos algoritmos não supervisionados para clusterização.</div><br>
>
>> ## Avaliação comparativa das métricas
>>
>><div style="text-align: justify">Tabelas foram construídas para comparar as métricas após as previsões utilizando os conjuntos de treino, validação e teste. Gráficos de dispersão foram construídos para avaliar a variação das métricas precisão e revocação bem como a variação das métricas em função de determinados parâmetros. A curva ROC foi utilizada para avaliar todos os algoritmos supervisionados de classificação.</div><br>

# Resultados Principais
>>
>><div style="text-align: justify">(<strong>Ensaio de classificação com KNN</strong>) O limiar de decisão aproximdamente igual a 0.6 maximizou as métricas acurácia, precisão e revocação. A partir do limiar de 0.73, a precisão diminuiu rapidamente (seções 3.1.2 e 3.1.3 do notebook ensaio_classificacao.ipynb). O algoritmo otimizado (k = 4; seções 3.1.4 e 3.1.5) indicou métricas de performance mais baixas sobre as previsões de x_test conforme esperado (df2; seção 3.1.6).</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de classificação com SGD</strong>)O algoritmo SGD apresentou baixa performance em geral, provavelmente porque foi um algoritmo exclusivamente aleatório tal como indicado pela pontuação ROC AUC igual a 0.5 (seção 4 do notebook ensaio_classificacao).</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de classificação com Decision Tree</strong>) Em geral, o algoritmo Decision Tree apresentou alta performance e as métricas demonstraram a inadequação (valores iguais a 1; seção 5) de uso do conjunto de treinamento (utilizado no ajuste do modelo) para previsão de novas classes. A otimização do modelo com a GridSearch indicou a seguinte combinação de hiperparâmetros: (seção 5.2)</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de classificação com Random Forest</strong>) O algoritmo apresentou performances muito satisfatórias (seção 6) e a curva ROC indicou distanciamento da curva proveniente de classificações aleatórias (seção 6.2). A otimização do modelo com a GridSearch indicou a seguinte combinação de hiperparâmetros: </div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de classificação com Logistic Regression</strong>) O algoritmo apresentou performances satisfatórias com curvas ROC distantes da curva de classificação aleatória (seção 7).</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de regressão</strong>)</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de clusterização</strong>)</div><br>

# Conclusões
>>
>><div style="text-align: justify">(<strong>Ensaio de classificação</strong>) O algoritmo Random Forest Classifier apresentou as métricas mais satisfatórias e curva ROC que mais se distanciou da curva das classificações aleatórias (seções 10 e 11), seguido dos algoritmos Decision Tree Classifier, Logistic Regression, KNN e SGD (seções 10 e 11).</div><br>
# Ferramentas utilizadas
# Habilidades Desenvolvidas










