# Apresentação

  <div style="text-align: justify">Este projeto de ciência de dados foi desenvolvido para adquirir experiência quanto à utilização dos algoritmos de <em>machine learning</em> para resolução de problemas de negócios em empresas. Os conjuntos de dados foram disponibilizados no curso Fundamentos de <em>Machine Learning</em> ministrado pelo professor Meigarom na <a href = "https://comunidadeds.com/)">Comunidade DS</a>.</div><br>

# Objetivo Geral
>
>> <div style="text-align: justify">O objetivo deste projeto foi avaliar a performance de algoritmos de aprendizado supervisionado e não supervisionado na execução de tarefas de classificação, regressão e agrupamento (clusterização).</div>
>
>
>> ## Objetivos Específicos
>>
>> - <div style="text-align: justify">Avaliar a performance de algoritmos supervisionados (<em>Stochastic Gradient Descent Classifier</em> - SGD, <em>K Nearest Neighbor Classifier</em> - KNN, <em>Decision Tree Classifier</em>, <em>Random Forest Classifier</em> e <em>Logistic Regression</em>) em ensaio de classificação utilizando métricas diversas.</div><br>

>> - <div style="text-align: justify">Avaliar a performance de algoritmos supervisionados (regressão linear e regressão linear com regularizações (Lasso, Ridge e Elastic Net), regressão polinomial e regressão polinomial com regularizações (Lasso, Ridge e Elastic Net), <em>Decision Tree Regressor</em> e <em>Random Forest Regressor</em>) em ensaio de regressão utilizando métricas diversas.</div><br>

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
>> <div style="text-align: justify">As métricas acurácia, precisão, revocação (<em>recall</em>) e pontuação F1 (<em>F1 score</em>) foram utilizadas para avaliar a performance dos algoritmos supervisionados para classificação. Elas foram calculadas a partir da construção da matriz de confusão (tabela abaixo).
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
>><div style="text-align: justify">Além das métricas acima mencionadas, foi utilizada a área sob a curva característica de operação do recebedor (<em>area under the receiver operating characteristic curve </em> - ROC AUC) para comparação com a curva proveniente de algoritmo aleatório.</div><br>
>
>><div style="text-align: justify">As métricas coeficiente de determinação (<em>R</em><sup>2</sup>), erro quadrático médio (MSE), raiz quadrada do erro quadrático médio (RMSE), erro absoluto médio (MAE) e erro absoluto médio percentual (MAPE) foram utilizadas para avaliar a performance dos algoritmos supervisionados para regressão.</div><br>
>
>><div style="text-align: justify">As métricas coeficiente médio de silhueta e soma dos quadrados intra-custers (<em>Within-Cluster Sum of Squares</em> - WCSS) foram utilizadas para avaliar a performance dos algoritmos não supervisionados para clusterização.</div><br>
>
>> ## Avaliação comparativa das métricas
>>
>><div style="text-align: justify">Tabelas foram construídas para comparar as métricas após as previsões utilizando os conjuntos de treino, validação e teste. Gráficos de dispersão foram construídos para avaliar a variação das métricas precisão e revocação bem como a variação das métricas em função de determinados parâmetros. A curva ROC foi utilizada para avaliar todos os algoritmos supervisionados de classificação.</div><br>

# Resultados Principais
>>
>><div style="text-align: justify">(<strong>Ensaio de classificação com KNN</strong>) O limiar de decisão aproximdamente igual a 0.6 maximizou as métricas acurácia, precisão e revocação. A partir do limiar de 0.73, a precisão diminuiu rapidamente (seções 3.1.2 e 3.1.3 do notebook ensaio_classificacao.ipynb). O algoritmo otimizado com base na precisão (k = 4; seções 3.1.4) indicou métricas de performance mais baixas (erro de generalização mais alto) sobre as previsões de x_test conforme esperado (df2; seção 3.1.6).</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de classificação com SGD</strong>)O algoritmo SGD apresentou baixa performance em geral, provavelmente porque foi um algoritmo exclusivamente aleatório tal como indicado pela pontuação ROC AUC igual a 0.5 (seção 4).</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de classificação com Decision Tree Classifier</strong>) Em geral, o algoritmo Decision Tree apresentou alta performance e as métricas demonstraram a inadequação (valores iguais a 1; seção 5) de uso do conjunto de treinamento (utilizado no ajuste do modelo) para previsão de novas classes.</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de classificação com Random Forest Classifier</strong>) O algoritmo apresentou performance muito satisfatória (seção 6) e a curva ROC indicou distanciamento máximo da curva proveniente de classificações aleatórias (seção 11).</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de classificação com Logistic Regression</strong>) O algoritmo apresentou performances satisfatórias com curvas ROC distantes da curva de classificação aleatória (seção 7).</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de regressão</strong>) Os algoritmos de regressão linear e polinomial não apresentaram performances satisfatórias (seções 3 e 6 do notebook ensaio_regressao.ipynb) e as regularizações não melhoraram significativamente as métricas de performance (seção 7).</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de regressão com  <em>Decision Tree Regressor</em></strong>) O algoritmo não apresentou performance satisfatória mesmo com a otimização dos hiperparâmetros utilizando o GridSearch (seção 4). Neste ensaio, observaram-se erros mais baixos com max_depth igual a 6 (seção 4.1) e max_features igual a 8 (seção 4.3). A otimização de hiperparâmetros com GridSearch não melhorou significativamente a performance do algoritmo (seção 4.5).</div><br>
>>
>><div style="text-align: justify">(<strong>Ensaio de regressão com  <em>Random Forest Regressor</em></strong>) O algoritmo apresentou performance moderada (seção 5) e a otimização com Grid Search aumentou um pouco a métrica <em>R</em><sup>2</sup> (seção 5.1).</div><br>

>><div style="text-align: justify">(<strong>Ensaios de clusterização</strong>) Os algoritmos apresentaram coeficientes de silhuetas baixos mesmo após a otimização de hiperparâmetros com GridSearch (notebook ensaio_clusterizacao.ipynb)</div><br>

# Conclusão
>>
>><div style="text-align: justify"> Os algoritmos Random Forest Classifier e Random Forest Regressor apresentaram as melhores performances na execução das tarefas de classificação e regressão, respectivamente.</div><br>

# Perspectivas
>>
>><div style="text-align: justify"> Os resultados dos algoritmos paramétricos de regressão (regressão linear e polinomial com ou sem regularizações) indicaram a necessidade de processamento prévio dos dados antes do ajuste do modelos. Esse processamento deve considerar a multicolinearidade de variáveis independentes (por exemplo, as variáveis <em>loudness</em> e <em>energy</em>) (seção 8 do notebook ensaio_regressao.ipynb), a existência de valores extremos e a distribuição dos valores. Esse processamento será feito para aperfeiçoar os algoritmos de aprendizado de máquina.</div><br>


# Ferramentas utilizadas
>>
>> - <em>Python</em> (versão 3.9.21), <em>Scikit-learn</em> e algoritmos de aprendizado de máquina.
>> - WSL para gerenciamento de ambiente virtual.
>> - Git para versionamento dos arquivos e envio para o repositório Github.
>> - Jupyter notebook para ajuste dos modelos de aprendizado de máquina
>> - VS Code para construção do arquivo Readme.md

# Habilidades Desenvolvidas
>>
>> - Análise de dados.
>> - Utilização de algoritmos supervisionados de aprendizado de máquina
>> - Utilização de algoritmos não supervisionados de aprendizado de máquina
>> - Interpretação de métricas de performance dos algoritmos










