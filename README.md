# Credit Card Fraud Detection

Este notebook Jupyter apresenta uma análise completa de detecção de fraudes em transações de cartão de crédito utilizando o dataset clássico do Kaggle (284.807 transações com 0,17% de fraudes). O projeto compara quatro algoritmos de machine learning com foco em métricas operacionais relevantes para cenários reais de negócio.

## Context

A detecção de fraudes é uma proteção central para o desempenho financeiro, a continuidade operacional e a reputação das organizações modernas. Em setores como bancos, comércio eletrônico, telecomunicações, seguros e saúde, as perdas relacionadas a fraudes continuam aumentando. O relatório Top Fraud Trends Report 2025 da TransUnion constatou que, entre 1.200 líderes empresariais globais entrevistados, as empresas relataram perder, em média, 7,7% de sua receita anual com fraudes, o que mostra o quanto esse problema impacta as operações digitais rotineiras. Além das perdas financeiras, as empresas também enfrentam obrigações regulatórias em estruturas como PSD2, PCI DSS e KYC/AML, que exigem práticas sólidas de monitoramento, verificação confiável de identidade e controles de risco auditáveis.[1]

O problema se torna ainda mais difícil porque as táticas de fraude evoluem mais rápido do que as defesas tradicionais conseguem se adaptar. Regras fixas podem identificar irregularidades óbvias, mas muitas vezes são ineficazes contra esquemas sofisticados, coordenados e em constante evolução. Trata-se de um problema real e crescente: em 2024, os americanos perderam US$ 12,5 bilhões em fraudes, um aumento de 25% em relação ao ano anterior. A IA, o aprendizado de máquina e os modelos generativos estão transformando a prevenção a fraudes ao processarem grandes volumes de dados comportamentais, transacionais e contextuais para identificar padrões sutis, reduzir falsos positivos e responder rapidamente a novas ameaças. Uma detecção de fraudes eficaz não se resume a bloquear transações fraudulentas, mas também a proteger a confiança dos clientes, atender às exigências regulatórias e manter uma experiência fluida para os usuários legítimos.[1]

## Funcionalidades Implementadas

- Análise Exploratória de Dados completa com visualizações
- Benchmark sistemático de 4 algoritmos diferentes
- Tratamento adequado de classes altamente desbalanceadas
- Avaliação com métricas específicas para detecção de fraudes
- Análise de trade-off entre falsos positivos e falsos negativos

## Dataset

Dataset público do Kaggle contendo transações anônimas de cartões europeus:

```
Total de transações: 284.807
Fraudes: 492 (0,17%)
Features: Time, V1-V28 (transformação PCA), Amount, Class
Período coberto: 2 dias
```

## Modelos Comparados

| Modelo | ROC AUC | Recall | Precision | Falsos Positivos | Fraudes Perdidas |
|--------|---------|--------|-----------|------------------|------------------|
| ANN | 0.974 | 87,8% | 9,0% | 871 | 12 |
| XGBoost | 0.968 | 83,7% | 88,2% | 11 | 16 |
| Logistic Regression | 0.971 | 91,8% | 3,9% | 2.199 | 8 |
| Random Forest | 0.953 | 74,5% | 96,1% | 3 | 25 |

## Resultados Principais

**XGBoost foi selecionado como o modelo ideal** por apresentar o melhor equilíbrio operacional:

- Precision de 88,2% (88% dos alertas são fraudes reais)
- Apenas 11 falsos positivos em 56k transações de teste
- 83,7% de recall mantendo eficiência operacional
- Equilíbrio ideal entre detecção de fraudes e experiência do cliente

## Estrutura do Notebook

1. **Carregamento de dados** e imports necessários
2. **Análise inicial** (shape, info, describe, distribuição de classes)
3. **Visualizações EDA** (Amount, Time, boxplots por classe)
4. **Pré-processamento** (train/test split estratificado)
5. **Quatro modelos implementados**:
   - Logistic Regression (baseline)
   - Random Forest 
   - XGBoost (com scale_pos_weight)
   - Rede Neural Artificial profunda
6. **Avaliação completa** com classification_report e confusion_matrix
7. **Análise de negócio** focada em custos operacionais

## Dependências Técnicas

```bash
pandas
numpy
scikit-learn
xgboost
tensorflow/keras
seaborn
matplotlib
imbalanced-learn
```

## Como Reproduzir

1. Faça download do arquivo `creditcard.csv` do Kaggle
2. Coloque o arquivo na mesma pasta do notebook
3. Execute as células sequencialmente
4. Os modelos são avaliados automaticamente no conjunto de teste

## Considerações de Negócio

A análise demonstra que **falsos positivos têm custo operacional muito superior** aos falsos negativos em cenários reais:

- 871 falsos positivos (ANN) = sobrecarga inviável da equipe antifraude
- 11 falsos positivos (XGBoost) = operação sustentável
- Trade-off entre recall e precision é crítico para escalabilidade

## Dashboard interativo com Streamlit

Aplicativo feito com a biblioteca Streamlit (arquivo: aplicativo_streamlit.py). Para visualizar o aplicativo, instale a biblioteca Streamlit e suas dependências executando o comando no terminal.

"pip install streamlit"

Em seguida, navegue até a pasta do projeto e digite

"streamlit run aplicativo_streamlit.py". 

O navegador abrirá automaticamente na página do aplicativo. Para que o aplicativo funcione corretamente, certifique-se de que os arquivos "creditcard.csv" (dataset original com 284.807 transações) e "xgb_fraude.pkl" (modelo XGBoost treinado) estão localizados na mesma pasta do arquivo "aplicativo_streamlit.py". O aplicativo possui duas páginas principais: a primeira página exibe um dashboard interativo com cinco visualizações diferentes obtidas através da análise realizada no arquivo "fraud_detection.ipynb", incluindo histogramas de distribuição de valores, timeline temporal das transações, boxplots comparativos em escala logarítmica e proporção de classes.

Na segunda página intitulada "Inserir Nova Transação", é possível realizar a classificação de novas transações através do upload de um arquivo CSV contendo dados no mesmo formato do dataset original (colunas: Time, V1-V28 e Amount). Para fins de teste e praticidade, utilize o arquivo "dados_teste_app.csv" já disponibilizado no projeto. Após o upload do arquivo e clicar no botão "Classificar Transações", o aplicativo processará os dados utilizando o modelo XGBoost e gerará um novo arquivo CSV contendo todas as colunas originais mais duas colunas adicionais: "Classe Predita" (indicando se a transação é fraude ou não) e "Probabilidade Fraude (%)" (percentual de probabilidade de ser uma fraude). Este arquivo resultante pode ser baixado diretamente através do botão de download disponibilizado na interface.  

## Próximos Passos Sugeridos

- Implementação de API REST para produção
- Monitoramento de performance em produção
- Testes A/B com diferentes limiares de decisão
- Integração com sistemas de pagamento em tempo real

## Licença

MIT License - sinta-se à vontade para usar e modificar o código.

---

Projeto desenvolvido como estudo prático de machine learning aplicado a detecção de fraudes financeiras, com foco especial nas implicações operacionais de cada modelo.

Fonte de dados: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Referências: [1] https://wandb.ai/mostafaibrahim17/ml-articles/reports/Fraud-detection-techniques-explained-with-tutorial---VmlldzoxNTI5ODg3MQ. Acessado em: 24/02/2026.
