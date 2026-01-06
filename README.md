# Credit Card Fraud Detection

Este notebook Jupyter apresenta uma análise completa de detecção de fraudes em transações de cartão de crédito utilizando o dataset clássico do Kaggle (284.807 transações com 0,17% de fraudes). O projeto compara quatro algoritmos de machine learning com foco em métricas operacionais relevantes para cenários reais de negócio.

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

## Próximos Passos Sugeridos

- Implementação de API REST para produção
- Dashboard interativo com Streamlit
- Monitoramento de performance em produção
- Testes A/B com diferentes limiares de decisão
- Integração com sistemas de pagamento em tempo real

## Licença

MIT License - sinta-se à vontade para usar e modificar o código.

---

Projeto desenvolvido como estudo prático de machine learning aplicado a detecção de fraudes financeiras, com foco especial nas implicações operacionais de cada modelo.
