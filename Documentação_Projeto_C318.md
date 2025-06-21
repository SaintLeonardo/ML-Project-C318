# Alunos:
Felipe Siqueira Mohallem Cellet
Leonardo Santos Pereira


# Documentação do Projeto: Predição da Taxa de Abandono Escolar

Este documento descreve detalhadamente cada célula de código utilizada no projeto de predição da taxa de abandono escolar com base em renda média e outros fatores.

---

## Célula 1 - Importação de bibliotecas

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

Importação das bibliotecas utilizadas para análise de dados, visualizações gráficas e aplicação de modelos de regressão.

---

## Célula 2 - Carregamento dos dados

```python
df = pd.read_csv("AbandonoEscolar_RendaMedia_2013_2023.csv")
df = df.dropna()
```

Leitura do arquivo de dados e remoção de registros com valores ausentes.

---

## Célula 3 - Análise exploratória inicial

```python
print(df.head())
print(df.info())
print(df.describe())
```

Exibição de amostras iniciais dos dados, informações sobre tipos e estrutura das colunas e estatísticas descritivas.

---

## Célula 4 - Gráfico de dispersão entre renda e taxa de abandono

```python
sns.scatterplot(data=df, x="Renda_Media", y="Taxa_Abandono", hue="Dependencia_Administrativa")
plt.title("Renda Média x Taxa de Abandono")
plt.grid(True)
plt.show()
```

Representação gráfica da relação entre renda média e taxa de abandono escolar, categorizada por tipo de rede de ensino.

---

## Célula 5 - Boxplot da taxa de abandono por rede

```python
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x="Dependencia_Administrativa", y="Taxa_Abandono")
plt.title("Taxa de Abandono por Tipo de Rede")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
```

Comparação das distribuições de taxa de abandono entre diferentes tipos de rede administrativa de ensino.

---

## Célula 6 - Evolução da taxa de abandono ao longo do tempo

```python
plt.figure(figsize=(10,5))
sns.lineplot(data=df.groupby("Ano")["Taxa_Abandono"].mean().reset_index(), x="Ano", y="Taxa_Abandono")
plt.title("Evolução da Taxa de Abandono Escolar (Média Brasil)")
plt.grid(True)
plt.show()
```

Visualização da evolução temporal da taxa média de abandono escolar no Brasil entre 2013 e 2023.

---

## Célula 7 - Pré-processamento dos dados

```python
df_encoded = pd.get_dummies(df, columns=["Dependencia_Administrativa"], drop_first=True)
X = df_encoded[["Ano", "Renda_Media"] + [col for col in df_encoded.columns if "Dependencia_Administrativa_" in col]]
y = df_encoded["Taxa_Abandono"]
```

Transformação da variável categórica em variáveis binárias (one-hot encoding) e definição das variáveis preditoras e alvo.

---

## Célula 8 - Separação entre treino e teste

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Divisão dos dados em conjuntos de treino e teste, com 80% para treino e 20% para teste.

---

## Célula 9 - Definição dos modelos de regressão

```python
modelos = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}
```

Criação de um dicionário contendo os algoritmos de regressão a serem avaliados no projeto.

---

## Célula 10 - Avaliação dos modelos com RMSE

```python
resultados = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    resultados[nome] = rmse

plt.bar(resultados.keys(), resultados.values())
plt.ylabel("RMSE")
plt.title("Comparação de Modelos")
plt.show()
```

Treinamento e avaliação dos modelos utilizando RMSE como métrica. Os resultados são exibidos em um gráfico de barras.

---

## Célula 11 - Comparação gráfica dos valores reais e previstos

```python
plt.scatter(y_test, y_pred)
plt.xlabel("Valor Real")
plt.ylabel("Valor Predito")
plt.title("Comparação entre valores reais e preditos")
plt.grid(True)
plt.show()
```

Gráfico de dispersão entre os valores reais da taxa de abandono e os valores previstos por um dos modelos. Nota: a variável `y_pred` deve ser definida antes.

---

## Célula 12 - Validação cruzada (K-Fold)

```python
for nome, modelo in modelos.items():
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(modelo, X, y, cv=kfold, scoring="neg_root_mean_squared_error")
    print(f"Modelo: {nome}")
    print("RMSE médio:", abs(scores.mean()))
    print("RMSE desvio padrão:", scores.std())
```

Validação cruzada com 10 divisões para análise da variabilidade dos resultados dos modelos.

---

## Célula 13 - Predição e visualização com Random Forest

```python
modelo = RandomForestRegressor()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Real")
plt.ylabel("Previsto")
plt.title("Random Forest - Real vs Previsto")
plt.grid(True)
plt.show()
```

Treinamento do modelo Random Forest e visualização comparativa entre os valores reais e previstos.

---

## Célula 14 - Métricas finais do modelo

```python
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
```

Cálculo das métricas finais de desempenho do modelo: erro absoluto médio (MAE), erro quadrático médio (RMSE) e coeficiente de determinação (R²).
