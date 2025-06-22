# 🎓 Predição da Taxa de Abandono Escolar com Machine Learning

Este projeto tem como objetivo analisar a relação entre a **renda média da população** e a **taxa de abandono escolar** no Brasil, utilizando técnicas de **Machine Learning** para modelagem preditiva e visualização de padrões relevantes.

---

## 👨‍💻 Alunos

- Felipe Siqueira Mohallem Cellet  
- Leonardo Santos Pereira  

---

## 📁 Estrutura do Projeto

- `notebook.ipynb` — Notebook principal com todas as etapas do projeto.
- `AbandonoEscolar_RendaMedia_2013_2023.csv` — Base de dados utilizada na análise.
- `README.md` — Este arquivo de descrição do projeto.
- `Documentação_Projeto_C318.md` — Arquivo contendo a explicação detalhada de cada célula do notebook e decisões técnicas do projeto.


---

## 🧠 Perguntas de Negócio

- Como a **renda média** da população se relaciona com a **taxa de abandono escolar**?
- É possível **prever a taxa de abandono** com base em renda, ano e rede de ensino?
- Quais **regiões ou redes** estão mais vulneráveis à evasão?

---

## 🔍 Etapas do Projeto

1. **Análise Exploratória (EDA)**
   - Visualizações com `seaborn` e `matplotlib`
   - Comparações por rede e evolução temporal

2. **Pré-processamento**
   - One-hot encoding para variáveis categóricas
   - Separação entre treino e teste (80/20)

3. **Modelagem**
   - Avaliação de 4 modelos:
     - Regressão Linear
     - Árvore de Decisão
     - Random Forest
     - Gradient Boosting
   - Métricas: RMSE, MAE e R²

4. **Validação Cruzada (K-Fold)**
   - Avaliação da estabilidade dos modelos
   - Random Forest teve o melhor desempenho médio

5. **Interpretação dos Resultados**
   - Importância das variáveis
   - Análise de outliers e casos de maior erro

---

## 📈 Resultado Final

- Modelo selecionado: **Random Forest**
- RMSE médio (cross-validation): `~3.10`
- Variáveis mais importantes: `Renda_Media`, tipo de rede e ano
- Insights relevantes para direcionar **políticas públicas educacionais**

---

## 💡 Conclusão

O projeto demonstrou que é possível prever a taxa de abandono escolar com boa precisão a partir de dados econômicos e institucionais. Modelos baseados em árvores, como o Random Forest, foram eficazes ao capturar padrões complexos nos dados e oferecer uma base analítica sólida para tomadas de decisão.

---

## 🚀 Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/projeto-abandono-escolar.git
