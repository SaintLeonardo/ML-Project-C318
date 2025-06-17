# Projeto de Previsão de Evasão Escolar no Brasil

## 1. Contexto e Objetivos

A evasão escolar é um dos maiores desafios enfrentados pelo sistema educacional brasileiro. Trata-se da saída precoce de estudantes da escola antes da conclusão da educação básica, o que impacta diretamente o desenvolvimento social e econômico do país. Com o avanço da tecnologia e o aumento na disponibilidade de dados educacionais, torna-se viável aplicar técnicas de Ciência de Dados e Aprendizado de Máquina para entender padrões e prever comportamentos de evasão.

Este projeto tem como objetivo analisar os dados do Censo Escolar e dados agregados de taxa de abandono e renda média, a fim de:

- Compreender os fatores que influenciam a evasão escolar.
- Identificar padrões por região, dependência administrativa e outros fatores institucionais.
- Aplicar algoritmos de Machine Learning para prever taxas de evasão escolar.

## 2. Perguntas de Negócio e Enquadramento

- Quais regiões apresentam maiores taxas de evasão escolar?
- A renda média tem relação com a evasão?
- As condições físicas das escolas influenciam na permanência dos alunos?
- É possível prever a taxa de evasão de uma escola com base em suas características estruturais e regionais?

### Tipo de Problema de ML

Este é um problema de **Regressão Supervisionada**, pois buscamos prever o valor da taxa de abandono (variável contínua) a partir de variáveis como:
- Localização geográfica (UF)
- Infraestrutura (salas, banheiros, acesso à internet, etc.)
- Dependência administrativa
- Renda média da população da região
