# Problema K-centros

Trabalho de Algoritmos II que aborda a implementação e comparação de algoritmos
aproximativos para o problema de k-centros, útil em agrupamento de dados em
aprendizado de máquina. O objetivo é comparar a demanda computacional e a
qualidade das soluções oferecidas pelos algoritmos.

## Algoritmos implementados

Ambos são 2-aproximações conhecidas para o problema de k-centros:

- **Gonzalez (1985)** — `gonzalez_k_center`: greedy *farthest-first traversal*.
  Parte de um centro aleatório e, a cada passo, adiciona o ponto mais distante
  do conjunto de centros já escolhidos. Rápido, mas não determinístico.
- **Hochbaum & Shmoys (1985)** — `hochbaum_shmoys_k_center`: busca binária
  sobre as distâncias par a par candidatas ao raio ótimo, verificando a cada
  passo se um conjunto dominante guloso cobre todos os pontos com até k
  centros. Mais caro computacionalmente (`O(n² log n)`), porém determinístico.

## Estrutura do projeto

```
src/alg2tp2/
  distance.py    # distância de Minkowski, matriz de distâncias, rótulos por centro mais próximo
  k_center.py     # os dois algoritmos de k-centros e cálculo do raio da solução
  data.py         # geração de dados sintéticos 2D (com rótulos verdadeiros)
  evaluation.py    # métricas de qualidade de clustering (silhueta, ARI)
  experiment.py    # execução repetida dos algoritmos e agregação dos resultados
  cli.py           # interface de linha de comando
tests/             # testes automatizados (pytest)
main.py            # ponto de entrada de conveniência (chama alg2tp2.cli.main)
```

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate   # no Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Uso

```bash
python main.py --n-samples 700 --centers 5 --k 5 --runs 30
# ou, com o pacote instalado:
kcenter-experiment --algorithm gonzalez
```

Rode `python main.py --help` para ver todas as opções.

## Testes

```bash
pytest
```
