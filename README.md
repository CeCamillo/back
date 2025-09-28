# UNSW-NB15 Anomaly Detection Toolkit

Este repositório reúne tudo o que você precisa para treinar, avaliar e apresentar modelos de detecção de anomalias no dataset [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset). O pacote inclui:

- Um pipeline de treinamento automatizado com otimização de hiperparâmetros (Randomized Search) para Random Forest, XGBoost e LightGBM.
- Rastreamento completo de experimentos via MLflow, com geração de métricas comparativas e artefatos (matrizes de confusão, CSVs, modelos em formato `joblib`).
- Um dashboard em Streamlit que consome os artefatos gerados e oferece visão executiva das métricas, exploração dos dados e um sandbox de inferência.

> **Use este projeto como ponto de partida** para pesquisas em segurança de redes, demonstrações executivas ou integração futura com APIs e pipelines de MLOps.

## 📁 Estrutura do projeto

```text
├── datasets/                         # Arquivos parquet de treino e teste (UNSW-NB15)
├── mlruns/                           # Diretório padrão do MLflow com experimentos versionados
├── model_training.py                 # Script principal de treinamento e logging dos modelos
├── model_training.ipynb              # Notebook exploratório opcional
├── streamlit_training_dashboard.py   # Dashboard interativo com Streamlit
├── model_comparison_metrics.csv      # Métricas agregadas (CSV gerado após treinamento)
├── per_class_f1_long.csv             # Métricas de F1 por classe para cada modelo
├── confusion_matrix_*.png            # Matrizes de confusão renderizadas para cada algoritmo
├── best_model_pipeline_*.joblib      # Pipeline completo vencedor (pré-processamento + modelo)
├── model_columns.joblib              # Colunas esperadas na etapa de inferência
├── label_encoder.joblib              # LabelEncoder com o mapeamento das classes
├── requirements.txt                  # Dependências da aplicação
└── README.md                         # Este documento
```

## 🚀 Começando

### Pré-requisitos

- Python 3.9 ou superior
- [pip](https://pip.pypa.io/) e [venv](https://docs.python.org/3/library/venv.html) (ou outra solução de ambiente virtual)
- ~12 GB de espaço em disco para armazenar datasets, artefatos e experimentos do MLflow

### Configuração rápida

```bash
git clone <url-do-repositorio>
cd back

python -m venv venv
source venv/bin/activate          # Em Windows use: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Os arquivos `.parquet` do UNSW-NB15 devem ficar dentro de `datasets/`. Caso não possua os arquivos, faça o download no site oficial e renomeie para:

- `UNSW_NB15_training-set.parquet`
- `UNSW_NB15_testing-set.parquet`

## 🧠 Treinando os modelos

O script `model_training.py` realiza todo o pipeline:

1. Carrega e pré-processa os datasets (filtro das classes de interesse, limpeza de colunas e one-hot encoding).
2. Ajusta e avalia três algoritmos (Random Forest, XGBoost, LightGBM) com `RandomizedSearchCV` (F1 ponderado como métrica-alvo).
3. Loga métricas, parâmetros e matrizes de confusão no MLflow.
4. Persiste o melhor pipeline completo (pré-processamento + modelo), mais o mapeamento de colunas e o label encoder.

Execute o treinamento com:

```bash
python model_training.py
```

### Artefatos gerados

- `best_model_pipeline_<ALG>.joblib`: Pipeline com pré-processamento e o modelo vencedor.
- `model_columns.joblib`: Index de colunas usado no one-hot encoding para alinhamento durante inferência.
- `label_encoder.joblib`: Encoder com as classes (`Normal`, `Backdoor`, `Worms`).
- `model_comparison_metrics.csv`: Tabela com Weighted F1, Macro F1, precisões e recalls.
- `per_class_f1_long.csv`: Tabela “long” com F1 por classe (ideal para gráficos comparativos).
- `confusion_matrix_<ALG>.png`: Matrizes de confusão normalizadas.
- `macro_f1_models.png` / `weighted_f1_models.png`: Gráficos opcionais gerados manualmente ou via notebook.

### Rastreamento com MLflow

Todos os experimentos ficam em `mlruns/`. Para uma interface visual, inicie o servidor local do MLflow:

```bash
mlflow ui --backend-store-uri mlruns --port 5001
```

Abra `http://127.0.0.1:5001` para inspecionar métricas, parâmetros, artefatos e comparar execuções.

## 📊 Dashboard em Streamlit

O arquivo `streamlit_training_dashboard.py` transforma os artefatos gerados em uma interface navegável com cinco seções:

- **Overview**: Destaque do melhor modelo, métricas agregadas e links para gráficos resumidos.
- **Metrics**: Gráficos de barras com Weighted F1, Macro F1 e comparação das classes.
- **Confusion Matrices**: Visualização lado a lado das matrizes de confusão.
- **Dataset Explorer**: Amostragens interativas do dataset (distribuições, frequências e estatísticas descritivas).
- **Inference Sandbox**: Seção para comparar predição vs. rótulo real usando o pipeline salvo.

Para rodar o dashboard:

```bash
streamlit run streamlit_training_dashboard.py
```

Abrirá uma URL no terminal (`http://localhost:8501` por padrão). Certifique-se de que os artefatos listados acima estejam presentes; o app exibirá mensagens amigáveis se algum arquivo estiver ausente.

## 📦 Dependências principais

As bibliotecas utilizadas estão descritas em `requirements.txt`. Destaques:

- **Pandas / NumPy / PyArrow**: Manipulação de dados tabulares e arquivos parquet.
- **scikit-learn**: Pré-processamento, pipelines, tuning de hiperparâmetros e métricas.
- **XGBoost / LightGBM**: Modelos de gradient boosting otimizados para classificação.
- **Seaborn / Matplotlib**: Visualizações estatísticas e gráficos customizados.
- **MLflow**: Rastreamento de experimentos, versionamento de modelos e logging de artefatos.
- **Streamlit**: Construção rápida do dashboard interativo.

## 🛠️ Dicas e solução de problemas

- **Memória insuficiente**: O UNSW-NB15 é volumoso. Ajuste o parâmetro `sample_size` no dashboard ou reduza `n_iter`/`cv` em `RandomizedSearchCV` durante experimentos locais.
- **Artefatos ausentes**: Se o dashboard exibir alertas sobre arquivos inexistentes, execute `python model_training.py` novamente para gerar tudo.
- **Falhas ao iniciar o MLflow UI**: Verifique se nenhuma outra aplicação está utilizando a porta informada (ex.: `lsof -i :5001`).
- **Ambiente virtual quebrado**: Recrie o diretório `venv/` e reinstale as dependências com `pip install -r requirements.txt`.

## ✅ Roadmap sugerido

- [ ] Disponibilizar uma API FastAPI para servir o pipeline vencedor.
- [ ] Automatizar o pipeline com agendamento (Airflow, Prefect ou GitHub Actions).
- [ ] Adicionar testes unitários para a etapa de pré-processamento e para o dashboard.
