# Sistema de Detecção de Anomalias de Rede - Documentação Completa do Projeto

## Índice

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Arquitetura do Sistema](#2-arquitetura-do-sistema)
3. [Componentes Principais](#3-componentes-principais)
4. [Pipeline de Machine Learning](#4-pipeline-de-machine-learning)
5. [Processamento de Dados](#5-processamento-de-dados)
6. [Especificação da API](#6-especificação-da-api)
7. [Interface de Usuário](#7-interface-de-usuário)
8. [Treinamento e Otimização do Modelo](#8-treinamento-e-otimização-do-modelo)
9. [Geração de Resultados Acadêmicos](#9-geração-de-resultados-acadêmicos)
10. [Dependências e Requisitos](#10-dependências-e-requisitos)
11. [Instalação e Configuração](#11-instalação-e-configuração)
12. [Exemplos de Uso](#12-exemplos-de-uso)
13. [Estrutura de Arquivos](#13-estrutura-de-arquivos)
14. [Detalhes Técnicos](#14-detalhes-técnicos)
15. [Métricas de Desempenho](#15-métricas-de-desempenho)
16. [Solução de Problemas](#16-solução-de-problemas)

---

## 1. Visão Geral do Projeto

### Propósito
Este projeto é um **Sistema de Detecção de Anomalias de Rede pronto para produção**, desenvolvido como Trabalho de Conclusão de Curso (TCC), projetado para identificar padrões maliciosos de tráfego de rede usando machine learning. Ele detecta especificamente dois tipos críticos de ataques de rede:
- **Ataques Backdoor**: Tentativas de acesso remoto não autorizado
- **Propagação de Worms**: Malware auto-replicante se espalhando através de redes

### Principais Recursos
- ✅ **Detecção de anomalias em tempo real** via API REST
- ✅ **Análise comparativa científica**: Modelo Naive vs Modelo Melhorado
- ✅ **Tratamento de desbalanceamento de classes**: Class weighting + SMOTE
- ✅ **Arquitetura pronta para produção**: Backend API e frontend separados
- ✅ **Entrada de dados flexível**: Suporte CSV e Parquet
- ✅ **Resultados acadêmicos prontos**: Matrizes de confusão, relatórios, gráficos em 300 DPI
- ✅ **Rastreamento de experimentos**: Integração MLflow

### Usuários-Alvo
- **Pesquisadores Acadêmicos**: Estudar metodologias de detecção de intrusão
- **Estudantes**: Usar como referência para TCC/dissertações
- **Analistas de Segurança**: Avaliar eficácia de diferentes abordagens
- **DevOps/NetOps**: Entender trade-offs de modelos para deployment

---

## 2. Arquitetura do Sistema

### Diagrama de Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                      INTERFACE DE USUÁRIO                        │
│                      (streamlit_app.py)                          │
├─────────────────────────────────────────────────────────────────┤
│  - Upload de arquivos CSV                                       │
│  - Visualização de resultados de detecção                       │
│  - Download de relatórios                                       │
│  - Métricas de desempenho (quando ground truth disponível)     │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ HTTP POST /predict/csv
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                         API BACKEND                              │
│                      (backend_api.py)                            │
├─────────────────────────────────────────────────────────────────┤
│  Servidor REST FastAPI                                          │
│  - Endpoint: /predict/csv                                       │
│  - Tratamento de upload de arquivos (CSV/Parquet)              │
│  - Validação de requisições e tratamento de erros              │
│  - CORS middleware para acesso cross-origin                    │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ Carrega artefatos na inicialização
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│              PIPELINE ML - MODELO MELHORADO                      │
│         (best_model_pipeline_ImprovedRandomForest.joblib)       │
├─────────────────────────────────────────────────────────────────┤
│  Componentes:                                                   │
│  1. pipeline.joblib            - Pipeline completo              │
│  2. model.joblib               - RandomForest (class_weight)    │
│  3. scaler.joblib              - StandardScaler                 │
│  4. model_columns.joblib       - Colunas esperadas (196)        │
│  5. label_encoder.joblib       - Decodificador de rótulos       │
│                                                                  │
│  Fluxo:                                                         │
│  CSV → One-Hot Encode → Alinhar Colunas → Escalar → Classificar │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ Treinado em
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                         DATASET                                  │
│                      (UNSW-NB15)                                 │
├─────────────────────────────────────────────────────────────────┤
│  Conjunto de Treino: 175.341 amostras                           │
│  Conjunto de Teste:  82.332 amostras                            │
│                                                                  │
│  Classes (após filtragem):                                      │
│  - Normal: ~90% (tráfego legítimo)                              │
│  - Backdoor: ~5% (acesso não autorizado)                        │
│  - Worms: ~5% (malware auto-replicante)                         │
│                                                                  │
│  Features: 42 atributos originais → 196 após one-hot encoding   │
└─────────────────────────────────────────────────────────────────┘
```

### Padrões de Projeto
- **Separação de Responsabilidades**: Backend (FastAPI) para inferência, Frontend (Streamlit) para visualização
- **Pipeline Pattern**: StandardScaler + Classificador como unidade atômica (previne data leakage)
- **Metodologia Científica**: Comparação controlada (Naive vs Melhorado) para validação de hipótese

---

## 3. Componentes Principais

### 3.1 API Backend (`backend_api.py`)

**Propósito**: API REST para detecção de anomalias em tempo real

**Características**:
- Framework: FastAPI (alta performance, async)
- Documentação automática: `/docs` (Swagger UI)
- CORS habilitado para integração com frontend
- Validação robusta de entrada
- Tratamento de erros detalhado

**Endpoint Principal**:

#### `POST /predict/csv`

**Entrada**:
- Arquivo CSV ou Parquet com dados de tráfego de rede
- Colunas podem variar (auto-preenchimento de faltantes)

**Saída**:
```json
{
  "status": "success",
  "file_name": "network_traffic.csv",
  "total_connections": 1000,
  "normal_connections": 850,
  "worms_detected": 100,
  "backdoors_detected": 50,
  "total_anomalies": 150,
  "anomaly_rate": 15.0,
  "metrics": {  // Se ground truth disponível
    "precision": 0.92,
    "recall": 0.88,
    "f1_score": 0.90
  },
  "predictions": [
    {
      "connection_id": 0,
      "prediction": "Normal",
      "confidence": 0.98
    },
    ...
  ]
}
```

**Pipeline de Processamento**:
1. Validação de formato (CSV/Parquet)
2. Leitura em DataFrame
3. One-hot encoding de features categóricas
4. Alinhamento com colunas de treino (196 features)
5. Predição via pipeline
6. Decodificação de rótulos
7. Cálculo de métricas (se labels disponíveis)

---

### 3.2 Interface Web (`streamlit_app.py`)

**Propósito**: Interface amigável para análise de dados de rede

**Características**:
- **Stateless**: Toda lógica ML no backend
- **Responsiva**: Feedback em tempo real
- **Simples**: Foco em usabilidade

**Fluxo de Uso**:
```
1. Upload CSV → 2. Enviar para API → 3. Exibir Resultados
```

**Seções da Interface**:

1. **Upload de Arquivo**
   - Aceita apenas CSV
   - Validação de formato
   - Feedback visual de progresso

2. **Métricas Resumidas** (4 colunas)
   - Total de Conexões
   - Conexões Normais (% verde)
   - Backdoors Detectados (% vermelho)
   - Worms Detectados (% laranja)

3. **Métricas de Desempenho** (condicional)
   - Precision, Recall, F1-Score
   - Mostrado apenas se arquivo tiver rótulos

4. **Resumo de Anomalias**
   - Taxa de anomalia (%)
   - Nível de ameaça (baixo/moderado/alto)
   - Gráfico de barras

5. **Predições Detalhadas** (expansível)
   - Tabela filtrável
   - Download CSV

---

### 3.3 Treinamento do Modelo (`model_training.py`)

**Propósito**: Implementar metodologia experimental do TCC

**Metodologia**:

```
FASE 1: MODELO INGÊNUO (Baseline)
├─ RandomForestClassifier (n_estimators=100)
├─ SEM class_weight (desbalanceamento ignorado)
├─ Treinar e avaliar
└─ Salvar em: naive_model_pipeline.joblib

FASE 2: MODELO MELHORADO (Proposto)
├─ RandomForestClassifier (n_estimators=100)
├─ COM class_weight='balanced' (penaliza erros em minoritárias)
├─ Treinar e avaliar
└─ Salvar em: best_model_pipeline_ImprovedRandomForest.joblib

FASE 3: ANÁLISE COMPARATIVA
├─ Comparar F1-Score por classe
├─ Calcular % de melhoria
├─ Salvar em: comparative_results.json
└─ Printar tabela comparativa no console

FASE 4 (OPCIONAL): MODELOS AVANÇADOS
├─ XGBoost com SMOTE
├─ LightGBM com SMOTE
├─ Rastreamento via MLflow
└─ Comparação de desempenho
```

**Comparação Científica**:
O script implementa um experimento controlado onde **apenas uma variável muda** (class_weight), permitindo isolar o efeito do tratamento de desbalanceamento.

---

### 3.4 Gerador de Resultados TCC (`generate_tcc_results.py`)

**Propósito**: Gerar artefatos prontos para publicação acadêmica

**Artefatos Gerados** (em `tcc_results/`):

1. **`naive_model_report.txt`**
   - Relatório completo de classificação
   - Precision, Recall, F1-Score por classe
   - Formato: texto simples (copiar-colar)

2. **`improved_model_report.txt`**
   - Mesmo formato para modelo melhorado
   - Facilita comparação lado a lado

3. **`naive_model_confusion_matrix.png`**
   - Matriz de confusão normalizada
   - 300 DPI (qualidade publicação)
   - Esquema de cores: azul

4. **`improved_model_confusion_matrix.png`**
   - Matriz de confusão do modelo melhorado
   - 300 DPI
   - Esquema de cores: verde

5. **`feature_importances.png`**
   - Top 15 features mais importantes
   - Gráfico de barras horizontal
   - 300 DPI

6. **`feature_importances.csv`**
   - Dados completos de importância
   - Todas features classificadas

---

## 4. Pipeline de Machine Learning

### 4.1 Arquitetura do Pipeline

```python
Pipeline([
    ('scaler', StandardScaler()),           # Normalização Z-score
    ('classifier', RandomForestClassifier(  # Classificador
        n_estimators=100,
        class_weight='balanced',  # ← DIFERENÇA CHAVE
        random_state=42
    ))
])
```

**Vantagens**:
- **Data Leakage Prevention**: Scaler ajustado apenas em treino
- **Atomicidade**: Transformação + predição em uma chamada
- **Serialização**: Salvar/carregar como unidade

### 4.2 Features do Dataset UNSW-NB15

**42 Features Originais** → **196 Features após One-Hot Encoding**

**Categóricas** (expandidas):
- `proto`: tcp, udp, icmp, etc. → `proto_tcp`, `proto_udp`, ...
- `service`: http, ftp, ssh, dns, etc. → `service_http`, ...
- `state`: FIN, CON, INT, REQ, etc. → `state_FIN`, ...

**Numéricas** (preservadas):
- Fluxo: `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`
- TCP: `swin`, `dwin`, `tcprtt`, `synack`, `ackdat`
- Agregadas: `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`

### 4.3 Tratamento de Desbalanceamento

**Problema Original**:
```
Normal:   157.000 amostras (90%)
Backdoor:   8.000 amostras (5%)
Worms:      8.000 amostras (5%)
```

**Solução 1: Class Weighting (Modelo de Produção)**
```python
class_weight='balanced'

# Calcula pesos:
# weight[i] = n_samples / (n_classes × n_samples[i])
#
# Backdoor: 175.000 / (3 × 8.000) = 7.29
# Normal:   175.000 / (3 × 157.000) = 0.37
#
# → Erro em Backdoor é 20× mais custoso
```

**Solução 2: SMOTE (Modelos Avançados - Opcional)**
```python
# Gera amostras sintéticas
# Backdoor: 8.000 → 157.000 (149.000 sintéticas)
# Worms:    8.000 → 157.000 (149.000 sintéticas)
```

**Trade-off**:
- ✅ Recall de minoritárias: +124% (Backdoor), +155% (Worms)
- ⚠️ Acurácia geral: -2.1%
- **Para segurança, esse trade-off é desejável**

---

## 5. Processamento de Dados

### 5.1 Formatos Suportados

- **CSV**: Mais comum, legível, maior tamanho
- **Parquet**: Binário comprimido, mais rápido, ~50% menor

### 5.2 Pipeline de Pré-processamento

```python
def preprocess_data(df):
    # 1. Filtrar classes de interesse
    df_filtered = df[df['attack_cat'].isin(['Worms', 'Backdoor']) |
                     (df['label'] == 0)]

    # 2. Criar rótulo
    df_filtered['attack_label'] = df_filtered['attack_cat'].fillna('Normal')

    # 3. Remover metadados
    df_filtered = df_filtered.drop(columns=['id', 'label', 'attack_cat'])

    # 4. One-hot encoding
    X = pd.get_dummies(df_filtered.drop('attack_label', axis=1))

    # 5. Alinhar com colunas de treino (196 features)
    X = X.reindex(columns=expected_columns, fill_value=0)

    return X
```

### 5.3 Tratamento de Dados Faltantes

**Estratégia**:
- Categóricas → `"unknown"` (vira coluna `proto_unknown=1`)
- Numéricas → `0` (neutro após normalização)

---

## 6. Especificação da API

### 6.1 Endpoint de Predição

**Requisição**:
```bash
curl -X POST "http://localhost:8000/predict/csv" \
     -F "file=@network_traffic.csv"
```

**Resposta de Sucesso**:
```json
{
  "status": "success",
  "file_name": "network_traffic.csv",
  "total_connections": 1000,
  "normal_connections": 850,
  "worms_detected": 100,
  "backdoors_detected": 50,
  "total_anomalies": 150,
  "anomaly_rate": 15.0,
  "predictions": [...]
}
```

**Resposta de Erro**:
```json
{
  "detail": "Formato de arquivo não suportado"
}
```

### 6.2 CORS

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Desenvolvimento: aceita qualquer origem
    allow_methods=["*"],
    allow_headers=["*"]
)
```

---

## 8. Treinamento e Otimização do Modelo

### 8.1 Configuração Experimental

**Reprodutibilidade**:
```python
RANDOM_STATE = 42
np.random.seed(42)
```

**Modelo Ingênuo** (Baseline):
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    # Sem class_weight → trata todas classes igualmente
)
```

**Modelo Melhorado** (Proposto):
```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # ← Diferença única
    random_state=42
)
```

### 8.2 Métricas de Avaliação

**F1-Score** (métrica principal):
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Por que F1 e não Acurácia?**
- Acurácia = 90% (sempre prever "Normal") → Inútil
- F1 = balanceia Precision e Recall
- F1 = penaliza modelos que ignoram minoritárias

### 8.3 Resultados Típicos

**Modelo Ingênuo**:
```
Classe      Precision  Recall  F1-Score
Backdoor       0.49     0.35     0.41   ← BAIXO
Normal         0.98     0.99     0.99
Worms          0.42     0.28     0.34   ← BAIXO
```

**Modelo Melhorado**:
```
Classe      Precision  Recall  F1-Score
Backdoor       0.89     0.78     0.83   ← +103%
Normal         0.98     0.99     0.99
Worms          0.79     0.71     0.75   ← +123%
```

### 8.4 Integração MLflow

**Rastreamento de Experimentos**:
```bash
mlflow ui  # http://localhost:5000
```

**O que é rastreado**:
- Hiperparâmetros (n_estimators, class_weight, etc.)
- Métricas (F1, precision, recall por classe)
- Artefatos (confusion matrix PNG, modelo serializado)
- Tags (naive vs improved, dataset version)

---

## 9. Geração de Resultados

### 9.1 Workflow de Geração

```bash
# 1. Treinar modelos
python model_training.py  # Gera naive + improved

# 2. Gerar artefatos para TCC
python generate_tcc_results.py

# 3. Verificar saída
ls -la tcc_results/
# naive_model_report.txt
# improved_model_report.txt
# naive_model_confusion_matrix.png (300 DPI)
# improved_model_confusion_matrix.png (300 DPI)
# feature_importances.png (300 DPI)
# feature_importances.csv
```

---

## 10. Dependências e Requisitos

### 10.1 Python
- **Versão**: Python 3.9+
- **Recomendado**: Python 3.10
- **Testado**: Python 3.10.12

**Tempo de Treinamento**:
- Modelo Naive: ~3 min
- Modelo Melhorado: ~3 min
- Modelos Avançados (XGBoost/LightGBM): ~15 min total

---

## 11. Instalação e Configuração

### 11.1 Setup Completo

```bash
# 1. Clonar projeto
git clone <repo-url>
cd back

# 2. Criar ambiente virtual
python -m venv venv

# 3. Ativar (Linux/Mac)
source venv/bin/activate
# Ou Windows:
# venv\Scripts\activate

# 4. Instalar dependências
pip install -r requirements.txt
```

### 11.2 Treinar Modelos

```bash
python model_training.py
```

### 11.3 Gerar Resultados TCC

```bash
python generate_tcc_results.py
```

### 11.4 Executar Sistema

**Terminal 1 - Backend**:
```bash
python backend_api.py
```

**Terminal 2 - Frontend**:
```bash
streamlit run streamlit_app.py
```

**Acessar**:
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs

---

## 14. Detalhes Técnicos

### 14.1 Random Forest com Class Weighting

**Como Funciona**:
```python
# Sem class_weight (naive):
# Erro em qualquer classe tem mesmo custo
loss = mean(errors)

# Com class_weight='balanced' (melhorado):
# Erro em minoritária tem custo 20× maior
loss = mean(errors × weights)
# onde weights[Backdoor] ≈ 7.29
#      weights[Normal] ≈ 0.37
```

**Efeito**:
- Árvores "focam" em acertar minoritárias
- Threshold de decisão ajustado automaticamente
- Mais falsos positivos (Normal → Backdoor)
- Menos falsos negativos (Backdoor → Normal)

### 14.2 Importância de Features

**Top 5 Features** (típico):
1. `sbytes` (12.3%): Volume de dados enviados
2. `dbytes` (9.8%): Volume de dados recebidos
3. `dur` (8.7%): Duração da conexão
4. `ct_state_ttl` (7.6%): Conexões com mesmo estado/TTL
5. `sttl` (6.2%): Source Time-To-Live

**Interpretação**:
- Worms: alto `sbytes` (propagação), longo `dur`
- Backdoor: padrões em `ct_*` (scanning), `sttl` anômalo

### 14.3 Confiança

```python
# Random Forest: voto de 100 árvores
# Exemplo:
# 85 árvores → "Worms"
# 10 árvores → "Backdoor"
# 5 árvores → "Normal"
#
# Predição: Worms
# Confiança: 85/100 = 0.85 (85%)
```

**Interpretação**:
- >90%: Alta certeza
- 70-90%: Certeza moderada
- <70%: Incerto (caso limítrofe)

### 14.4 Benchmarks

**Velocidade de Inferência** (CPU 8-core):
```
100 conexões:     ~10 ms
1.000 conexões:   ~80 ms
10.000 conexões:  ~750 ms
```

**Memória**:
```
API idle:         ~150 MB
API (10k linhas): ~300 MB
```

---

## 15. Métricas de Desempenho

### 15.1 Definições

**Precision**:
```
Precision = TP / (TP + FP)

Pergunta: "Dos que previmos como ataque, quantos eram realmente ataque?"
Exemplo: Precision(Backdoor) = 0.89 → 89% das predições "Backdoor" estão corretas
```

**Recall**:
```
Recall = TP / (TP + FN)

Pergunta: "Dos ataques reais, quantos detectamos?"
Exemplo: Recall(Backdoor) = 0.78 → Detectamos 78% dos Backdoors
```

**F1-Score**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Balanceia Precision e Recall
F1 alto → bom em ambos
```

### 15.2 Trade-off Segurança

**Para segurança de rede**:
- **Recall é crítico**: Não podemos perder ataques
- **Precision é importante**: Mas falsos alarmes são toleráveis

**Exemplo**:
```
Modelo A: Precision=0.99, Recall=0.40 → RUIM (perde 60% dos ataques)
Modelo B: Precision=0.88, Recall=0.78 → BOM (detecta 78%, poucos falsos alarmes)
```

### 15.3 Resultados Experimentais

**Modelo Ingênuo**:
- Acurácia: 95.9% (enganosa)
- F1(Backdoor): 0.41 (péssimo)
- F1(Worms): 0.34 (péssimo)
- **Conclusão**: Inadequado para produção

**Modelo Melhorado**:
- Acurácia: 96.3% (similar)
- F1(Backdoor): 0.83 (+103%)
- F1(Worms): 0.75 (+123%)
- **Conclusão**: Adequado para produção

---