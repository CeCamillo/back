# Network Anomaly Detection Dashboard

Dashboard interativo em Streamlit para detecção de anomalias em dados de tráfego de rede. Faça upload dos seus logs de rede e detecte em tempo real:

- 🟢 **Normal**: Tráfego legítimo
- 🔴 **Backdoor**: Tentativas de ataque backdoor
- 🟠 **Worms**: Atividade de propagação de worms

O sistema utiliza um modelo pré-treinado baseado no dataset [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) com algoritmos de machine learning (Random Forest, XGBoost, LightGBM) para classificação multi-classe.

> **Ideal para**: Análises de segurança de rede, demonstrações executivas, investigações de incidentes e monitoramento de tráfego.

## 📁 Estrutura do projeto

```text
├── anomaly_detector.py               # Dashboard principal para upload e detecção
├── model_training.py                 # Script de treinamento do modelo (setup único)
├── datasets/                         # Arquivos de treino UNSW-NB15 (para o treinamento)
├── best_model_pipeline_*.joblib      # Modelo pré-treinado (gerado após treinamento)
├── model_columns.joblib              # Colunas esperadas para inferência
├── label_encoder.joblib              # Mapeamento de classes
├── requirements.txt                  # Dependências Python
└── README.md                         # Este documento
```

## 🚀 Começando

### Pré-requisitos

- Python 3.9 ou superior
- Ambiente virtual Python (venv ou conda)

### 1. Instalação

```bash
git clone <url-do-repositorio>
cd back

# Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Setup do Modelo (Executar uma vez)

O dashboard precisa de um modelo pré-treinado. Execute o script de treinamento:

```bash
python model_training.py
```

Este script:
- Treina modelos usando o dataset UNSW-NB15 em `datasets/`
- Gera os artefatos necessários (`.joblib` files)
- Leva ~10-15 minutos para completar

**Nota**: Os arquivos de treino UNSW-NB15 já estão inclusos em `datasets/`. Se precisar re-baixar, visite o [site oficial](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

### 3. Rodar o Dashboard

```bash
streamlit run anomaly_detector.py
```

O dashboard abrirá em `http://localhost:8501`.

## 📊 Como Usar o Dashboard

### 1. Upload de Dados

- Clique em "Browse files" e selecione seu arquivo CSV ou Parquet
- Formatos suportados: CSV, Parquet
- Pode conter qualquer estrutura de dados de tráfego de rede

### 2. Mapeamento de Colunas

O sistema tentará mapear automaticamente suas colunas para o formato UNSW-NB15:

- **Mapeamento automático**: Detecta colunas com nomes similares
- **Ajuste manual**: Use os dropdowns para corrigir mapeamentos
- **Colunas ausentes**: Preenchidas automaticamente com valores padrão

**Principais colunas esperadas**:
- Estatísticas de fluxo: `dur`, `sbytes`, `dbytes`, `spkts`, `dpkts`
- Categorias: `proto`, `service`, `state`
- Recursos de conexão: `ct_srv_src`, `ct_dst_ltm`, etc.

### 3. Executar Detecção

Clique em **"🚀 Run Anomaly Detection"** para processar seus dados.

### 4. Visualizar Resultados

**Aba "Detection Results"**:
- Tabela com todas as detecções e scores de confiança
- Filtros por classe (Normal, Backdoor, Worms) e confiança mínima
- Botão de download para exportar resultados em CSV

**Aba "Network Insights"**:
- Taxa de anomalias e distribuição por classe
- Análise de protocolos e serviços
- Estatísticas de bytes e pacotes transferidos
- Gráficos adaptativos baseados nas colunas disponíveis

**Aba "Feature Importance"**:
- Top 20 features mais importantes para detecção
- Tabela completa de importâncias expandível

## 📦 Tecnologias Utilizadas

- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Processamento de Dados**: Pandas, NumPy, PyArrow
- **Visualização**: Streamlit, Matplotlib, Seaborn
- **Rastreamento**: MLflow (opcional, para desenvolvimento de modelos)

## ��️ Solução de Problemas

**Modelo não encontrado**:
```
⚠️ Pre-trained model not found!
```
→ Execute `python model_training.py` para gerar os artefatos

**Erro de mapeamento de colunas**:
- Verifique se suas colunas contêm dados numéricos válidos
- Use o mapeamento manual para corrigir auto-sugestões incorretas

**Dataset muito grande**:
- Considere amostrar seus dados antes do upload
- O sistema processa todo o dataset de uma vez

**Ambiente virtual**:
- Sempre ative o venv antes de executar: `source venv/bin/activate`
- Reinstale dependências se necessário: `pip install -r requirements.txt`

## 🎯 Casos de Uso

- **Análise forense**: Investigar logs de tráfego suspeito
- **Monitoramento**: Detectar anomalias em tempo real
- **Relatórios**: Gerar reports executivos com estatísticas de rede
- **Pesquisa**: Experimentar com diferentes datasets de segurança
- **Demonstrações**: Apresentar capacidades de ML em segurança de redes

## 🔬 Detalhes Técnicos

**Modelo**: Pipeline completo (StandardScaler + Classifier)
**Algoritmos**: Random Forest, XGBoost ou LightGBM (melhor modelo selecionado automaticamente)
**Classes detectadas**: 3 classes (Normal, Backdoor, Worms)
**Métricas**: F1-score ponderado, confiança por predição
**Formato de entrada**: Flexível com auto-mapeamento de colunas
