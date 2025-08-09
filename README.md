# API de Detecção de Anomalias de Rede

Este projeto fornece uma API de alto desempenho para detectar anomalias em redes de computadores usando um modelo de machine learning. A API foi construída com FastAPI e serve um modelo `RandomForestClassifier` treinado no dataset UNSW-NB15 para classificar o tráfego de rede como 'Normal', 'Backdoor' ou 'Worms'.

A API foi projetada para ser facilmente consumida por uma aplicação frontend, fornecendo respostas ricas e detalhadas para alimentar visualizações e dashboards interativos.

## Funcionalidades

- **API Rápida e Moderna**: Construída com FastAPI para alta performance e documentação interativa automática (Swagger UI).
- **Múltiplos Métodos de Predição**: Suporta predições a partir de objetos JSON únicos, lotes de objetos e upload de arquivos CSV.
- **Respostas da API Enriquecidas**: Retorna informações detalhadas para cada predição, incluindo scores de confiança, distribuições de probabilidade completas e uma cópia dos dados de entrada, tornando-a ideal para integração com o frontend.
- **Insights do Modelo**: Inclui um endpoint que retorna as features (características) mais importantes que o modelo utiliza para tomar suas decisões.
- **Modelo de ML Otimizado**: O modelo é treinado para lidar com o desbalanceamento de classes e utiliza ajuste de hiperparâmetros (`RandomizedSearchCV`) para uma melhor precisão em tipos de ataques raros.

## Configuração e Instalação

Siga estes passos para configurar e executar o projeto localmente.

### 1. Pré-requisitos

- Python 3.9+
- `pip` e `venv`

### 2. Clonar o Repositório

```bash
git clone <url-do-seu-repositorio>
cd backend
```

### 3. Configurar um Ambiente Virtual (Virtual Environment)

É altamente recomendado usar um ambiente virtual para gerenciar as dependências do projeto.

```bash
# Crie o ambiente virtual
python -m venv venv

# Ative o ambiente
# No Windows
venv\Scripts\activate
# No macOS/Linux
source venv/bin/activate
```

### 4. Instalar Dependências

Instale todos os pacotes necessários usando o arquivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

*Nota: Se você não tiver um arquivo `requirements.txt`, pode criar um com o seguinte conteúdo:*
```txt
fastapi
uvicorn[standard]
pandas
joblib
scikit-learn
python-multipart
requests
```
*Em seguida, execute o comando `pip install -r requirements.txt`.*


## Executando a Aplicação

A aplicação é servida usando Uvicorn, um servidor ASGI de alta performance.

Para iniciar o servidor da API, execute o seguinte comando a partir do diretório raiz do projeto:

```bash
uvicorn app:app --reload --port 5000
```

- `app:app`: O primeiro `app` se refere ao arquivo `app.py`; o segundo se refere à instância `FastAPI` criada no arquivo.
- `--reload`: Ativa o recarregamento automático, para que o servidor reinicie sempre que houver alterações no código.

Após a execução, a API estará disponível em `http://127.0.0.1:5000`.

## Documentação Interativa da API

O FastAPI fornece documentação interativa e automática da API (gerada pelo Swagger UI). Com o servidor em execução, você pode acessá-la navegando para:

**`http://127.0.0.1:5000/docs`**

Esta interface permite que você explore e teste todos os endpoints da API diretamente do seu navegador.

## Endpoints da API

### 1. Obter Importância das Features

Fornece as 15 features mais importantes utilizadas pelo modelo. Ideal para exibir insights estáticos do modelo em um frontend.

- **URL**: `/features/importances`
- **Método**: `GET`
- **Resposta de Sucesso**: `200 OK`
- **Exemplo de Corpo da Resposta**:
  ```json
  [
    {
      "feature": "sbytes",
      "importance": 0.07664473928776257
    },
    {
      "feature": "sttl",
      "importance": 0.0728189815152829
    }
  ]
  ```

### 2. Predição via JSON

Envia um único objeto JSON ou uma lista de objetos para predição.

- **URL**: `/predict`
- **Método**: `POST`
- **Resposta de Sucesso**: `200 OK`
- **Exemplo de Corpo da Requisição (Único)**:
  ```json
  {
    "dur": 0.000011, "proto": "udp", "service": "-", "state": "INT", "spkts": 2, "dpkts": 0, "sbytes": 104, "dbytes": 0, "rate": 90909.0902, "sttl": 254, "dttl": 0, "sload": 37818180.0, "dload": 0.0, "sloss": 0, "dloss": 0, "sinpkt": 0.000011, "dinpkt": 0.0, "sjit": 0.0, "djit": 0.0, "swin": 0, "stcpb": 0, "dtcpb": 0, "dwin": 0, "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0, "smean": 52, "dmean": 0, "trans_depth": 0, "response_body_len": 0, "ct_srv_src": 2, "ct_state_ttl": 2, "ct_dst_ltm": 1, "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1, "is_ftp_login": 0, "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1, "ct_srv_dst": 2, "is_sm_ips_ports": 0
  }
  ```
- **Exemplo de Corpo da Resposta (Único)**:
  ```json
  {
    "input_data": {
      "dur": 1.1e-05, "proto": "udp", ...
    },
    "result": {
      "prediction": "Normal",
      "is_anomaly": false,
      "confidence": 0.975,
      "probabilities": {
        "Backdoor": 0.0,
        "Normal": 0.975,
        "Worms": 0.025
      }
    }
  }
  ```

### 3. Predição via CSV

Envia um arquivo CSV para predição. Cada linha é processada e uma lista de resultados é retornada.

- **URL**: `/predict/csv`
- **Método**: `POST`
- **Corpo da Requisição**: `multipart/form-data` contendo o arquivo CSV.
- **Resposta de Sucesso**: `200 OK`
- **Exemplo de Corpo da Resposta**:
  ```json
  [
    {
      "input_data": { "dur": 1.1e-05, ... },
      "result": { "prediction": "Normal", ... }
    },
    {
      "input_data": { "dur": 0.043303, ... },
      "result": { "prediction": "Backdoor", ... }
    }
  ]
  ```

## Executando o Script de Teste

Um script de teste, `test_api.py`, é fornecido para verificar se todos os endpoints da API estão funcionando corretamente.

1.  Certifique-se de que o servidor da API esteja em execução.
2.  Abra um novo terminal (com o ambiente virtual ativado).
3.  Execute o script:
    ```bash
    python test_api.py
    ```
O script imprimirá os códigos de status e as respostas para cada caso de teste.

## Treinamento do Modelo

Para treinar novamente o modelo com novos dados ou configurações diferentes, você pode usar o script `model_training.py`.

1.  Coloque seus datasets (`UNSW_NB15_training-set.parquet`, `UNSW_NB15_testing-set.parquet`) no diretório `./datasets/`.
2.  Execute o script de treinamento:
    ```bash
    python model_training.py
    ```
Este processo realizará uma busca por hiperparâmetros e salvará o melhor modelo, o scaler e a lista de colunas como `model.joblib`, `scaler.joblib`, e `model_columns.joblib`, respectivamente. A API carregará automaticamente esses novos arquivos ao ser reiniciada.