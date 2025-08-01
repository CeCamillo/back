# Backend de Detecção de Intrusão com Machine Learning

Este projeto consiste em um backend desenvolvido em Python com Flask que serve um modelo de Machine Learning treinado para detectar e classificar tipos específicos de ataques de rede: 'Worms' e 'Backdoor'.

O sistema é capaz de receber dados de tráfego de rede, tanto em formato JSON quanto via upload de arquivos CSV, e retornar a classificação do tráfego como 'Normal', 'Worms' ou 'Backdoor', além das probabilidades associadas a cada classificação.

## Como Utilizar

Siga os passos abaixo para configurar e executar o projeto em seu ambiente local.

### 1. Pré-requisitos

- Python 3.8 ou superior
- `pip` (gerenciador de pacotes do Python)

### 2. Instalação

Primeiro, clone ou faça o download deste repositório. Em seguida, navegue até o diretório raiz do projeto e crie um ambiente virtual.

**Windows:**
```bash
python -m venv .
.\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .
source bin/activate
```

Com o ambiente virtual ativado, instale todas as dependências necessárias executando o seguinte comando:

```bash
pip install -r requirements.txt
```

### 3. Treinamento do Modelo

Antes de iniciar o servidor, você precisa treinar o modelo de Machine Learning. O script de treinamento utilizará os datasets localizados na pasta `/datasets` para gerar os artefatos do modelo (`model.joblib`), o scaler (`scaler.joblib`) e as colunas do modelo (`model_columns.joblib`).

Para treinar o modelo, execute:

```bash
python model_training.py
```

Este processo pode levar alguns minutos. Ao final, você verá uma mensagem confirmando que o modelo e os artefatos foram salvos com sucesso.

### 4. Iniciando o Servidor da API

Com o modelo treinado e os artefatos salvos, você pode iniciar o servidor Flask.

```bash
python app.py
```

O servidor estará em execução no endereço `http://127.0.0.1:5000`.

### 5. Utilizando os Endpoints da API

O backend oferece dois endpoints principais para realizar predições.

#### A. Endpoint para Dados em JSON (`/predict`)

Este endpoint aceita um `POST` request com dados em formato JSON. Você pode enviar um único registro (um objeto JSON) ou múltiplos registros (uma lista de objetos JSON).

**URL:** `http://127.0.0.1:5000/predict`
**Método:** `POST`
**Body (Exemplo com um único registro):**
```json
{
  "proto": "tcp",
  "state": "FIN",
  "dur": 0.001,
  "sbytes": 300,
  "dbytes": 400
}
```

#### B. Endpoint para Arquivos CSV (`/predict/csv`)

Este endpoint aceita o upload de um arquivo `.csv` para predições em lote. O arquivo deve ser enviado como parte de um `multipart/form-data` request, com a chave do arquivo sendo `file`.

**URL:** `http://127.0.0.1:5000/predict/csv`
**Método:** `POST`
**Body:** `multipart/form-data` com o arquivo CSV.

---

Seu amigo do frontend pode agora fazer requisições para esses endpoints e receberá uma resposta JSON contendo a `prediction` (classificação) e as `probabilities` (probabilidades) para cada linha de dados enviada. 