# Predição de Diabetes com IA (Machine Learning + FastAPI + HTML)

Este projeto tem como objetivo prever o risco de diabetes em pacientes com base em dados clínicos, utilizando Machine Learning com Python. O projeto inclui:

- Treinamento de modelo com Random Forest
- Backend em FastAPI
- Frontend simples em HTML + JavaScript
- Comunicação via API

---

## 📦 Requisitos

- Python 3.9+
- pip
- VS Code (opcional, mas recomendado)

---

## ⚙️ Instalação e Configuração

1. Clone o repositório:

```bash
git clone https://github.com/Arturr1001/predicao_diabates.git
cd predicao_diabates
```

2. Crie e ative um ambiente virtual:

Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## 📊 Treinando o Modelo

Antes de executar a API, é necessário gerar o arquivo do modelo (.pkl) com os dados.

1. Certifique-se de que o arquivo diabetes_dataset.csv esteja na raiz do projeto.

2. Execute o script de treinamento:

```bash
python model/treino_modelo.py
```

Isso irá treinar um modelo RandomForestClassifier e salvar o arquivo diabetes_model.pkl dentro da pasta app/.

---

## 🚀 Rodando o Backend (FastAPI)

Com o modelo treinado, agora você pode rodar o servidor:

```bash
uvicorn app.main:app --reload
```

- A API estará disponível em: http://127.0.0.1:8000
- A interface de testes Swagger estará em: http://127.0.0.1:8000/docs

---

## 🌐 Rodando o Frontend

O frontend é uma página HTML simples que envia dados para a API.

1. Abra o arquivo frontend/index.html em seu navegador:

Linux/macOS:
```bash
open frontend/index.html
```

Windows:
```bash
start frontend/index.html
```

2. Preencha o formulário com os dados do paciente e clique em “Prever”. O resultado será exibido na tela com base na resposta da API.

---

## 🥪 Testando com Dados Fictícios

Exemplo de entrada para testar:

| Campo               | Valor       |
|---------------------|-------------|
| gender              | male        |
| age                 | 54          |
| bmi                 | 32.1        |
| hbA1c_level         | 7.2         |
| blood_glucose_level | 145         |
| hypertension        | 1           |
| heart_disease       | 0           |
| smoking_history     | current     |
| race_Caucasian      | 1           |
| (demais raças)      | 0           |

Resultado esperado: diagnóstico positivo para diabetes.

---

## 🧠 Sobre o Modelo

- Modelo: RandomForestClassifier
- Acurácia aproximada: 97%
- Técnicas utilizadas:
  - OneHotEncoding para variáveis categóricas
  - StandardScaler para variáveis numéricas
  - Pipeline de pré-processamento
  - joblib para persistência do modelo

---

## 📂 Estrutura de Pastas

- model/ → Treinamento do modelo
- app/ → Backend com FastAPI + modelo .pkl
- frontend/ → Página HTML com formulário

---

## 📝 Licença

Este projeto é de uso educacional. Fique à vontade para usar e modificar.
