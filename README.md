# 🏀 PontuAI — NBA Game Prediction AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.ai/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-lightgrey.svg)](https://flask.palletsprojects.com/)

**PontuAI** é um sistema de **Inteligência Artificial** que prevê resultados de partidas da **NBA** usando estatísticas históricas, com interface web desenvolvida em **Flask** para exibir resultados, acurácia e previsões de forma interativa.

---

## 📌 Motivação

O basquete da NBA é altamente competitivo, e prever resultados ajuda em análises esportivas, estudos acadêmicos e exploração de técnicas de aprendizado de máquina aplicadas ao esporte.

PontuAI une **estatística, programação e paixão pelo basquete**, aplicando **ciência de dados** em um cenário real e acessível via web.

---

## ⚡ Funcionalidades

* 📊 Coleta e processamento de dados históricos da NBA via [nba\_api](https://github.com/swar/nba_api).
* 🧮 Criação de features como médias móveis, taxa de vitória e desempenho por temporada.
* 🤖 Treinamento do modelo preditivo com **XGBoost**.
* ✅ Validação cruzada (KFold 5x) para avaliação robusta.
* 🔮 Previsões de partidas futuras da NBA.
* 🖥️ Interface web com Flask para visualização de resultados, gráficos e acertos.

---

## 🛠️ Tecnologias e Bibliotecas

* [Python 3.10+](https://www.python.org/)
* [Flask](https://flask.palletsprojects.com/) – Interface web
* [Pandas](https://pandas.pydata.org/) – Manipulação de dados
* [Matplotlib](https://matplotlib.org/) – Visualização de gráficos
* [Scikit-learn](https://scikit-learn.org/) – Pré-processamento e validação
* [XGBoost](https://xgboost.ai/) – Modelo de classificação
* [Pickle](https://docs.python.org/3/library/pickle.html) – Salvamento do modelo
* [nba\_api](https://github.com/swar/nba_api) – Coleta de dados

---

## 📂 Estrutura do Projeto

```
pontuai/
│
├─ data/
│  └─ nba_games.csv               # Dados históricos
│
├─ database/
│  ├─ fetch_data.py               # Coleta de dados
│  ├─ process_data.py             # Criação de features
│  └─ save_data.py                # Salvamento
│
├─ models/
│  ├─ train_model.py              # Treinamento do modelo
│  ├─ predict.py                  # Previsões futuras
│  ├─ model.pkl                   # Modelo salvo
│  ├─ xgb_model.pkl               # Outra versão do modelo
│  └─ accuracy.txt                # Salva a acurácia
│
├─ static/
│  ├─ css/                     # Estilos da página
│  ├─ imagens/                 # Salva os Gráficos           
│  └─ logos/                   # Logos da NBA e das equipes
│
├─ templates/                 # HTML da interface
│  ├─ base.html               
│  ├─ index.html              # Pagina principal           
│  └─ predicitions.html       # Página de predição dos jogos do dia
│
├─ utils/
│  └─ helpers.py                  # Funções auxiliares
│
├─ app.py
├─ flask_app.py                   # Inicia servidor Flask│
├─ main.py                        # Script CLI (opcional)
├─ LICENSE
├─ README.md                      # Documentação
└─ requirements.txt               # Dependências
```

---

## 🚀 Como Rodar o Projeto

1. **Clone o repositório**

```bash
git clone https://github.com/Marcondes05/previsao_nba.git
cd previsao_nba
```

2. **Instale as dependências**

```bash
pip install -r requirements.txt
```

3. **Treine o modelo ou use um modelo salvo**

```bash
python models/train_model.py
# Isso cria 'model.pkl' ou 'xgb_model.pkl'
```

4. **Inicie o servidor Flask**

```bash
python run.py
```

5. **Acesse a interface web**
   Abra no navegador: `http://127.0.0.1:5000`

---

## 📈 Resultados

* ✅ **Acurácia média:** `67.8%` em validação cruzada.
* 🏆 Identificação da equipe com maior probabilidade de vitória via interface web ou terminal.

---

## 🔮 Próximos Passos

* 📌 Explorar novas features (desempenho por período do jogo).
* 📌 Ajustar hiperparâmetros para melhorar a acurácia.
* 📌 Criar dashboards interativos com gráficos em tempo real.

---

## 👨‍💻 Autor

Desenvolvido por **[Marcondes05](https://github.com/Marcondes05)** ✨
Se gostou do projeto, deixe uma ⭐ no repositório!

---

## 📜 Licença

Este projeto está licenciado sob **[MIT License](LICENSE)**.

Você pode usar, copiar, modificar e distribuir livremente, desde que mantenha os créditos ao autor.
