# 🏀 NBA Game Prediction AI  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.ai/)  

Este projeto tem como objetivo desenvolver um modelo de **Inteligência Artificial** capaz de prever os resultados de partidas da **National Basketball Association (NBA)** com base em estatísticas históricas das equipes.  

O sistema utiliza **Machine Learning** para analisar desempenho esportivo, identificar padrões e gerar previsões sobre partidas futuras.  

---

## 📌 Motivação  

O basquete da NBA é altamente competitivo, e prever resultados pode ajudar em análises esportivas, estudos acadêmicos, apostas responsáveis e exploração de técnicas avançadas de **aprendizado de máquina aplicado ao esporte**.  

Este projeto nasceu como uma forma de unir **estatística, programação e paixão pelo basquete**, aplicando técnicas de **ciência de dados** em um cenário real.  

---

## ⚡ Funcionalidades  

- 📊 **Coleta de dados históricos** da NBA via [nba_api](https://github.com/swar/nba_api).  
- 🧮 **Criação de features** (médias móveis, taxa de vitória, desempenho por temporada).  
- 🤖 **Treinamento do modelo preditivo** com **XGBoost**.  
- ✅ **Validação cruzada (KFold 5x)** para avaliação robusta.  
- 🔮 **Previsões de partidas futuras** da NBA.  
- 🖥️ **Interface em terminal** para exibição de resultados e acertos.  

---

## 🛠️ Tecnologias e Bibliotecas  

- [Python 3.10+](https://www.python.org/)  
- [Pandas](https://pandas.pydata.org/) – Manipulação de dados  
- [Scikit-learn](https://scikit-learn.org/) – Pré-processamento e validação  
- [XGBoost](https://xgboost.ai/) – Modelo de classificação  
- [Pickle](https://docs.python.org/3/library/pickle.html) – Salvamento do modelo treinado  
- [nba_api](https://github.com/swar/nba_api) – Coleta de dados  

---

## 📂 Estrutura do Projeto  

```
projeto_basquete/
│
├─ data/
│  └─ nba_games.csv             # Dados históricos da NBA
│
├─ database/
│  ├─ fetch_data.py             # Coleta de dados da NBA
│  ├─ process_data.py           # Processamento e criação de features
│  └─ save_data.py              # Script de salvamento
│
├─ models/
│  ├─ train_model.py            # Treinamento do modelo
│  ├─ predict.py                # Previsões futuras
│  ├─ model.pkl                 # Modelo salvo
│  └─ xgb_model.pkl             # Outra versão do modelo
│
├─ utils/
│  └─ helpers.py                # Funções auxiliares
│
├─ main.py                      # Script principal
├─ README.md                    # Documentação
└─ requirements.txt             # Dependências
```

---

## 🚀 Como Usar  

1. **Clone o repositório**  
```bash
git clone https://github.com/Marcondes05/previsao_nba.git
cd previsao_nba
```

2. **Instale as dependências**  
```bash
pip install -r requirements.txt
```

3. **Execute o script principal**  
```bash
python main.py
```

---

## 📈 Resultados  

- ✅ **Acurácia média:** `67.8%` em validação cruzada.  
- 🏆 Permite identificar a equipe com maior probabilidade de vitória.  

---

## 🔮 Próximos Passos  

- 📌 Explorar novas features (ex.: desempenho por período do jogo).  
- 📌 Ajustar hiperparâmetros para melhorar a acurácia.  
- 📌 Criar interface web ou dashboard interativo.  

---

## 👨‍💻 Autor  

Desenvolvido por **[Marcondes05](https://github.com/Marcondes05)** ✨  
Se gostou do projeto, deixe uma ⭐ no repositório!  

---

## 📜 Licença  

Este projeto está licenciado sob os termos da **[Licença MIT](LICENSE)**.  

Você pode usar, copiar, modificar e distribuir este projeto livremente, desde que mantenha os devidos créditos ao autor.  
