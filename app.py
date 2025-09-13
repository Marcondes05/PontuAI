import time
from models.train_model import train_model
from models.predict import show_predictions


def main():
    print("🏀 Previsor de Resultados da NBA")
    print("=" * 50)
    print("Este projeto utiliza técnicas de IA para prever os resultados da NBA.")
    print("Pipeline:")
    print("1. Coleta de dados com nba_api.")
    print("2. Processamento e criação de estatísticas.")
    print("3. Treinamento de modelo XGBoost.")
    print("4. Predição de resultados futuros.")
    print("=" * 50)

    start = time.time()

    # Treinar modelo
    print("\n🤖 Treinando modelo...")
    acc = train_model("data/nba_games.csv")

    # Mostrar previsões
    print("\n📊 Resultados dos últimos jogos:")
    last_10, today_preds = show_predictions(return_data=True)
    print(last_10)

    # Contagem de acertos
    correct = sum(last_10["WINNER"] == last_10["Predição"])
    total = len(last_10)
    print(f"\n✅ Acertos nos últimos jogos: {correct}/{total} ({correct/total:.2%})")

    # Acurácia global
    print(f"🎯 Acurácia do modelo: {acc:.2%}")

    # Tempo de execução
    end = time.time()
    print(f"⏱️ Tempo total (s): {end - start:.2f}")

    # Previsões para hoje
    print("\n📅 Previsões para os jogos de hoje:")
    print(today_preds)


if __name__ == "__main__":
    main()
