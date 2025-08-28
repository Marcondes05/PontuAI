from database.fetch_data import fetch_games
from database.process_data import process_games
from database.save_data import save_csv
from models.train_model import train_model
from models.predict import show_predictions

def main():
    print(" Coletando dados das últimas 5 temporadas...")
    raw_data = fetch_games()

    print(" Processando dados e criando features...")
    df = process_games(raw_data)

    print(" Salvando dataset processado...")
    save_csv(df, "data/nba_games.csv")

    print(" Treinando modelo XGBoost com validação KFold...")
    train_model("data/nba_games.csv")

    print(" Mostrando previsões para os últimos jogos...")
    show_predictions()

if __name__ == "__main__":
    main()
