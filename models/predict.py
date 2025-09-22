import pandas as pd
import pickle
from database.fetch_data import fetch_games
from database.process_data import process_games

def load_model(path="models/xgb_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_pred_side(pred):
    return "HOME" if pred == 1 else "AWAY"

def show_predictions(return_data=False):
    """
    Mostra ou retorna as previsões.
    - return_data=False → imprime no terminal (modo atual)
    - return_data=True  → retorna DataFrames (para Flask ou outros usos)
    """
    # Carregar modelo
    model = load_model()

    # Buscar e processar dados
    raw_data = fetch_games()
    df = process_games(raw_data)

    # Features usadas no modelo
    stats = ["PTS", "REB", "AST", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]
    features = []
    for feat in stats:
        features.append(f'HOME_{feat}_ROLL20')
        features.append(f'AWAY_{feat}_ROLL20')
        features.append(f'DIFF_{feat}')
    features += ["HOME_WINRATE_20", "AWAY_WINRATE_20"]

    # Remover linhas com valores faltantes
    df = df.dropna(subset=features)

    # Últimos 10 jogos
    df = df.sort_values("GAME_DATE")
    last_10 = df.tail(10).copy()
    X_last10 = last_10[features]
    preds_last10 = model.predict(X_last10)

    last_10["Predição"] = [get_pred_side(p) for p in preds_last10]
    last_10["Acerto"] = last_10.apply(
        lambda x: x["WINNER"] == (x["HOME_TEAM"] if x["Predição"] == "HOME" else x["AWAY_TEAM"]),
        axis=1
    )

    # Jogos do dia
    today = pd.to_datetime("today").date()
    today_games = df[df["GAME_DATE"].dt.date == today].copy()
    if not today_games.empty:
        X_today = today_games[features]
        preds_today = model.predict(X_today)
        today_games["Predição"] = [get_pred_side(p) for p in preds_today]

    # Retorno para Flask
    if return_data:
        return last_10, today_games

    # Impressão no terminal
    print("\n=== 10 ÚLTIMOS JOGOS ===")
    for _, row in last_10.iterrows():
        real_winner = row["WINNER"]
        pred_winner = row["HOME_TEAM"] if row["Predição"] == "HOME" else row["AWAY_TEAM"]
        print(f"{row['GAME_DATE'].date()} - {row['HOME_TEAM']} X {row['AWAY_TEAM']} - "
              f"Vencedor real: {real_winner} - "
              f"Predição: {pred_winner} - "
              f"{'✅' if row['Acerto'] else '❌'}")

    if today_games.empty:
        print("\nNenhum jogo encontrado para hoje.")
    else:
        print("\n=== JOGOS DO DIA ===")
        for _, row in today_games.iterrows():
            pred_winner = row["HOME_TEAM"] if row["Predição"] == "HOME" else row["AWAY_TEAM"]
            print(f"{row['GAME_DATE'].date()} - {row['HOME_TEAM']} X {row['AWAY_TEAM']} - Predição: {pred_winner}")
