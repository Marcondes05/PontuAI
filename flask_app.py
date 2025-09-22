from flask import Flask, render_template
from models.predict import show_predictions
from models.train_model import train_model

app = Flask(__name__)

def normalize_name(name: str) -> str:
    return name.lower().replace(" ", "-")

@app.route("/")
def home():
    # Treina o modelo e pega acurácia
    acc = train_model("data/nba_games.csv")
    # Últimos 10 jogos + jogos de hoje
    last_10, today_games = show_predictions(return_data=True)

    # Converter datas
    last_10["GAME_DATE"] = last_10["GAME_DATE"].dt.strftime("%Y-%m-%d")
    today_games["GAME_DATE"] = today_games["GAME_DATE"].dt.strftime("%Y-%m-%d")

    # Adicionar caminho das logos
    last_10["HOME_LOGO"] = last_10["HOME_TEAM"].apply(lambda x: f"logos/{normalize_name(x)}.png")
    last_10["AWAY_LOGO"] = last_10["AWAY_TEAM"].apply(lambda x: f"logos/{normalize_name(x)}.png")
    today_games["HOME_LOGO"] = today_games["HOME_TEAM"].apply(lambda x: f"logos/{normalize_name(x)}.png")
    today_games["AWAY_LOGO"] = today_games["AWAY_TEAM"].apply(lambda x: f"logos/{normalize_name(x)}.png")

    # 🔹 Adicionar logo do vencedor
    last_10["WINNER_LOGO"] = last_10["WINNER"].apply(lambda x: f"logos/{normalize_name(x)}.png")

    return render_template(
        "index.html",
        last_10=last_10.to_dict(orient="records"),
        today_games=today_games.to_dict(orient="records"),
        accuracy=round(acc * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
