import json
from flask import Flask, render_template, request
from models.predict import show_predictions

app = Flask(__name__)

def load_accuracy():
    try:
        with open("static/images/metrics.json", "r") as j:
            metrics = json.load(j)
            return float(metrics.get("avg_acc", 0.0)) * 100
    except:
        return None

@app.route("/")
def home():
    accuracy_pct = load_accuracy()
    last_10, today_games = show_predictions(return_data=True)

    # formatar datas
    last_10["GAME_DATE"] = last_10["GAME_DATE"].dt.strftime("%Y-%m-%d")
    today_games["GAME_DATE"] = today_games["GAME_DATE"].dt.strftime("%Y-%m-%d")

    return render_template(
        "index.html",
        last_10=last_10.to_dict(orient="records"),
        today_games=today_games.to_dict(orient="records"),
        accuracy=round(accuracy_pct, 2) if accuracy_pct else None,
        training_cv="images/training_cv.png",
        feature_imp="images/feature_importance.png"
    )

@app.route("/predictions")
def predictions():
    _, today_games = show_predictions(return_data=True)
    today_games["GAME_DATE"] = today_games["GAME_DATE"].dt.strftime("%Y-%m-%d")

    return render_template("predictions.html", today_games=today_games.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
