import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score


def train_model(csv_path="data/nba_games.csv", save_plots=True):
    """
    Treina o XGBoost no dataset e opcionalmente salva:
      - models/xgb_model.pkl
      - static/images/training_cv.png
      - static/images/feature_importance.png
      - static/images/metrics.json (com avg_acc)
    Retorna avg_acc (float, 0..1).
    """
    df = pd.read_csv(csv_path)

    # Vitória em casa = 1, visitante = 0
    df["WL_bin"] = (df["WINNER"] == df["HOME_TEAM"]).astype(int)

    # Features
    stats = ["PTS","REB","AST","FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS"]
    features = []
    for feat in stats:
        features.append(f'HOME_{feat}_ROLL20')
        features.append(f'AWAY_{feat}_ROLL20')
        features.append(f'DIFF_{feat}')
    features += ["HOME_WINRATE_20","AWAY_WINRATE_20"]

    df = df.dropna(subset=features + ["WL_bin"])
    X = df[features]
    y = df["WL_bin"]

    # Modelo + KFold
    model = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf)
    avg_acc = scores.mean()

    # Treino final
    model.fit(X, y)

    # Salvar modelo
    os.makedirs("models", exist_ok=True)
    with open("models/xgb_model.pkl","wb") as f:
        pickle.dump(model, f)

    print(f"🎯 Acurácia média (KFold 5x): {avg_acc*100:.2f}%")
    print("✅ Modelo salvo em models/xgb_model.pkl")

    if save_plots:
        os.makedirs("static/images", exist_ok=True)

        # Plot CV scores
        plt.figure(figsize=(7,4))
        plt.plot(range(1, len(scores)+1), scores * 100, marker='o', linewidth=2, label="Acurácia por Fold")
        plt.axhline(y=avg_acc*100, color="red", linestyle="--", label=f"Média: {avg_acc*100:.2f}%")
        plt.title("Acurácia por Validação Cruzada (K-Fold)", fontsize=12, fontweight="bold")
        plt.xlabel("Número do Fold")
        plt.ylabel("Acurácia (%)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("static/images/training_cv.png", dpi=150)
        plt.close()


        # Feature importance (top 20)
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)[:20]

        plt.figure(figsize=(8,6))
        feat_imp[::-1].plot(kind="barh", color="#1D428A")
        plt.title("Importância das Variáveis no Modelo", fontsize=12, fontweight="bold")
        plt.xlabel("Importância Relativa")
        plt.ylabel("Variáveis")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig("static/images/feature_importance.png", dpi=150)
        plt.close()


        # Salvar métricas
        metrics = {"avg_acc": float(avg_acc)}
        with open("static/images/metrics.json", "w") as j:
            json.dump(metrics, j)

        print("✅ Plots salvos em static/images/")
        print("✅ Métricas salvas em static/images/metrics.json")

    return avg_acc


if __name__ == "__main__":
    train_model()
