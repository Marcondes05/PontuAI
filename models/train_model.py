import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score

def train_model(csv_path="data/nba_games.csv"):
    df = pd.read_csv(csv_path)

    # Mapear vitória da casa para binário usando 'WINNER'
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

    # XGBoost com KFold
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf)
    avg_acc = scores.mean()  # média da acurácia

    print(f"🎯 Acurácia média (KFold 5x): {avg_acc*100:.2f}%")

    model.fit(X, y)
    with open("models/xgb_model.pkl","wb") as f:
        pickle.dump(model, f)
    print("✅ Modelo salvo em models/xgb_model.pkl")

    return avg_acc
