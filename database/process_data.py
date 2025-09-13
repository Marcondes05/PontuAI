import pandas as pd

TEAM_MAP = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets", "DET": "Detroit Pistons", "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets", "IND": "Indiana Pacers", "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

def get_opponent(row):
    matchup = row.get("MATCHUP")
    team = row.get("TEAM_ABBREVIATION")
    if not matchup or pd.isna(matchup):
        return "DESCONHECIDO"

    parts = matchup.split(" ")
    siglas = [parts[0], parts[2]]
    opponent_sigla = siglas[0] if siglas[0] != team else siglas[1]
    return TEAM_MAP.get(opponent_sigla, opponent_sigla)

def classify_home_away(row):
    matchup = row.get("MATCHUP")
    if matchup is None or pd.isna(matchup):
        return pd.Series(["DESCONHECIDO", "DESCONHECIDO"], index=["HOME_TEAM", "AWAY_TEAM"])
    if "@" in matchup:
        return pd.Series([row["OPPONENT_NAME"], row["TEAM_NAME"]], index=["HOME_TEAM", "AWAY_TEAM"])
    else:
        return pd.Series([row["TEAM_NAME"], row["OPPONENT_NAME"]], index=["HOME_TEAM", "AWAY_TEAM"])

def process_games(df, rolling_window=20):
    # Adiciona coluna de oponente
    df["OPPONENT_NAME"] = df.apply(get_opponent, axis=1)
    df["TEAM_NAME"] = df["TEAM_ABBREVIATION"].map(TEAM_MAP)

    # Converte GAME_DATE para datetime
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Define HOME_TEAM e AWAY_TEAM
    df[["HOME_TEAM", "AWAY_TEAM"]] = df.apply(classify_home_away, axis=1)

    # Stats que vamos usar
    stats = ["PTS", "REB", "AST", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]

    # Cria uma linha por jogo: consolida home e away
    games = df.groupby(["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]).agg({
        "PTS": ["first", "last"],
        "REB": ["first", "last"],
        "AST": ["first", "last"],
        "FG_PCT": ["first", "last"],
        "FG3_PCT": ["first", "last"],
        "FT_PCT": ["first", "last"],
        "PLUS_MINUS": ["first", "last"]
    }).reset_index()

    # Renomeia colunas
    games.columns = [
        "GAME_DATE", "HOME_TEAM", "AWAY_TEAM",
        "HOME_PTS", "AWAY_PTS",
        "HOME_REB", "AWAY_REB",
        "HOME_AST", "AWAY_AST",
        "HOME_FG_PCT", "AWAY_FG_PCT",
        "HOME_FG3_PCT", "AWAY_FG3_PCT",
        "HOME_FT_PCT", "AWAY_FT_PCT",
        "HOME_PLUS_MINUS", "AWAY_PLUS_MINUS"
    ]

    # Define vencedor real com base nos pontos
    games["WINNER"] = games.apply(
        lambda x: x["HOME_TEAM"] if x["HOME_PTS"] > x["AWAY_PTS"] else x["AWAY_TEAM"],
        axis=1
    )

    # Cria rolling médias móveis por time
    for team_col, prefix in [("HOME_TEAM", "HOME"), ("AWAY_TEAM", "AWAY")]:
        for stat in stats:
            roll_col = f"{prefix}_{stat}_ROLL{rolling_window}"
            games[roll_col] = (
                games.groupby(team_col)[f"{prefix}_{stat}"]
                .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
            )

        # Rolling winrate
        win_col = f"{prefix}_WINRATE_{rolling_window}"
        games[win_col] = (
            games.groupby(team_col).apply(
                lambda g: g.apply(lambda row: row[prefix+"_PTS"], axis=1)
                .rolling(rolling_window, min_periods=1)
                .apply(lambda x: sum(x>0)/len(x))
            ).reset_index(level=0, drop=True)
        )

    # Diferença HOME - AWAY
    for stat in stats:
        games[f"DIFF_{stat}"] = games[f"HOME_{stat}_ROLL{rolling_window}"] - games[f"AWAY_{stat}_ROLL{rolling_window}"]

    return games