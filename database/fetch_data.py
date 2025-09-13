from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

def fetch_games(seasons=['2020-21','2021-22','2022-23','2023-24', '2024-25']):
    all_games = pd.DataFrame()
    for season in seasons:
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        games = gamefinder.get_data_frames()[0]
        all_games = pd.concat([all_games, games], ignore_index=True)
    return all_games