import pandas as pd
import joblib

player_matches =pd.read_csv('player_matches.csv')

player_stats={}

for player in player_matches['player'].unique():
    player_data=player_matches[player_matches['player']==player]

    player_stats[player]={
        'rank':player_data['player_rank'].iloc[-1],
        'age':player_data['player_age'].iloc[-1],
        'hand':player_data['player_hand'].mode()[0] if len(player_data)>0 else 'R',
        'ht':player_data['player_ht'].median(),
        'elo':player_data['player_elo'].iloc[-1] if 'player_elo' in player_data.columns else 1500,
        'win_rate':player_data['win'].mean(),
        'Hard_win_rate':player_data[player_data['surface']=='Hard']['win'].mean() if len(player_data[player_data['surface']=='Hard']) >0 else 0.5,
        'Clay_win_rate':player_data[player_data['surface']=='Clay']['win'].mean() if len(player_data[player_data['surface']=='Clay']) >0 else 0.5,
        'Grass_win_rate':player_data[player_data['surface']=='Grass']['win'].mean() if len(player_data[player_data['surface']=='Grass']) >0 else 0.5,
        'Carpet_win_rate':player_data[player_data['surface']=='Carpet']['win'].mean() if len(player_data[player_data['surface']=='Carpet']) >0 else 0.5,
        'avg_aces':player_data['player_average_aces'].mean() if 'player_average_aces' in player_data.columns else 5.0,
        'avg_df':player_data['player_avg_df'].mean() if 'player_avg_df' in player_data.columns else 5.0
    }

joblib.dump(player_stats,'player_stats.joblib')
print(f'{len(player_stats)} players processed and saved to player_stats.joblib')
