from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load('ao_open.joblib')
encoders=joblib.load('encoder.joblib')
features=joblib.load('features.joblib')
player_stats=joblib.load('player_stats.joblib')

class MatchInput(BaseModel):
    player_name:str
    opponent_name:str
    surface:str
    round:str

@app.post("/predict")
async def predict_match(match : MatchInput):
    try:
        p1=player_stats.get(match.player_name)
        p2=player_stats.get(match.opponent_name)

        if not p1 or not p2:
            return {"error": "Player not found"}
        
        match_data={
            'surface':encoders['surface'].transform([match.surface])[0],
            'tourney_level':encoders['level'].transform(['G'])[0],
            'round':encoders['round'].transform([match.round])[0],
            'best_of':5,
            'player_rank':p1['rank'],
            'opp_rank':p2['rank'],
            'player_age':p1['age'],
            'opp_age':p2['age'],
            'player_hand':encoders['hand'].transform([p1['hand']])[0],
            'opp_hand':encoders['hand'].transform([p2['hand']])[0],
            'player_ht':p1['ht'],
            'opp_ht':p2['ht'],
            'player_average_aces':p1['player_average_aces'],
            'player_avg_df':p1['avg_df'],
            'player_win_rate':p1['win_rate'],
            'player_surface_win_rate':p1[f'{match.surface}_win_rate'],
            'opp_average_aces':p2['player_average_aces'],
            'opp_avg_df':p2['avg_df'],  
            'opp_win_rate':p2['win_rate'],
            'opp_surface_win_rate':p2[f'{match.surface}_win_rate'],
            'player_elo':p1['elo'],
            'opp_elo':p2['elo'],
            'player_surface_elo':p1['player_surface_elo'],
            'opp_surface_elo':p2['opp_surface_elo '],


        }
        X=pd.DataFrame([match_data],columns=features)
        prob=model.predict_proba(X)[0]

        return {
            "player_name":match.player_name,
            "opponent_name":match.opponent_name,
            "player_win_probability":prob[1],
            "opponent_win_probability":prob[0]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get('/')
async def root():
    return {"message":"Tennis Match Outcome Prediction API"}
@app.get('/players')
async def get_players():
    return {"players":list(player_stats.keys())[:100]}

    