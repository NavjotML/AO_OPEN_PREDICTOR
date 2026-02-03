import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('ao_open.joblib')
encoders = joblib.load('encoder.joblib')
features = joblib.load('features.joblib')
player_stats = joblib.load('player_stats.joblib')

def predict_match(player1_name, player2_name, surface='Hard', round_val='SF'):
    """Predict match outcome with detailed metrics"""
    
    p1 = player_stats.get(player1_name)
    p2 = player_stats.get(player2_name)
    
    if not p1 or not p2:
        return f"Player not found in database"
    
    match_data = {
        'surface': encoders['surface'].transform([surface])[0],
        'tourney_level': encoders['level'].transform(['G'])[0],
        'round': encoders['round'].transform([round_val])[0],
        'best_of': 5,
        'player_rank': p1['rank'],
        'opp_rank': p2['rank'],
        'player_age': p1['age'],
        'opp_age': p2['age'],
        'player_hand': encoders['hand'].transform([p1['hand']])[0],
        'opp_hand': encoders['hand'].transform([p2['hand']])[0],
        'player_ht': p1['ht'],
        'opp_ht': p2['ht'],
        'player_average_aces': p1['avg_aces'],
        'player_avg_df': p1['avg_df'],
        'player_win_rate': p1['win_rate'],
        'player_surface_win_rate': p1[f'{surface}_win_rate'],
        'opp_average_aces': p2['avg_aces'],
        'opp_avg_df': p2['avg_df'],
        'opp_win_rate': p2['win_rate'],
        'opp_surface_win_rate': p2[f'{surface}_win_rate'],
        'player_elo': p1['elo'],
        'opp_elo': p2['elo'],
        'player_surface_elo': p1['elo'],
        'opp_surface_elo': p2['elo']
    }
    
    X = pd.DataFrame([match_data])[features]
    prob = model.predict_proba(X)[0]
    
    # Calculate confidence metrics
    winner = player1_name if prob[1] > 0.5 else player2_name
    winner_prob = max(prob)
    margin = abs(prob[1] - prob[0])
    
    # Confidence level
    if margin < 0.1:
        confidence = "⚠️ Very Close Match"
    elif margin < 0.2:
        confidence = "📊 Slight Edge"
    elif margin < 0.35:
        confidence = "✅ Moderate Confidence"
    else:
        confidence = "🔥 High Confidence"
    
    print(f"\n{'='*60}")
    print(f"🎾 {player1_name} vs {player2_name}")
    print(f"📍 Surface: {surface} | Round: {round_val} | Best of 5")
    print(f"{'='*60}")
    
    # Player stats comparison
    print(f"\n📊 KEY STATS COMPARISON:")
    print(f"{'Metric':<20} {player1_name:<15} {player2_name:<15}")
    print(f"{'-'*60}")
    print(f"{'Rank':<20} #{p1['rank']:<14} #{p2['rank']:<14}")
    print(f"{'ELO Rating':<20} {p1['elo']:<14} {p2['elo']:<14}")
    print(f"{'Overall Win Rate':<20} {p1['win_rate']*100:.1f}%{'':<10} {p2['win_rate']*100:.1f}%")
    print(f"{f'{surface} Win Rate':<20} {p1[f'{surface}_win_rate']*100:.1f}%{'':<10} {p2[f'{surface}_win_rate']*100:.1f}%")
    print(f"{'Avg Aces':<20} {p1['avg_aces']:.1f}{'':<12} {p2['avg_aces']:.1f}")
    print(f"{'Hand':<20} {p1['hand']:<14} {p2['hand']:<14}")
    
    print(f"\n🔮 PREDICTION:")
    print(f"{player1_name}: {prob[1]*100:.1f}%")
    print(f"{player2_name}: {prob[0]*100:.1f}%")
    
    print(f"\n🏆 PREDICTED WINNER: {winner}")
    print(f"📈 Confidence: {confidence} ({margin*100:.1f}% margin)")
    print(f"{'='*60}\n")
    
    return prob

# PREDICT TODAY'S MATCHES
print("\n" + "🏆 AUSTRALIAN OPEN 2025 - SEMI FINALS PREDICTIONS 🏆".center(60))
print("Model Accuracy: 71%\n")

prob1 = predict_match("Carlos Alcaraz", "Novak Djokovic", surface='Hard', round_val='F')


# Ove