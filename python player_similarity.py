# Funzione per caricare le statistiche della stagione corrente

def get_nba_player_stats(season="2023-24"):
    print("Scaricamento dati... potrebbe volerci qualche secondo.")
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed="PerGame")
    df = stats.get_data_frames()[0]
    return df
# Preleva i dati
nba_df = get_nba_player_stats()

# Selezione delle colonne utili
features = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
player_names = nba_df['PLAYER_NAME']
X = nba_df[features].fillna(0)

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# Similarità coseno
similarity_matrix = cosine_similarity(X_pca)

# Funzione per trovare i più simili
def find_similar_players(player_name, k=5):
    try:
        idx = player_names[player_names == player_name].index[0]
    except:
        return f"Giocatore '{player_name}' non trovato."

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]

    similar_players = [(player_names[i], round(score, 3)) for i, score in sim_scores]
    return similar_players
