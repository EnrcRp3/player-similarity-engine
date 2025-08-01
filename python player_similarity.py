# Streamlit - NBA Player Similarity Engine

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from nba_api.stats.endpoints import leaguedashplayerstats
import streamlit as st
import matplotlib.pyplot as plt

# --- Funzioni ---
@st.cache_data(show_spinner=False)
def get_nba_stats(season="2023-24"):
    stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season, per_mode_detailed="PerGame")
    df = stats.get_data_frames()[0]
    return df

@st.cache_data(show_spinner=False)
def process_data(df, features):
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    similarity_matrix = cosine_similarity(X_pca)
    return X, X_pca, similarity_matrix

@st.cache_data(show_spinner=False)
def find_similar_players(player_name, k, df, sim_matrix):
    names = df['PLAYER_NAME']
    try:
        idx = names[names == player_name].index[0]
    except:
        return []
    sims = list(enumerate(sim_matrix[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:k+1]
    return [(names[i], round(score, 3)) for i, score in sims]

def plot_radar(player_1_stats, player_2_stats, labels, player_1, player_2):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats_1 = player_1_stats.tolist()
    stats_2 = player_2_stats.tolist()

    stats_1 += stats_1[:1]
    stats_2 += stats_2[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats_1, label=player_1, color='blue')
    ax.fill(angles, stats_1, alpha=0.25, color='blue')
    ax.plot(angles, stats_2, label=player_2, color='red')
    ax.fill(angles, stats_2, alpha=0.25, color='red')
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Confronto Radar Statistico")
    ax.legend(loc='upper right')
    st.pyplot(fig)

# --- Streamlit UI ---
st.set_page_config(page_title="NBA Player Similarity Engine", layout="wide")
st.title("🏀 NBA Player Similarity Engine")

season = st.sidebar.selectbox("Seleziona stagione", ["2023-24"])
k_sim = st.sidebar.slider("Numero di giocatori simili", 1, 10, 5)

features = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
labels = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FG%', '3P%', 'FT%']

nba_df = get_nba_stats(season)
X, X_pca, sim_matrix = process_data(nba_df, features)
player_names = nba_df['PLAYER_NAME'].tolist()

selected_player = st.selectbox("Scegli un giocatore", player_names)
similar = find_similar_players(selected_player, k_sim, nba_df, sim_matrix)

st.markdown(f"### Giocatori simili a **{selected_player}**")
for name, score in similar:
    st.write(f"🔹 {name} (similarità: {score})")

# Radar chart
if similar:
    selected_sim_name = st.selectbox("Confronta con uno dei simili:", [s[0] for s in similar])

    player1_df = nba_df[nba_df['PLAYER_NAME'] == selected_player]
    player2_df = nba_df[nba_df['PLAYER_NAME'] == selected_sim_name]

    if not player1_df.empty and not player2_df.empty:
        player1_row = player1_df[features].iloc[0]
        player2_row = player2_df[features].iloc[0]
        plot_radar(player1_row, player2_row, labels, selected_player, selected_sim_name)
    else:
        st.warning("Impossibile trovare dati completi per uno dei due giocatori.")
else:
    st.info("Nessun giocatore simile trovato. Riduci il valore di 'k' nella sidebar o scegli un altro giocatore.")
