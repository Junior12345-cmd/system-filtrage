import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from io import StringIO

# Configuration de la page
st.set_page_config(page_title="Système de Recommandation", layout="wide")
st.title("🎬 Système de Recommandation de Films")
st.title("Réalisé par Junior SANNI | IFRI MASTER 2025")

def safe_pearson_sim(df, item1, item2):
    try:
        users_rated_both = df[(df[item1] > 0) & (df[item2] > 0)]
        if len(users_rated_both) < 2: return 0
        corr, _ = pearsonr(users_rated_both[item1], users_rated_both[item2])
        return max(0, corr) if not np.isnan(corr) else 0
    except:
        return 0

def get_recommendations(df, user, sim_matrix, n=5):
    try:
        user_ratings = df.set_index('Utilisateur').loc[user]
        unrated_movies = [m for m in df.columns if m != 'Utilisateur' and user_ratings[m] == 0]
        
        predictions = []
        for movie in unrated_movies:
            try:
                similar_movies = sim_matrix[movie][sim_matrix[movie] > 0].index
                rated_similar_movies = [m for m in similar_movies if m in user_ratings.index and user_ratings[m] > 0]
                
                if not rated_similar_movies:
                    continue
                    
                numerator = sum(user_ratings[m] * sim_matrix.loc[movie, m] for m in rated_similar_movies)
                denominator = sum(sim_matrix.loc[movie, m] for m in rated_similar_movies)
                
                if denominator > 0:
                    predicted_rating = numerator / denominator
                    predictions.append((movie, predicted_rating))
            except:
                continue
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]
    except:
        return []

def main():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'sim_matrix' not in st.session_state:
        st.session_state.sim_matrix = None
    
    # Déplacer la déclaration de load_method ici
    st.sidebar.header("Options de chargement")
    load_method = st.sidebar.radio("Choisir la méthode:",
                                 ["📝 Saisie manuelle", "📂 Upload CSV"])
    
    # Option 1: Saisie manuelle
    if load_method == "📝 Saisie manuelle":
        st.header("Saisie manuelle des données")
        
        # Configuration de la structure
        col1, col2, col3 = st.columns([1,1,2])
        num_users = col1.number_input("Nombre d'utilisateurs", min_value=1, value=3, key='num_users')
        num_movies = col2.number_input("Nombre de films", min_value=1, value=4, key='num_movies')
        
        # Initialisation/mise à jour du DataFrame
        if 'edited_df' not in st.session_state:
            data = {'Utilisateur': [f'User{i+1}' for i in range(num_users)]}
            for i in range(num_movies):
                data[f'Film{i+1}'] = [0] * num_users
            st.session_state.edited_df = pd.DataFrame(data)
        else:
            # Ajuster selon les inputs
            current_users = len(st.session_state.edited_df)
            current_movies = len([col for col in st.session_state.edited_df.columns if col.startswith("Film")])
            
            # Gestion utilisateurs
            if num_users > current_users:
                for i in range(current_users, num_users):
                    new_user = f'User{i+1}'
                    st.session_state.edited_df.loc[len(st.session_state.edited_df)] = {
                        'Utilisateur': new_user,
                        **{f'Film{j+1}': 0 for j in range(current_movies)}
                    }
            elif num_users < current_users:
                st.session_state.edited_df = st.session_state.edited_df.iloc[:num_users]
            
            # Gestion films
            if num_movies > current_movies:
                for i in range(current_movies, num_movies):
                    st.session_state.edited_df[f'Film{i+1}'] = 0
            elif num_movies < current_movies:
                cols_to_drop = [f'Film{i+1}' for i in range(num_movies, current_movies)]
                st.session_state.edited_df = st.session_state.edited_df.drop(columns=cols_to_drop)
        
        # Éditeur de données
        st.subheader("Modifier les notes (0 = non noté, 1-5 = note)")
        st.session_state.edited_df = st.data_editor(
            st.session_state.edited_df,
            column_config={
                "Utilisateur": st.column_config.TextColumn("Utilisateur", width="medium"),
                **{col: st.column_config.NumberColumn(format="%d", min_value=0, max_value=5) 
                   for col in st.session_state.edited_df.columns if col.startswith("Film")}
            },
            num_rows="fixed",
            hide_index=True
        )
        
        if st.button("Valider les données"):
            st.session_state.df = st.session_state.edited_df
            st.success("Données validées!")
    
    # Option 2: Upload CSV
    else:
        st.header("Chargement par fichier CSV")
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if all(col in df.columns for col in ['Utilisateur', 'Film', 'Note']):
                    # Nettoyage des noms de films
                    df['Film'] = df['Film'].str.strip()
                    # Conversion format long vers large
                    df_wide = df.pivot(index='Utilisateur', columns='Film', values='Note').reset_index()
                    df_wide.columns.name = None
                    
                    # Remplacer les NaN par 0
                    df_wide = df_wide.fillna(0)
                    
                    st.session_state.df = df_wide
                    st.info("Format long détecté - Conversion automatique en format large")
                    st.success("Fichier chargé avec succès!")
                else:
                    # Si format large directement
                    df = df.fillna(0)
                    st.session_state.df = df
                    st.success("Fichier chargé avec succès!")
                
            except Exception as e:
                st.error(f"Erreur de lecture: {str(e)}")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.subheader("📊 Matrice Utilisateur-Films")
        st.dataframe(df.set_index('Utilisateur'), use_container_width=True)
        
        if st.button("Calculer les similarités et recommandations"):
            movies = [col for col in df.columns if col != 'Utilisateur']
            n = len(movies)
            sim_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        sim_matrix[i][j] = 1
                    else:
                        sim = safe_pearson_sim(df, movies[i], movies[j])
                        sim_matrix[i][j] = sim
                        sim_matrix[j][i] = sim
            
            st.session_state.sim_matrix = pd.DataFrame(sim_matrix, index=movies, columns=movies)
            st.success("Matrice de similarité calculée!")
        
        if st.session_state.sim_matrix is not None:
            st.subheader("🔍 Matrice de Similarité")
            st.dataframe(st.session_state.sim_matrix, use_container_width=True)
            
            st.subheader("🎯 Recommandations Personnalisées")
            selected_user = st.selectbox("Sélectionner un utilisateur:", df['Utilisateur'].unique())
            n_recommendations = st.slider("Nombre de recommandations:", 1, 10, 3)
            
            if st.button("Générer les recommandations"):
                recommendations = get_recommendations(df, selected_user, st.session_state.sim_matrix, n_recommendations)
                
                if recommendations:
                    st.success(f"Top {n_recommendations} recommandations pour {selected_user}:")
                    for i, (movie, rating) in enumerate(recommendations, 1):
                        st.write(f"{i}. **{movie}** (note prédite: {rating:.2f}/5)")
                else:
                    st.warning("Pas assez de données pour générer des recommandations")
            
            st.subheader("🔎 Recherche Spécifique")
            col1, col2 = st.columns(2)
            search_user = col1.selectbox("Utilisateur:", df['Utilisateur'].unique(), key='search_user')
            available_movies = [col for col in df.columns if col != 'Utilisateur']
            search_movie = col2.selectbox("Film:", available_movies, key='search_movie')
            
            if st.button("Analyser cette combinaison"):
                try:
                    user_ratings = df.set_index('Utilisateur').loc[search_user]
                    
                    if user_ratings[search_movie] > 0:
                        st.warning(f"⚠️ {search_user} a déjà noté {search_movie}: {user_ratings[search_movie]}/5")
                    else:
                        similar_movies = st.session_state.sim_matrix[search_movie][st.session_state.sim_matrix[search_movie] > 0].index
                        rated_similar_movies = [m for m in similar_movies if m in user_ratings.index and user_ratings[m] > 0]
                        
                        if not rated_similar_movies:
                            st.error("Pas assez de données pour faire une prédiction fiable")
                        else:
                            numerator = sum(user_ratings[m] * st.session_state.sim_matrix.loc[search_movie, m] for m in rated_similar_movies)
                            denominator = sum(st.session_state.sim_matrix.loc[search_movie, m] for m in rated_similar_movies)
                            
                            if denominator > 0:
                                predicted_rating = numerator / denominator
                                st.success(f"**Note prédite pour {search_movie}:** {predicted_rating:.2f}/5")
                                
                                threshold = 3.0
                                if predicted_rating >= threshold:
                                    st.info("✅ **Recommandation:** OUI (score ≥ 3)")
                                else:
                                    st.info("❌ **Recommandation:** NON (score < 3)")
                                
                                st.write("**Basée sur la similarité avec:** " + ", ".join(rated_similar_movies))
                            else:
                                st.error("Pas assez de similarité pour faire une prédiction fiable")
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse: {str(e)}")

if __name__ == "__main__":
    main()