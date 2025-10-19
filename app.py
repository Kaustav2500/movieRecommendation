import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import ast

# Data Loading and Preprocessing
@st.cache_data
def load_and_prepare_data():
    """
    Loads the movie dataset from a URL, performs initial cleaning, 
    and extracts necessary features. This function is cached to run only once.
    """
    # Load dataset
    df = pd.read_csv("https://raw.githubusercontent.com/Kaustav2500/movieRecommendation/main/movies_recommendation.csv")
    
    # Initial cleaning
    df.drop('homepage', axis=1, inplace=True)
    df.dropna(inplace=True)

    # Helper functions to extract names from JSON-like strings
    def extract_first_name(column_str):
        try:
            items = ast.literal_eval(column_str)
            if isinstance(items, list) and items:
                return items[0].get('name', None)
        except (ValueError, SyntaxError):
            return None
    
    # Apply feature extraction
    df['main_production_company'] = df['production_companies'].apply(extract_first_name)
    df['main_production_country'] = df['production_countries'].apply(extract_first_name)
    df['main_spoken_language'] = df['spoken_languages'].apply(extract_first_name)
    
    return df

@st.cache_resource
def create_recommender(df):
    """
    Creates the feature matrix, trains a quality prediction model,
    calculates the cosine similarity matrix, and returns necessary objects.
    This entire resource-intensive process is cached.
    """
    # Create a copy to ensure the original DataFrame 
    df = df.copy()

    # Quality Prediction Model
    y_quality = (df['vote_average'] >= 6.5).astype(int)
    X_quality = df[['budget', 'popularity', 'revenue', 'runtime', 'vote_count']].fillna(0)
    
    param_grid = {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    gs = GridSearchCV(logreg, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_quality, y_quality)
    quality_pipeline = gs.best_estimator_
    
    quality_predictions = quality_pipeline.predict_proba(X_quality)[:, 1]
    df['ml_quality_score'] = quality_predictions

    # Content-Based Feature Engineering
    numerical_features = ['budget', 'revenue', 'popularity', 'runtime', 'vote_count', 'vote_average']
    numerical_data = df[numerical_features].fillna(0)
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_data)
    
    language_dummies = pd.get_dummies(df['original_language'], prefix='lang')
    country_dummies = pd.get_dummies(df['main_production_country'], prefix='country')
    spoken_lang_dummies = pd.get_dummies(df['main_spoken_language'], prefix='spoken')
    
    genre_vectorizer = TfidfVectorizer(max_features=50)
    genre_features = genre_vectorizer.fit_transform(df['genres'].fillna('')).toarray()
    
    keyword_vectorizer = TfidfVectorizer(max_features=100)
    keyword_features = keyword_vectorizer.fit_transform(df['keywords'].fillna('')).toarray()
    
    text_data = (df['overview'].fillna('') + ' ' + df['tagline'].fillna(''))
    text_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    text_features = text_vectorizer.fit_transform(text_data).toarray()
    
    company_dummies = pd.get_dummies(df['main_production_company'].fillna('Unknown'), prefix='company')
    
    roi = ((df['revenue'] - df['budget']) / (df['budget'].replace(0, 1))).fillna(0)
    roi_scaled = MinMaxScaler().fit_transform(roi.values.reshape(-1, 1))
    
    dates = pd.to_datetime(df['release_date'], errors='coerce')
    year_scaled = MinMaxScaler().fit_transform(dates.dt.year.fillna(dates.dt.year.median()).values.reshape(-1, 1))
    month_dummies = pd.get_dummies(dates.dt.month.fillna(0), prefix='month')

    # Combine all features with weights
    feature_matrix = np.hstack([
        numerical_scaled * 2.0,
        language_dummies.values * 1.5,
        genre_features * 3.0,
        keyword_features * 2.5,
        text_features * 1.0,
        country_dummies.values * 1.0,
        spoken_lang_dummies.values * 1.0,
        company_dummies.values * 0.8,
        roi_scaled * 1.2,
        year_scaled * 0.5,
        month_dummies.values * 0.3,
        df['ml_quality_score'].values.reshape(-1, 1) * 2.5
    ])
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(feature_matrix)
    
    # Create movie index mapping
    movie_indices = pd.Series(df.index, index=df['title'].str.lower()).to_dict()

    return similarity_matrix, movie_indices, df

# Streamlit UI

st.set_page_config(layout="wide")
st.title('🎬 Hybrid Movie Recommendation System')
st.write('This system combines content similarity with a machine learning quality score to provide smart recommendations.')
st.markdown("---")

# Load data and recommender system
movies_df_original = load_and_prepare_data()
similarity_matrix, movie_indices, movies_df_with_scores = create_recommender(movies_df_original)
movie_titles = movies_df_with_scores['title'].sort_values().tolist()

# User input on the main screen 
st.header("Select a Movie and Your Preferences")

# Use columns to center the input controls on the page
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    selected_movie_title = st.selectbox(
        'Choose a movie you like to get recommendations:',
        options=movie_titles,
        label_visibility="collapsed"
    )
    
    top_n = st.slider(
        'Number of Recommendations', 
        min_value=5, max_value=15, value=5
    )
    
    min_quality_score = st.slider(
        'Minimum Quality Score (Predicted)', 
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    
    recommend_button = st.button('Get Recommendations', use_container_width=True)


# Display Recommendations
if recommend_button:
    movie_name_lower = selected_movie_title.lower()
    
    if movie_name_lower not in movie_indices:
        st.error(f"Movie '{selected_movie_title}' not found in the database.")
    else:
        movie_idx = movie_indices[movie_name_lower]
        movie_data = movies_df_with_scores.iloc[movie_idx]
        
        sim_scores = list(enumerate(similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 30] # Search more to find quality matches

        # Filter and rank recommendations
        recommended_movies = []
        for idx, score in sim_scores:
            movie_quality = movies_df_with_scores.iloc[idx]['ml_quality_score']
            if (movies_df_with_scores.iloc[idx]['title'] != movie_data['title'] and 
                score > 0.1 and
                movie_quality >= min_quality_score):
                
                combined_score = (score * 0.7) + (movie_quality * 0.3)
                recommended_movies.append((idx, score, movie_quality, combined_score))
            
            if len(recommended_movies) >= top_n:
                break
        
        recommended_movies = sorted(recommended_movies, key=lambda x: x[3], reverse=True)[:top_n]
        
        st.markdown("---")
        
        if not recommended_movies:
            st.warning("No recommendations found with the selected quality score. Try lowering the 'Minimum Quality Score' slider.")
        else:
            st.subheader(f"Top {len(recommended_movies)} Recommendations for '{selected_movie_title}':")
            for i, (idx, sim, qual, comb) in enumerate(recommended_movies, 1):
                rec_movie = movies_df_with_scores.iloc[idx]
                with st.expander(f"**{i}. {rec_movie['title']}** (Combined Score: {comb:.0%})"):
                    st.write(f"**Rating:** {rec_movie['vote_average']:.1f}/10 ({rec_movie['vote_count']} votes)")
                    st.write(f"**Genre:** {rec_movie['genres']}")
                    st.write(f"**Content Similarity:** {sim:.0%}")
                    st.write(f"**Predicted Quality:** {qual:.0%}")
                    st.write(f"*Overview:* {rec_movie['overview']}")