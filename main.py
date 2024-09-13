import streamlit as st
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set page config at the very beginning
st.set_page_config(page_title="Spotify Music Explorer", page_icon="🎵", layout="wide")

# Load environment variables
load_dotenv()

# Configure Spotify
client_credentials_manager = SpotifyClientCredentials(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load the trained model, preprocessor, and label encoder
model = load_model('genre_classification_model.h5')
preprocessor = joblib.load('preprocessor.pkl')
le = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

st.markdown("""
<style>
    /* Fondo general con gradiente basado en la nueva paleta */
    .main {
        background: linear-gradient(135deg, #000000, #2B2B2B);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Cambia el color del texto de los títulos y párrafos a blanco */
    h1, h2, p {
        color: white;
    }

    /* Específicamente el texto "Spotify Music Explorer" y "Enter a track name:" */
    .title-text {
        color: white !important;
    }

    /* Texto de entrada */
    .stTextInput > div > div > input {
        background: transparent;
        border: 2px solid #FF0000;
        padding: 10px;
        color: black;
        font-size: 18px;
        border-radius: 10px;
        transition: border-color 0.3s;
    }
    
    /* Cambia color del borde en hover */
    .stTextInput > div > div > input:hover {
        border-color: #FFFFFF;
    }

    /* Texto del typewriter */
    .typewriter-text {
        overflow: hidden;
        white-space: nowrap;
        margin: 20px 0;
        font-size: 20px;
        color: #FFFFFF;
        letter-spacing: .15em;
        animation: typing 3.5s steps(40, end), blink-caret .75s step-end infinite;
    }

    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    
    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: white; }
    }

    /* Diseño para los resultados de búsqueda */
    .result-card {
        padding: 15px;
        background: rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
        border-radius: 10px;
        transition: background 0.3s ease, transform 0.2s ease;
        cursor: pointer;
    }

    .result-card:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.02);
    }

    /* Texto de los resultados */
    .result-card h3 {
        font-size: 24px;
        margin-bottom: 5px;
        color: #FF0000;
    }

    .result-card p {
        margin: 0;
        font-size: 16px;
        color: #FFFFFF;
    }

    /* Estilo de la información de la canción seleccionada */
    .song-details {
        background: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .song-details h2 {
        font-size: 34px;
        color: #FF0000;
        margin-bottom: 10px;
    }

    .song-details p {
        font-size: 18px;
        color: white;
        margin-bottom: 5px;
    }

    /* Estilo de botones */
    .stButton > button {
        background-color: #FF0000;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #FF0000; /* Keep the red background on hover */
        color: white !important; /* Ensure text remains white on hover */
        border: 2px solid white; /* Add a white border for contrast */
    }

</style>
""", unsafe_allow_html=True)



# Helper functions (combining from both files)

def create_improved_genre_mapping():
    return {
        'ambient': 'Ambient/Chill',
        'chill': 'Ambient/Chill',
        'classical': 'Classical/Orchestral',
        'opera': 'Classical/Orchestral',
        'country': 'Country/Americana',
        'rock': 'Country/Americana',
        'breakbeat': 'Electronic/Dance',
        'club': 'Electronic/Dance',
        'dance': 'Electronic/Dance',
        'deep-house': 'Electronic/Dance',
        'drum-and-bass': 'Electronic/Dance',
        'dubstep': 'Electronic/Dance',
        'edm': 'Electronic/Dance',
        'electro': 'Electronic/Dance',
        'electronic': 'Electronic/Dance',
        'garage': 'Electronic/Dance',
        'hardstyle': 'Electronic/Dance',
        'house': 'Electronic/Dance',
        'minimal-techno': 'Electronic/Dance',
        'progressive-house': 'Electronic/Dance',
        'techno': 'Electronic/Dance',
        'trance': 'Electronic/Dance',
        'acoustic': 'Folk/Acoustic',
        'folk': 'Folk/Acoustic',
        'singer-songwriter': 'Folk/Acoustic',
        'disco': 'Funk/Disco',
        'funk': 'Funk/Disco',
        'hip-hop': 'Hip-Hop/R&B',
        'soul': 'Hip-Hop/R&B',
        'trip-hop': 'Hip-Hop/R&B',
        'guitar': 'Instrumental',
        'piano': 'Instrumental',
        'blues': 'Jazz/Blues',
        'jazz': 'Jazz/Blues',
        'salsa': 'Latin',
        'samba': 'Latin',
        'tango': 'Latin',
        'cantopop': 'Pop/Mainstream',
        'indie-pop': 'Pop/Mainstream',
        'k-pop': 'Pop/Mainstream',
        'pop': 'Pop/Mainstream',
        'power-pop': 'Pop/Mainstream',
        'dancehall': 'Reggae/Ska',
        'dub': 'Reggae/Ska',
        'ska': 'Reggae/Ska',
        'alt': 'Rock/Metal',
        'black-metal': 'Rock/Metal',
        'death-metal': 'Rock/Metal',
        'emo': 'Rock/Metal',
        'goth': 'Rock/Metal',
        'grindcore': 'Rock/Metal',
        'hard-rock': 'Rock/Metal',
        'hardcore': 'Rock/Metal',
        'heavy-metal': 'Rock/Metal',
        'industrial': 'Rock/Metal',
        'metal': 'Rock/Metal',
        'metalcore': 'Rock/Metal',
        'psych': 'Rock/Metal',
        'punk': 'Rock/Metal',
        'punk-rock': 'Rock/Metal',
        'afrobeat': 'World Music',
        'forro': 'World Music',
        'french': 'World Music',
        'german': 'World Music',
        'indian': 'World Music',
        'sertanejo': 'World Music',
        'spanish': 'World Music',
        'swedish': 'World Music'
    }

genre_mapping = create_improved_genre_mapping()

def create_unified_genre_mapping():
    return {
        'ambient': 'electronic',
        'chill': 'electronic',
        'classical': 'classical',
        'opera': 'classical',
        'country': 'country',
        'rock': 'rock',
        'breakbeat': 'electronic',
        'club': 'electronic',
        'dance': 'electronic',
        'deep-house': 'house',
        'drum-and-bass': 'electronic',
        'dubstep': 'electronic',
        'edm': 'electronic',
        'electro': 'electronic',
        'electronic': 'electronic',
        'garage': 'electronic',
        'hardstyle': 'electronic',
        'house': 'house',
        'minimal-techno': 'electronic',
        'progressive-house': 'house',
        'techno': 'electronic',
        'trance': 'electronic',
        'acoustic': 'folk',
        'folk': 'folk',
        'singer-songwriter': 'folk',
        'disco': 'funk_soul',
        'funk': 'funk_soul',
        'hip-hop': 'hip-hop',
        'soul': 'funk_soul',
        'trip-hop': 'hip-hop',
        'guitar': 'rock',
        'piano': 'classical',
        'blues': 'blues',
        'jazz': 'jazz',
        'salsa': 'latin',
        'samba': 'latin',
        'tango': 'latin',
        'cantopop': 'pop',
        'indie-pop': 'pop',
        'k-pop': 'pop',
        'pop': 'pop',
        'power-pop': 'pop',
        'dancehall': 'reggae',
        'dub': 'reggae',
        'ska': 'reggae',
        'alt-rock': 'rock',
        'black-metal': 'metal',
        'death-metal': 'metal',
        'emo': 'rock',
        'goth': 'rock',
        'grindcore': 'metal',
        'hard-rock': 'rock',
        'hardcore': 'metal',
        'heavy-metal': 'metal',
        'industrial': 'metal',
        'metal': 'metal',
        'metalcore': 'metal',
        'psych-rock': 'rock',
        'punk': 'rock',
        'punk-rock': 'rock',
        'afrobeat': 'world',
        'forro': 'world',
        'french': 'world',
        'german': 'world',
        'indian': 'world',
        'sertanejo': 'world',
        'spanish': 'world',
        'swedish': 'world'
    }

unified_genre_mapping = create_unified_genre_mapping()

def map_unified_genre_to_label(unified_genre):
    genre_mapping = create_improved_genre_mapping()
    unified_genre_mapping = create_unified_genre_mapping()
    return genre_mapping.get(unified_genre_mapping.get(unified_genre), 'Unknown Genre')



# print("Preprocessor feature names:")
# print(preprocessor.get_feature_names_out())
# print("\nLabel Encoder classes:")
# print(le.classes_)



def get_unified_genre(spotify_genres):
    print(f"Received Spotify genres: {spotify_genres}")  # Debug print
    for genre in spotify_genres:
        genre_lower = genre.lower()

        # Check if genre matches a key in the mapping
        for key, value in unified_genre_mapping.items():
            if key in genre_lower:
                print(f"Mapped genre '{genre}' to unified genre '{value}' (matched key '{key}')")  # Debug print
                return value

        # Check if genre matches a value in the mapping
        for value in set(unified_genre_mapping.values()):  # Use a set to avoid duplicate checks
            if value in genre_lower:
                print(f"Genre '{genre}' matches unified genre '{value}' (matched value)")  # Debug print
                return value

    print("No unified genre found, defaulting to 'other'")  # Debug print
    return 'other'


def get_track_features(track_id):
    try:
        # print(f"Fetching features for track ID: {track_id}")  # Debug print
        features = sp.audio_features(track_id)[0]
        track_info = sp.track(track_id)
        
        artist_id = track_info['artists'][0]['id']
        artist_info = sp.artist(artist_id)
        
        spotify_genres = artist_info['genres']
        unified_genre = get_unified_genre(spotify_genres)
        
        # print(f"Track info: {track_info}")  # Debug print
        print(f"Artist genres: {spotify_genres}")  # Debug print
        print(f"Unified genre: {unified_genre}")  # Debug print
        
        return pd.DataFrame({
            'popularity': [track_info['popularity']],
            'danceability': [features['danceability']],
            'energy': [features['energy']],
            'key': [features['key']],
            'loudness': [features['loudness']],
            'mode': [features['mode']],
            'speechiness': [features['speechiness']],
            'acousticness': [features['acousticness']],
            'instrumentalness': [features['instrumentalness']],
            'liveness': [features['liveness']],
            'valence': [features['valence']],
            'tempo': [features['tempo']],
            'duration_ms': [features['duration_ms']],
            'time_signature': [features['time_signature']],
            'unified_genre': [unified_genre]
        })
    
    except Exception as e:
        st.error(f"Error getting track features: {str(e)}")
        print(f"Error in get_track_features: {str(e)}")  # Debug print
        return None


def preprocess_api_data(track_data, scaler):
    print(f"Original track data: {track_data}")  # Debug print

    # Map the unified genre to the improved genre categories
    track_data['mapped_genre'] = track_data['unified_genre'].map(unified_genre_mapping)
    # print(['unified_genre'].unique())
    # Handle unmapped genres by printing the issue and raising an error
    unmapped_genres = track_data[track_data['mapped_genre'].isna()]['unified_genre'].unique()
    if len(unmapped_genres) > 0:
        print(f"Unmapped unified genres: {unmapped_genres}")  # Debug print
        raise ValueError(f"The genre(s) {unmapped_genres} of the track are not recognized and cannot be mapped.")
    
    print(f"Data after genre mapping: {track_data}")  # Debug print
    
    # Scale numeric features
    numeric_features = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    
    track_data[numeric_features] = scaler.transform(track_data[numeric_features])
    print(f"Data after scaling: {track_data}")  # Debug print
    return track_data


def predict_genre(track_id):
    try:
        # print(f"Predicting genre for track ID: {track_id}")  # Debug print
        track_features = get_track_features(track_id)
        if track_features is None:
            return "Error", "Error", []

        X_new = preprocess_api_data(track_features, scaler)
        print(f"Preprocessed data: {X_new}")  # Debug print
        
        X_new_preprocessed = preprocessor.transform(X_new)
        print(f"Data after preprocessing: {X_new_preprocessed}")  # Debug print
        
        y_pred_probs = model.predict(X_new_preprocessed)
        y_pred_classes = y_pred_probs.argmax(axis=1)
        y_pred_labels = le.inverse_transform(y_pred_classes)
        predicted_genre = y_pred_labels[0]
        
        print(f"Prediction probabilities: {y_pred_probs}")  # Debug print
        print(f"Predicted genre: {predicted_genre}")  # Debug print
        
        # Sort predictions by probability
        sorted_predictions = sorted(zip(le.classes_, y_pred_probs[0]), key=lambda x: x[1], reverse=True)
        
        # Get top 3 predictions
        top_predictions = sorted_predictions[:3]
        
        print(f"Top predictions: {top_predictions}")  # Debug print
        
        return predicted_genre, predicted_genre, top_predictions
    
    except Exception as e:
        st.write(f"Genre outside limits of data: {str(e)}")
        # print(f"Error in predict_genre: {str(e)}")  # Debug print
        return "Error", "Error", []



def search_spotify(query, type='track'):
    try:
        result = sp.search(q=query, type=type, limit=10)
        return result
    except Exception as e:
        st.error(f"Error searching Spotify: {str(e)}")
        return None

def typewriter_text(text, speed=0.05):
    container = st.empty()
    for i in range(len(text) + 1):
        displayed_text = f'<div class="typewriter-text">{text[:i]}</div>'
        container.markdown(displayed_text, unsafe_allow_html=True)
        time.sleep(speed)
    return container

def display_info(info):
    for key, value in info.items():
        st.markdown(f'<p class="info-text">{key.capitalize()}: {value}</p>', unsafe_allow_html=True)

def get_track_collaborations(track_artists):
    collaborators = set()
    for artist in track_artists:
        collaborators.add(artist['name'])
    return collaborators

def get_artist_collaborations_history(artist_id, limit=10):
    try:
        top_tracks = sp.artist_top_tracks(artist_id)['tracks']
        collaborations_history = []
        
        for track in top_tracks:
            for artist in track['artists']:
                if artist['id'] != artist_id and len(collaborations_history) < limit:
                    collaborations_history.append(artist['name'])
        
        return collaborations_history[:limit]
    except Exception as e:
        st.error(f"Error fetching historical collaborations: {str(e)}")
        return []

def get_decade(release_date):
    try:
        if len(release_date) == 4:  # Year only
            year = int(release_date)
        else:  # Full date
            year = datetime.datetime.strptime(release_date, "%Y-%m-%d").year
        return f"{(year // 10) * 10}s"
    except ValueError:
        return "Unknown"

def get_similar_genres(artist_id):
    related_artists = sp.artist_related_artists(artist_id)['artists']
    all_genres = [genre for artist in related_artists for genre in artist['genres']]
    return list(set(all_genres))[:5]  # Return up to 5 unique genres

def display_info_without_unified_genre(info):
    filtered_info = {k: v for k, v in info.items() if k != 'unified_genre'}
    for key, value in filtered_info.items():
        st.markdown(f'<p class="info-text">{key.capitalize()}: {value}</p>', unsafe_allow_html=True)

def main():
    st.title("Spotify Music Explorer")

    if 'stage' not in st.session_state:
        st.session_state.stage = 'input'
    if 'query' not in st.session_state:
        st.session_state.query = ''
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None

    def return_to_search():
        st.session_state.stage = 'input'
        st.session_state.query = ''
        st.session_state.search_results = None

    if st.session_state.stage == 'input':
        query = st.text_input("Enter a track name:", value=st.session_state.query, key="search_query")

        if st.button("Search"):
            if query:
                st.session_state.query = query
                st.session_state.stage = 'searching'
                st.rerun()

    elif st.session_state.stage == 'searching':
        st.empty()
        
        container = typewriter_text("Searching the vast world of music... 🎵")
        time.sleep(1)
        container.empty()
        container = typewriter_text("Fetching results from Spotify...")
        
        results = search_spotify(st.session_state.query, type='track')
        st.session_state.search_results = results
        st.session_state.stage = 'results'
        st.rerun()

    elif st.session_state.stage == 'results':
        if st.session_state.search_results and st.session_state.search_results['tracks']['items']:
            st.subheader("Search Results:")
            for i, track in enumerate(st.session_state.search_results['tracks']['items'], 1):
                artist_names = ", ".join([artist['name'] for artist in track['artists']])
                if st.button(f"{i}. {track['name']} by {artist_names}", key=f"track_{i}"):
                    st.session_state.selected_track = track
                    st.session_state.stage = 'details'
                    st.rerun()
        else:
            st.write("No tracks found matching your search. Please try again.")
        
        if st.button("Return to Search", on_click=return_to_search):
            st.rerun()

    elif st.session_state.stage == 'details':
        track = st.session_state.selected_track
        artist_info = sp.artist(track['artists'][0]['id'])
        track_features = get_track_features(track['id'])

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(track['album']['images'][0]['url'] if track['album']['images'] else None, width=200)
            st.audio(track['preview_url'])
        with col2:
            st.subheader(f"🎵 {track['name']} by {track['artists'][0]['name']}")
            
            st.write(f"**Main Artist:** {track['artists'][0]['name']}")

            if artist_info:
                api_genres = artist_info.get('genres', [])
                st.write(f"Genres: {', '.join(api_genres)}")
                # st.write(f"🎉 Popularity: {artist_info['popularity']}/100")

                if track_features is not None:

                    display_info_without_unified_genre(track_features.iloc[0].to_dict())
                    
                    predicted_genre, predicted_category, top_predictions = predict_genre(track['id'])

                    st.write(f"**Classified Genre (Our Model):** {predicted_category}")
                    st.write("Top 3 classifications:")
                    for genre, prob in top_predictions:
                        st.write(f"- {genre}: {prob:.2%}")
                
                if api_genres:
                    final_genre = api_genres[0] if predicted_genre.lower() not in [g.lower() for g in api_genres] else predicted_genre
                    st.write(f"**Final Genre Classification:** {final_genre.capitalize()}")
                    
                    similar_genres = get_similar_genres(track['artists'][0]['id'])
                    st.markdown("**Similar Genres:**")
                    for genre in similar_genres:
                        st.markdown(f"<font color='white'>{genre.capitalize()}</font>", unsafe_allow_html=True)
                    
                    release_date = track['album']['release_date']
                    decade = get_decade(release_date)
                    st.write(f"**Decade:** {decade}")
                else:
                    st.write(f"**Final Genre Classification:** {predicted_genre.capitalize()}")
        
        with st.expander("Collaborations"):
            current_collaborations = get_track_collaborations(track['artists'])
            if len(current_collaborations) > 1:
                st.write("Collaborators on this track: " + ', '.join(current_collaborations))
            else:
                st.write("No collaborators found for this track.")

            artist_collaborations_history = get_artist_collaborations_history(track['artists'][0]['id'])
            if artist_collaborations_history:
                st.write("Examples of other collaborations by this artist: " + ', '.join(artist_collaborations_history))
            else:
                st.write("No historical collaborations found.")

        with st.expander("Related Artists"):
            related_artists = sp.artist_related_artists(track['artists'][0]['id'])
            if related_artists and related_artists['artists']:
                for artist in related_artists['artists'][:5]:  # Limit to 5 related artists
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(artist['images'][0]['url'] if artist['images'] else None, width=100, caption=artist['name'])
                    with col2:
                        st.write(f"**Name:** {artist['name']}")
                        st.write(f"**Genres:** {', '.join(artist['genres'])}")
            else:
                st.write("No related artists found.")

        if st.button("Back to Results"):
            st.session_state.stage = 'results'
            st.rerun()

        if st.button("Return to Search", on_click=return_to_search):
            st.rerun()

if __name__ == "__main__":
    main()