import streamlit as st
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from collections import Counter

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

# Load the trained model and scaler
model = load_model('genre_classification_model.h5')
scaler = joblib.load('scaler.pkl')

# Custom CSS for styling
st.markdown("""
<style>
    .typewriter-text {
        overflow: hidden;
        white-space: nowrap;
        margin: 0;
        letter-spacing: .15em;
        animation: typing 3.5s steps(40, end);
    }
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    .stTextInput > div > div > input {
        caret-color: transparent;
        background: transparent;
        border: none;
        outline: none;
        color: inherit;
    }
    .stAudio > div > div {
        background-color: #1DB954 !important;
    }
    .info-text {
        font-size: 14px;
        margin-bottom: 5px;
    }
    .artist-photo {
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

def typewriter_text(text, speed=0.05):
    container = st.empty()
    for i in range(len(text) + 1):
        displayed_text = f'<div class="typewriter-text">{text[:i]}</div>'
        container.markdown(displayed_text, unsafe_allow_html=True)
        time.sleep(speed)
    return container

def search_spotify(query, type='track'):
    try:
        result = sp.search(q=query, type=type, limit=10)
        return result
    except Exception as e:
        st.error(f"Error searching Spotify: {str(e)}")
        return None

def get_track_features(track_id):
    try:
        features = sp.audio_features(track_id)[0]
        track_info = sp.track(track_id)
        
        combined_info = {
            'popularity': (track_info['popularity'], '🎉'),
            'danceability': (features['danceability'], '💃'),
            'energy': (features['energy'], '⚡'),
            'key': (features['key'], '🎹'),
            'loudness': (features['loudness'], '🔊'),
            'mode': ('Major' if features['mode'] == 1 else 'Minor', '🎼'),
            'speechiness': (features['speechiness'], '🗣️'),
            'acousticness': (features['acousticness'], '🎸'),
            'instrumentalness': (features['instrumentalness'], '🎺'),
            'liveness': (features['liveness'], '🎭'),
            'valence': (features['valence'], '😊'),
            'tempo': (round(features['tempo'], 2), '🏃'),
            'duration': (f"{features['duration_ms'] // 60000}:{(features['duration_ms'] % 60000 // 1000):02d}", '⏱️'),
            'time_signature': (features['time_signature'], '📝')
        }
        return combined_info
    except Exception as e:
        st.error(f"Error getting track features: {str(e)}")
        return None

def display_info(info):
    for key, (value, icon) in info.items():
        st.markdown(f'<p class="info-text">{icon} {key.capitalize()}: {value}</p>', unsafe_allow_html=True)

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
    
# La función predict_genre ahora devuelve directamente la clasificación del track que deduce el modelo.
def predict_genre(features):
    feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    feature_values = []
    for name in feature_names:
        value = features.get(name, (0, ''))[0]  # Use get() method with a default value
        if name == 'mode':
            value = 1 if value == 'Major' else 0
        elif isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                value = 0  # or some default value
        feature_values.append(value)
    
    if not feature_values:  # Check if the list is empty
        return "Unknown"  # Return a default genre if we don't have any features
    
    X = np.array(feature_values).reshape(1, -1)
    
    # Ajuste para manejar la discrepancia en el número de características
    if scaler.n_features_in_ > X.shape[1]:
        padding = np.zeros((X.shape[0], scaler.n_features_in_ - X.shape[1]))
        X_padded = np.hstack((X, padding))
        X_scaled = scaler.transform(X_padded)
    else:
        X_scaled = scaler.transform(X[:, :scaler.n_features_in_])
    
    prediction = model.predict(X_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    genre_names = ['Electronic', 'Anime', 'Jazz', 'Alternative', 'Country', 'Rap', 'Blues', 'Classical', 'Hip-Hop', 'Reggae', 'Rock', 'Pop', 'Metal']
    predicted_genre = genre_names[predicted_class]
    
    return predicted_genre

#La función combine_genre_predictions ha sido modificada para implementar la lógica solicitada:

# Si hay una coincidencia exacta entre la predicción del modelo y los géneros de la API, se usa la predicción del modelo.
# Si no hay coincidencia exacta, se buscan coincidencias parciales en los géneros de la API.
# Si no hay coincidencias, se usa el primer género proporcionado por la API (si existe).
# Si la API no proporciona géneros, se usa la predicción del modelo.
def combine_genre_predictions(model_prediction, api_genres):
    # Lista de géneros que nuestro modelo puede predecir
    model_genres = ['electronic', 'anime', 'jazz', 'alternative', 'country', 'rap', 'blues', 'classical', 'hip-hop', 'reggae', 'rock', 'pop', 'metal']
    
    # Convertir géneros de la API a minúsculas para comparación
    api_genres_lower = [genre.lower() for genre in api_genres]
    
    # Si hay una coincidencia exacta entre la predicción del modelo y los géneros de la API, usar la predicción del modelo
    if model_prediction.lower() in api_genres_lower:
        return model_prediction
    
    # Si no hay coincidencia exacta, buscar coincidencias parciales en los géneros de la API
    for genre in api_genres_lower:
        if any(model_genre in genre for model_genre in model_genres):
            return genre.capitalize()
    
    # Si no hay coincidencias, usar el primer género de la API (si existe)
    if api_genres:
        return api_genres[0].capitalize()
    
    # Si no hay géneros de la API, usar la predicción del modelo
    return model_prediction

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
        track_info = get_track_features(track['id'])

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(track['album']['images'][0]['url'] if track['album']['images'] else None, width=200)
            st.audio(track['preview_url'])
        with col2:
            st.subheader(f"🎵 {track['name']} by {track['artists'][0]['name']}")
            
            st.write(f"**Main Artist:** {track['artists'][0]['name']}")
            
            if artist_info:
                api_genres = artist_info.get('genres', [])  # Use get() method with a default empty list
                st.write(f"Genres: {', '.join(api_genres)}")
                st.write(f"🎉 Popularity: {artist_info['popularity']}/100")
            if track_info:
                display_info(track_info)
                
                # Predict genre using our model
                predicted_genre = predict_genre(track_info)
                st.write(f"**Predicted Genre (Our Model):** {predicted_genre}")
                
                # Combine predictions
                if api_genres:
                    combined_genre = combine_genre_predictions(predicted_genre, api_genres)
                    st.write(f"**Final Genre Prediction:** {combined_genre}")
                else:
                    st.write(f"**Final Genre Prediction:** {predicted_genre}")
        
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
                for artist in related_artists['artists']:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(artist['images'][0]['url'] if artist['images'] else None, width=100, caption=artist['name'])
                    with col2:
                        st.write(f"**Name:** {artist['name']}")
                        st.write(f"**Genres:** {', '.join(artist['genres'])}")
            else:
                st.write("No related artists found.")

        with st.expander("Genres of Related Artists"):
            if related_artists and related_artists['artists']:
                related_genres = set()
                for related_artist in related_artists['artists']:
                    related_genres.update(related_artist['genres'])
                st.write(', '.join(related_genres) if related_genres else "No genres found for related artists.")
            else:
                st.write("No related artists to show genres for.")
        
        if st.button("Back to Results"):
            st.session_state.stage = 'results'
            st.rerun()

        if st.button("Return to Search", on_click=return_to_search):
            st.rerun()

if __name__ == "__main__":
    main()