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
import datetime

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

# Add the genre mapping
genre_mapping = {
    'electronic': 'Electronic/Dance',
    'edm': 'Electronic/Dance',
    'electro': 'Electronic/Dance',
    'dance': 'Electronic/Dance',
    'club': 'Electronic/Dance',
    'breakbeat': 'Electronic/Dance',
    'drum-and-bass': 'Electronic/Dance',
    'dubstep': 'Electronic/Dance',
    'garage': 'Electronic/Dance',
    'hardstyle': 'Electronic/Dance',
    'house': 'Electronic/Dance',
    'chicago-house': 'Electronic/Dance',
    'deep-house': 'Electronic/Dance',
    'progressive-house': 'Electronic/Dance',
    'minimal-techno': 'Electronic/Dance',
    'techno': 'Electronic/Dance',
    'detroit-techno': 'Electronic/Dance',
    'trance': 'Electronic/Dance',
    'funk': 'Electronic/Dance',
    'disco': 'Electronic/Dance',
    'rock': 'Rock/Metal',
    'alt-rock': 'Rock/Metal',
    'hard-rock': 'Rock/Metal',
    'metal': 'Rock/Metal',
    'heavy-metal': 'Rock/Metal',
    'black-metal': 'Rock/Metal',
    'death-metal': 'Rock/Metal',
    'metalcore': 'Rock/Metal',
    'punk': 'Rock/Metal',
    'punk-rock': 'Rock/Metal',
    'emo': 'Rock/Metal',
    'goth': 'Rock/Metal',
    'grindcore': 'Rock/Metal',
    'hardcore': 'Rock/Metal',
    'industrial': 'Rock/Metal',
    'pop': 'Pop/Mainstream',
    'indie-pop': 'Pop/Mainstream',
    'k-pop': 'Pop/Mainstream',
    'power-pop': 'Pop/Mainstream',
    'cantopop': 'Pop/Mainstream',
    'hip-hop': 'Hip-Hop/R&B',
    'trip-hop': 'Hip-Hop/R&B',
    'soul': 'Hip-Hop/R&B',
    'jazz': 'Traditional',
    'blues': 'Traditional',
    'classical': 'Traditional',
    'opera': 'Traditional',
    'folk': 'Country/Folk',
    'acoustic': 'Country/Folk',
    'singer-songwriter': 'Country/Folk',
    'songwriter': 'Country/Folk',
    'country': 'Country/Folk',
    'rock-n-roll': 'Country/Folk',
    'afrobeat': 'World Music',
    'indian': 'World Music',
    'spanish': 'World Music',
    'french': 'World Music',
    'german': 'World Music',
    'swedish': 'World Music',
    'forro': 'World Music',
    'sertanejo': 'World Music',
    'salsa': 'Latin',
    'samba': 'Latin',
    'tango': 'Latin',
    'ambient': 'Ambient/Chill',
    'chill': 'Ambient/Chill',
    'dub': 'Reggae/Ska',
    'dancehall': 'Reggae/Ska',
    'ska': 'Reggae/Ska',
    'piano': 'Instrumental',
    'guitar': 'Instrumental',
    'romance': 'Miscellaneous',
    'sad': 'Miscellaneous',
    'pop-film': 'Miscellaneous'
}

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

def predict_genre(features):
    feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    feature_values = []
    for name in feature_names:
        value = features.get(name, (0, ''))[0]
        if name == 'mode':
            value = 1 if value == 'Major' else 0
        elif isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                value = 0
        feature_values.append(value)
    
    if not feature_values:
        return "Unknown", "Unknown", []

    X = np.array(feature_values).reshape(1, -1)
    
    # Pad the input to match the expected 30 features
    if scaler.n_features_in_ > X.shape[1]:
        padding = np.zeros((X.shape[0], scaler.n_features_in_ - X.shape[1]))
        X_padded = np.hstack((X, padding))
        X_scaled = scaler.transform(X_padded)
    else:
        X_scaled = scaler.transform(X)
    
    prediction_probs = model.predict(X_scaled)[0]
    
    # Get unique mapped genres (values in genre_mapping)
    unique_mapped_genres = list(set(genre_mapping.values()))
    
    # Sort predictions by probability
    sorted_predictions = sorted(zip(unique_mapped_genres, prediction_probs), key=lambda x: x[1], reverse=True)
    
    # Get top 3 predictions
    top_predictions = sorted_predictions[:3]
    
    # If the highest probability is below a threshold, return "Unknown"
    if top_predictions[0][1] < 0.4:  # You can adjust this threshold
        return "Unknown", "Unknown", top_predictions
    
    predicted_category = top_predictions[0][0]
    
    # Find a corresponding subgenre
    for subgenre, mapped_genre in genre_mapping.items():
        if mapped_genre == predicted_category:
            return subgenre, predicted_category, top_predictions
    
    return "Unknown", "Unknown", top_predictions

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
                api_genres = artist_info.get('genres', [])
                st.write(f"Genres: {', '.join(api_genres)}")
                st.write(f"🎉 Popularity: {artist_info['popularity']}/100")

            if track_info:
                display_info(track_info)
                
                predicted_genre, predicted_category, top_predictions = predict_genre(track_info)

                st.write(f"**Predicted Genre (Our Model):** {predicted_category} (Subgenre: {predicted_genre})")
                st.write("Top 3 predictions:")
                for genre, prob in top_predictions:
                    st.write(f"- {genre}: {prob:.2%}")
                
                if api_genres:
                    final_genre = api_genres[0] if predicted_genre.lower() not in [g.lower() for g in api_genres] else predicted_genre
                    st.write(f"**Final Genre Prediction:** {final_genre.capitalize()}")
                    
                    # Get similar genres from related artists
                    similar_genres = get_similar_genres(track['artists'][0]['id'])
                    st.markdown("**Similar Genres:**")
                    for genre in similar_genres:
                        st.markdown(f"<font color='blue'>{genre.capitalize()}</font>", unsafe_allow_html=True)
                    
                    release_date = track['album']['release_date']
                    decade = get_decade(release_date)
                    st.write(f"**Decade:** {decade}")
                else:
                    st.write(f"**Final Genre Prediction:** {predicted_genre.capitalize()}")
        
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