from sklearn.calibration import LabelEncoder
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
from collections import Counter
import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

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
model = load_model('genre_classification_model.keras')
preprocessor = joblib.load('preprocessor.pkl')
le = joblib.load('label_encoder.pkl')

def create_improved_genre_mapping():
    return {
        'ambient': 'Ambient/Chill',
        'chill': 'Ambient/Chill',
        'classical': 'Classical/Orchestral',
        'opera': 'Classical/Orchestral',
        'country': 'Country/Americana',
        'rock-n-roll': 'Country/Americana',
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
        'alt-rock': 'Rock/Metal',
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
        'psych-rock': 'Rock/Metal',
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

genre_unified_mapping = {
    'ambient': 'electronic',
    'chill': 'electronic',
    'classical': 'classical',
    'opera': 'classical',
    'country': 'country',
    'rock-n-roll': 'rock',
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

def create_reverse_mapping(genre_unified_mapping):
    reverse_mapping = {}
    for subgenre, unified in genre_unified_mapping.items():
        if unified not in reverse_mapping:
            reverse_mapping[unified] = set()
        reverse_mapping[unified].add(subgenre)
    return reverse_mapping

reverse_genre_mapping = create_reverse_mapping(genre_unified_mapping)

def get_unified_genre(spotify_genres):
    print(f"Spotify genres: {spotify_genres}")
    for genre in spotify_genres:
        genre_lower = genre.lower()
        print(f"Checking genre: {genre_lower}")
        
        # Check for exact matches
        if genre_lower in genre_unified_mapping:
            unified = genre_unified_mapping[genre_lower]
            print(f"Exact match found: {genre_lower} -> {unified}")
            return unified
        
        # Check for partial matches
        for key, value in genre_unified_mapping.items():
            if key in genre_lower:
                print(f"Partial match found: {genre_lower} contains {key} -> {value}")
                return value
        
        # Additional check for 'rock' specifically
        if 'rock' in genre_lower:
            print(f"'rock' found in {genre_lower}, mapping to 'rock'")
            return 'rock'
    
    print("No match found, returning 'other'")
    return 'other'

# # Print out the entire genre_unified_mapping for debugging
# print("\nFull genre_unified_mapping:")
# for key, value in genre_unified_mapping.items():
#     print(f"{key}: {value}")

# # Test the function
# test_genres = ['australian rock', 'hard rock', 'rock']
# result = get_unified_genre(test_genres)
# print(f"Final result: {result}")

def get_track_features(track_id):
    try:
        features = sp.audio_features(track_id)[0]
        track_info = sp.track(track_id)
        
        artist_id = track_info['artists'][0]['id']
        artist_info = sp.artist(artist_id)
        
        spotify_genres = artist_info['genres']
        print(f"Artist genres from Spotify: {spotify_genres}")
        
        unified_genre = get_unified_genre(spotify_genres)
        print(f"Unified genre: {unified_genre}")
        
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
            'time_signature': (features['time_signature'], '📝'),
            'unified_genre': (unified_genre, '🔠')
        }
        print(f"Combined info: {combined_info}")
        return combined_info
    
    except Exception as e:
        print(f"Error getting track features: {str(e)}")
        return None

    

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# List of all possible unified genres (from your feature names)
all_unified_genres = ['blues', 'classical', 'country', 'electronic', 'folk',
                      'funk_soul', 'hip-hop', 'house', 'jazz', 'latin', 'metal', 
                      'pop', 'reggae', 'rock', 'world']

def one_hot_encode_genre(unified_genre):
    # Initialize a dictionary for one-hot encoding
    genre_encoding = {f"cat__unified_genre_{genre}": 0 for genre in all_unified_genres}
    
    # Set the correct genre column to 1
    column_name = f"cat__unified_genre_{unified_genre}"
    if column_name in genre_encoding:
        genre_encoding[column_name] = 1

    print(genre_encoding)
    return genre_encoding

def preprocess_features(track_features):
    # Create the one-hot encoding for the unified genre
    genre_one_hot = one_hot_encode_genre(track_features['unified_genre'][0])  # Accessing the actual genre value
    
    # Extract and scale numerical features
    numerical_features = {
        'num__popularity': track_features['popularity'][0],
        'num__danceability': track_features['danceability'][0],
        'num__energy': track_features['energy'][0],
        'num__loudness': track_features['loudness'][0],
        'num__mode': 1 if track_features['mode'][0] == 'Major' else 0,
        'num__speechiness': track_features['speechiness'][0],
        'num__acousticness': track_features['acousticness'][0],
        'num__instrumentalness': track_features['instrumentalness'][0],
        'num__liveness': track_features['liveness'][0],
        'num__valence': track_features['valence'][0],
        'num__tempo': track_features['tempo'][0],
        'num__duration_ms': track_features['duration'][0],  # Use the raw duration in ms for consistency
        'num__time_signature': track_features['time_signature'][0],
    }
    
    # Combine numerical features with one-hot encoded genre
    combined_features = {**numerical_features, **genre_one_hot}
    
    # Convert to DataFrame
    combined_features_df = pd.DataFrame([combined_features])
    print(combined_features_df)
    return combined_features_df

def predict_genre(features):
    # Preprocess the features
    track_features = get_track_features(features)
    if track_features is None:
        return "Error", "Error", []

    X_preprocessed = preprocess_features(track_features)
    
    # Make predictions
    try:
        prediction_probs = model.predict(X_preprocessed)[0]
        
        # Get unique genres from label encoder
        unique_genres = le.classes_
        
        # Sort predictions by probability
        sorted_predictions = sorted(zip(unique_genres, prediction_probs), key=lambda x: x[1], reverse=True)
        
        # Get top 3 predictions
        top_predictions = sorted_predictions[:3]
        
        # If the highest probability is below a threshold, return "Unknown"
        if top_predictions[0][1] < 0.4:  # You can adjust this threshold
            return "Unknown", "Unknown", top_predictions
        
        predicted_category = top_predictions[0][0]
        
        return predicted_category, predicted_category, top_predictions
    
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return "Error", "Error", []


# # Add this to check the preprocessor and label encoder
# print("Preprocessor feature names:")
# print(preprocessor.get_feature_names_out())
# print("\nLabel Encoder classes:")
# print(le.classes_)

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