import streamlit as st
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

# Configure Spotify
client_credentials_manager = SpotifyClientCredentials(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load the trained model, preprocessor, and label encoder
model = tf.keras.models.load_model('genre_classification_model.keras')
preprocessor = joblib.load('preprocessor.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load scaler used for training
scaler = joblib.load('scaler.pkl')

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


def create_unified_genre_mapping():
    return {
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


improved_genre_mapping = create_improved_genre_mapping()
unified_genre_mapping = create_unified_genre_mapping()

def get_unified_genre(spotify_genres):
    genre_unified_mapping = {
        'ambient': 'electronic', 'chill': 'electronic', 'classical': 'classical',
        'country': 'country', 'dance': 'electronic', 'electronic': 'electronic',
        'house': 'house', 'folk': 'folk', 'funk': 'funk_soul', 'hip-hop': 'hip-hop',
        'jazz': 'jazz', 'latin': 'latin', 'metal': 'metal', 'pop': 'pop',
        'reggae': 'reggae', 'rock': 'rock', 'soul': 'funk_soul'
    }
    
    for genre in spotify_genres:
        genre_lower = genre.lower()
        for key, value in genre_unified_mapping.items():
            if key in genre_lower:
                return value
    return 'other'

def get_track_features(track_id):
    try:
        features = sp.audio_features(track_id)[0]
        track_info = sp.track(track_id)
        
        artist_id = track_info['artists'][0]['id']
        artist_info = sp.artist(artist_id)
        
        spotify_genres = artist_info['genres']
        unified_genre = get_unified_genre(spotify_genres)
        
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
        return None

def preprocess_api_data(track_data, scaler):
    """
    Preprocesses data obtained from Spotify API for a single track to match the model input format.
    
    :param track_data: DataFrame containing track data from Spotify API
    :param scaler: Pre-fitted Scaler object used for the training dataset
    :return: Pandas DataFrame row in the format required by the model
    """

    # Map genres
    track_data['mapped_genre'] = track_data['unified_genre'].map(improved_genre_mapping)
    track_data['unified_genre'] = track_data['unified_genre'].map(unified_genre_mapping)
    
    # Remove rows with unmapped genres
    track_data = track_data.dropna(subset=['mapped_genre', 'unified_genre'])

    if track_data.empty:
        raise ValueError("The genre of the track is not recognized and cannot be mapped.")
    
    # Scale numeric features
    numeric_features = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    
    track_data[numeric_features] = scaler.transform(track_data[numeric_features])
    print(track_data)
    return track_data

def preprocess_and_predict(new_data):
    try:
        X_new = preprocess_api_data(new_data, scaler)
        st.subheader("Raw Features:")
        st.write(X_new)
        
        X_new_preprocessed = preprocessor.transform(X_new)
        st.subheader("Preprocessed Features:")
        st.write(X_new_preprocessed)
        
        y_pred_probs = model.predict(X_new_preprocessed)
        y_pred_classes = y_pred_probs.argmax(axis=1)
        y_pred_labels = label_encoder.inverse_transform(y_pred_classes)
        predicted_genre = y_pred_labels[0]
        
        return predicted_genre, y_pred_probs[0]
    except Exception as e:
        st.error(f"Error in preprocessing and predicting: {str(e)}")
        return None, None

def main():
    st.title("Spotify Genre Classifier")

    track_url = st.text_input("Enter a Spotify track URL:")

    if st.button("Classify Genre"):
        if track_url:
            try:
                track_id = track_url.split('/')[-1].split('?')[0]
                track_features = get_track_features(track_id)

                if track_features is not None:
                    predicted_genre, prediction_probs = preprocess_and_predict(track_features)
                    
                    if predicted_genre:
                        st.subheader("Genre Classification Result:")
                        st.write(f"Predicted genre: '{predicted_genre}'")
                        
                        st.subheader("Prediction Probabilities:")
                        for genre, prob in zip(label_encoder.classes_, prediction_probs):
                            st.write(f"{genre}: {prob:.2%}")
                    else:
                        st.error("Failed to classify the genre.")
                else:
                    st.error("Failed to fetch track features.")
            except Exception as e:
                st.error(f"Error processing the track: {str(e)}")
        else:
            st.warning("Please enter a Spotify track URL.")

if __name__ == "__main__":
    main()
