from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import requests
from functools import lru_cache
import traceback
import json
from datetime import datetime
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import logging

logging.basicConfig(level=logging.WARNING)

# Cargar variables de entorno desde el archivo .env
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Configuración de Spotify
client_credentials_manager = SpotifyClientCredentials(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def test_spotify_credentials():
    try:
        result = sp.search(q='test', type='track', limit=1)
        if result:
            app.logger.info("Spotify credentials are working correctly")
        else:
            app.logger.error("Spotify credentials are not working")
    except Exception as e:
        app.logger.error(f"Error with Spotify credentials: {str(e)}")

test_spotify_credentials()

# Cargar el dataset local (si existe)
try:
    local_dataset = pd.read_csv(r'PROYECTO_FINAL\dataset_limpio2.csv')
    app.logger.info("Local dataset loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading local dataset: {str(e)}")
    local_dataset = None

@lru_cache(maxsize=100)
def fetch_spotify_data(method, *args, **kwargs):
    try:
        result = getattr(sp, method)(*args, **kwargs)
        return result
    except Exception as e:
        app.logger.error(f"Error fetching data from Spotify: {str(e)}")
        return None

def search_spotify(query, type='track,artist,album'):
    try:
        result = fetch_spotify_data('search', q=query, type=type, limit=10)
        return result if result else None
    except json.JSONDecodeError:
        app.logger.error(f"Error decoding JSON for Spotify search: {query}")
        return None

def search_playlists_for_track(track_name):
    try:
        # Buscar playlists por nombre de la pista
        result = fetch_spotify_data('search', q=f'playlist:{track_name}', type='playlist', limit=10)
        return result.get('playlists', {}).get('items', []) if result else []
    except Exception as e:
        app.logger.error(f"Error searching playlists for track '{track_name}': {str(e)}")
        return []

def get_track_details(track_id):
    try:
        track_info = fetch_spotify_data('track', track_id)
        audio_features_list = fetch_spotify_data('audio_features', track_id)
        
        # Verificar si la respuesta es una lista y contiene datos
        if not track_info or not audio_features_list or not isinstance(audio_features_list, list) or len(audio_features_list) == 0:
            return None
        
        audio_features = audio_features_list[0]  # Obtén el primer (y único) elemento de la lista
        
        # Busca playlists que contengan la canción
        playlists = search_playlists_for_track(track_info['name'])
        
        return {
            'track_info': track_info,
            'audio_features': audio_features,  # Incluye las características de audio
            'playlists': playlists  # Incluye la lista de playlists encontradas
        }
    except Exception as e:
        app.logger.error(f"Error getting track details from Spotify: {str(e)}")
        return None

def get_artist_details(artist_id):
    try:
        artist_info = fetch_spotify_data('artist', artist_id)
        top_tracks = fetch_spotify_data('artist_top_tracks', artist_id)
        albums = fetch_spotify_data('artist_albums', artist_id, album_type='album,single,compilation')
        related_artists = fetch_spotify_data('artist_related_artists', artist_id)
        
        return {
            'info': artist_info,
            'top_tracks': top_tracks,
            'albums': albums,
            'related_artists': related_artists
        }
    except Exception as e:
        app.logger.error(f"Error getting artist details from Spotify: {str(e)}")
        return None

def get_album_details(album_id):
    try:
        album_info = fetch_spotify_data('album', album_id)
        return album_info
    except Exception as e:
        app.logger.error(f"Error getting album details from Spotify: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_page')
def search_page():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    app.logger.info(f"Searching for: {query}")
    
    try:
        # Buscar en Spotify
        spotify_result = search_spotify(query)
        app.logger.info(f"Search result from Spotify: {spotify_result}")
        
        if not spotify_result:
            app.logger.info(f"No results found for: {query}")
            return jsonify({"error": "No se encontraron resultados."}), 404
        
        tracks = spotify_result.get('tracks', {}).get('items', [])
        artists = spotify_result.get('artists', {}).get('items', [])
        albums = spotify_result.get('albums', {}).get('items', [])
        
        results = []
        
        for track in tracks:
            track_details = get_track_details(track['id'])
            if track_details:
                track_info = track_details['track_info']
                audio_features = track_details['audio_features']
                playlists = track_details['playlists']
                
                artist_details = get_artist_details(track_info['artists'][0]['id'])
                album_details = get_album_details(track_info['album']['id'])
                
                result = {
                    'type': 'Track',
                    'id': track_info['id'],
                    'name': track_info['name'],
                    'popularity': track_info['popularity'],
                    'release_date': track_info['album']['release_date'],
                    'duration_ms': track_info['duration_ms'],
                    'explicit': track_info['explicit'],
                    'audio_features': {
                        'acousticness': audio_features.get('acousticness'),
                        'danceability': audio_features.get('danceability'),
                        'energy': audio_features.get('energy'),
                        'instrumentalness': audio_features.get('instrumentalness'),
                        'liveness': audio_features.get('liveness'),
                        'valence': audio_features.get('valence'),
                        'tempo': audio_features.get('tempo'),
                        'speechiness': audio_features.get('speechiness')
                    },
                    'artist': {
                        'name': artist_details['info']['name'],
                        'popularity': artist_details['info']['popularity'],
                        'genres': artist_details['info']['genres'],
                        'followers': artist_details['info']['followers']['total']
                    },
                    'album': {
                        'name': album_details['name'],
                        'type': album_details['album_type'],
                        'release_date': album_details['release_date']
                    },
                    'playlists': [{'id': pl['id'], 'name': pl['name'], 'external_urls': pl['external_urls']['spotify']} for pl in playlists],
                    'collaborations': [artist['name'] for artist in track_info['artists'][1:]]  # Excluyendo el artista principal
                }
                results.append(result)
        
        return jsonify(results)
    
    except Exception as e:
        app.logger.error(f"Error processing search request: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Error al procesar la búsqueda: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
