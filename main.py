import streamlit as st
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

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

# Custom CSS for typewriter effect and styling
st.markdown("""
<style>
    .typewriter-text {
        overflow: hidden;
        border-right: .15em solid orange;
        white-space: nowrap;
        margin: 0;
        letter-spacing: .15em;
        animation: typing 3.5s steps(40, end), blink-caret .75s step-end infinite;
    }
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: orange; }
    }
    .stTextInput > div > div > input {
        caret-color: orange;
    }
    .stAudio > div > div {
        background-color: #1DB954 !important;
    }
    .info-text {
        font-size: 14px;
        margin-bottom: 5px;
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

def search_spotify(query, type='track,artist'):
    try:
        result = sp.search(q=query, type=type, limit=10)
        return result
    except Exception as e:
        st.error(f"Error searching Spotify: {str(e)}")
        return None

def get_artist_profile(artist_id):
    try:
        artist_info = sp.artist(artist_id)
        top_tracks = sp.artist_top_tracks(artist_id)
        return artist_info, top_tracks
    except Exception as e:
        st.error(f"Error getting artist profile: {str(e)}")
        return None, None

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

def main():
    st.title("Spotify Music Explorer")

    # Initialize session state
    if 'stage' not in st.session_state:
        st.session_state.stage = 'intro'
    if 'typed_text' not in st.session_state:
        st.session_state.typed_text = []

    if st.session_state.stage == 'intro':
        st.session_state.typed_text.append(typewriter_text("Welcome to Spotify Music Explorer! Here you can discover detailed information about artists and tracks."))
        time.sleep(1)
        st.session_state.typed_text.append(typewriter_text("Let's start by searching for an artist or a track."))
        time.sleep(1)
        st.session_state.stage = 'input'
        st.rerun()

    elif st.session_state.stage == 'input':
        # Display all previously typed text
        for text_container in st.session_state.typed_text:
            text_container.empty()
            text_container.markdown(text_container.markdown_text, unsafe_allow_html=True)

        query = st.text_input("Enter an artist or song name:", key="search_query")
        search_type = st.radio("Search for:", ("Artist", "Track"))

        if query:
            st.session_state.query = query
            st.session_state.search_type = search_type
            st.session_state.stage = 'search'
            st.rerun()

    elif st.session_state.stage == 'search':
        # Display all previously typed text
        for text_container in st.session_state.typed_text:
            text_container.empty()
            text_container.markdown(text_container.markdown_text, unsafe_allow_html=True)

        st.session_state.typed_text.append(typewriter_text("Searching the vast world of music... 🎵"))
        with st.spinner("Fetching results from Spotify..."):
            results = search_spotify(st.session_state.query, type=st.session_state.search_type.lower())
            
        if results:
            st.session_state.typed_text.append(typewriter_text("Great! Here's what I found for you:"))
            
            if st.session_state.search_type == "Artist":
                artists = results['artists']['items']
                if artists:
                    artist = artists[0]  # Get the first artist
                    artist_info, top_tracks = get_artist_profile(artist['id'])
                    
                    if artist_info:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(artist_info['images'][0]['url'] if artist_info['images'] else None, width=200)
                        with col2:
                            st.subheader(f"🎤 {artist_info['name']}")
                            st.write(f"Genres: {', '.join(artist_info['genres'])}")
                            st.write(f"🎉 Popularity: {artist_info['popularity']}/100")
                            st.write(f"👥 Followers: {artist_info['followers']['total']:,}")
                        
                        st.session_state.typed_text.append(typewriter_text("Check out the artist's top tracks:"))
                        for track in top_tracks['tracks'][:5]:
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.image(track['album']['images'][0]['url'] if track['album']['images'] else None, width=100)
                                st.audio(track['preview_url'])
                            with col2:
                                st.write(f"🎵 {track['name']}")
                                features = get_track_features(track['id'])
                                if features:
                                    display_info(features)
                            st.markdown("---")
                else:
                    st.session_state.typed_text.append(typewriter_text("Hmm, I couldn't find any artists with that name. Let's try a different search!"))
            else:
                tracks = results['tracks']['items']
                st.session_state.typed_text.append(typewriter_text("Here are some tracks that match your search:"))
                for track in tracks:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(track['album']['images'][0]['url'] if track['album']['images'] else None, width=200)
                        st.audio(track['preview_url'])
                    with col2:
                        st.subheader(f"🎵 {track['name']} by {track['artists'][0]['name']}")
                        features = get_track_features(track['id'])
                        if features:
                            display_info(features)
                    st.markdown("---")
        else:
            st.session_state.typed_text.append(typewriter_text("Oops! I couldn't find anything matching your search. Let's try something else!"))

        if st.button("Return to Search"):
            st.session_state.stage = 'input'
            st.session_state.typed_text = st.session_state.typed_text[:2]  # Keep only the intro messages
            st.rerun()

if __name__ == "__main__":
    main()