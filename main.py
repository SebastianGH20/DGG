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
        white-space: nowrap;
        margin: 0;
        letter-spacing: .15em;
        animation: typing 3.5s steps(40, end);
    }
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    /* Remove caret and border */
    .stTextInput > div > div > input {
        caret-color: transparent; /* Hide the caret */
        background: transparent; /* Optional: make the background transparent */
        border: none; /* Remove the border */
        outline: none; /* Remove the outline */
        color: inherit; /* Inherit text color */
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
        related_artists = sp.artist_related_artists(artist_id)
        return artist_info, top_tracks, related_artists
    except Exception as e:
        st.error(f"Error getting artist profile: {str(e)}")
        return None, None, None

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

def get_artist_collaborations(artist_id):
    """Get all unique collaborators from the artist's top tracks."""
    try:
        top_tracks = sp.artist_top_tracks(artist_id)['tracks']
        collaborators = set()
        for track in top_tracks:
            for artist in track['artists']:
                if artist['id'] != artist_id:
                    collaborators.add(artist['name'])
        return collaborators
    except Exception as e:
        st.error(f"Error fetching artist collaborations: {str(e)}")
        return set()

def get_track_collaborations(track_artists):
    """Get all unique collaborators from the track's artists."""
    collaborators = set()
    for artist in track_artists:
        collaborators.add(artist['name'])
    return collaborators

def main():
    st.title("Spotify Music Explorer")

    # Initialize session state
    if 'stage' not in st.session_state:
        st.session_state.stage = 'intro'
    if 'query' not in st.session_state:
        st.session_state.query = ''
    if 'search_type' not in st.session_state:
        st.session_state.search_type = 'Artist'
    
    if st.session_state.stage == 'intro':
        typewriter_text("Welcome to Spotify Music Explorer! Here you can discover detailed information about artists and tracks.")
        time.sleep(1)
        typewriter_text("Let's start by searching for an artist or a track.")
        time.sleep(1)
        st.session_state.stage = 'input'

    if st.session_state.stage == 'input':
        query = st.text_input("Enter an artist or song name:", value=st.session_state.query, key="search_query")
        search_type = st.radio("Search for:", ("Artist", "Track"), index=["Artist", "Track"].index(st.session_state.search_type))

        if st.button("Search"):
            if query:
                st.session_state.query = query
                st.session_state.search_type = search_type
                st.session_state.stage = 'search'

    if st.session_state.stage == 'search':
        typewriter_text("Searching the vast world of music... 🎵")
        with st.spinner("Fetching results from Spotify..."):
            results = search_spotify(st.session_state.query, type=st.session_state.search_type.lower())
            
        if results:
            typewriter_text("Great! Here's what I found for you:")
            
            if st.session_state.search_type == "Artist":
                artists = results['artists']['items']
                if artists:
                    artist = artists[0]  # Get the first artist
                    artist_info, top_tracks, related_artists = get_artist_profile(artist['id'])
                    
                    if artist_info:
                        # Artist Information Block
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(artist_info['images'][0]['url'] if artist_info['images'] else None, width=200)
                        with col2:
                            st.subheader(f"🎤 {artist_info['name']}")
                            st.write(f"Genres: {', '.join(artist_info['genres'])}")
                            st.write(f"🎉 Popularity: {artist_info['popularity']}/100")
                            st.write(f"👥 Followers: {artist_info['followers']['total']:,}")

                        # Collaborations Block
                        with st.expander("Collaborations"):
                            collaborations = get_artist_collaborations(artist['id'])
                            if collaborations:
                                st.write(', '.join(collaborations))
                            else:
                                st.write("No collaborations found.")

                        # Related Artists Block
                        with st.expander("Related Artists"):
                            if related_artists and related_artists['artists']:
                                for related_artist in related_artists['artists']:
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.image(related_artist['images'][0]['url'] if related_artist['images'] else None, width=100)
                                    with col2:
                                        st.write(f"🎤 {related_artist['name']}")
                                        st.write(f"Genres: {', '.join(related_artist['genres'])}")
                                        st.write(f"Popularity: {related_artist['popularity']}/100")
                            else:
                                st.write("No related artists found.")

                        # Genres of Related Artists Block
                        with st.expander("Genres of Related Artists"):
                            if related_artists and related_artists['artists']:
                                related_genres = []
                                for related_artist in related_artists['artists']:
                                    artist_genres = sp.artist(related_artist['id'])['genres']
                                    related_genres.extend(artist_genres)
                                st.write(', '.join(set(related_genres)) if related_genres else "No genres found for related artists.")
                            else:
                                st.write("No related artists to show genres for.")

                else:
                    typewriter_text("Hmm, I couldn't find any artists with that name. Let's try a different search!")

            else:
                # Handling track search
                tracks = results['tracks']['items']
                if tracks:
                    track = tracks[0]  # Get the first track
                    artist_info = sp.artist(track['artists'][0]['id'])
                    track_info = get_track_features(track['id'])

                    # Track Information Block
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(track['album']['images'][0]['url'] if track['album']['images'] else None, width=200)
                        st.audio(track['preview_url'])
                    with col2:
                        st.subheader(f"🎵 {track['name']} by {track['artists'][0]['name']}")
                        if artist_info:
                            st.write(f"Genres: {', '.join(artist_info['genres'])}")
                            st.write(f"🎉 Popularity: {artist_info['popularity']}/100")
                        if track_info:
                            display_info(track_info)
                    
                    # Collaborations Block for Track
                    with st.expander("Collaborations"):
                        collaborations = get_track_collaborations(track['artists'])
                        if collaborations:
                            st.write(', '.join(collaborations))
                        else:
                            st.write("No collaborations found.")

                    # Related Artists Block for Track's Artist
                    with st.expander("Related Artists"):
                        related_artists = sp.artist_related_artists(artist_info['id'])
                        if related_artists and related_artists['artists']:
                            for related_artist in related_artists['artists']:
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(related_artist['images'][0]['url'] if related_artist['images'] else None, width=100)
                                with col2:
                                    st.write(f"🎤 {related_artist['name']}")
                                    st.write(f"Genres: {', '.join(related_artist['genres'])}")
                                    st.write(f"Popularity: {related_artist['popularity']}/100")
                        else:
                            st.write("No related artists found.")

                    # Genres of Related Artists Block for Track's Artist
                    with st.expander("Genres of Related Artists"):
                        related_artists = sp.artist_related_artists(artist_info['id'])
                        if related_artists and related_artists['artists']:
                            related_genres = []
                            for related_artist in related_artists['artists']:
                                artist_genres = sp.artist(related_artist['id'])['genres']
                                related_genres.extend(artist_genres)
                            st.write(', '.join(set(related_genres)) if related_genres else "No genres found for related artists.")
                        else:
                            st.write("No related artists to show genres for.")
                    
                else:
                    typewriter_text("Oops! I couldn't find any tracks matching your search. Let's try something else!")

        if st.button("Return to Search"):
            st.session_state.stage = 'input'

if __name__ == "__main__":
    main()
