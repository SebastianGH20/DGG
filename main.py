import streamlit as st
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

# Set page config at the very beginning
st.set_page_config(page_title="Spotify Search Visualization", page_icon="🎵", layout="wide")

# Load environment variables
load_dotenv()

# Configure Spotify
client_credentials_manager = SpotifyClientCredentials(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Custom CSS for typewriter effect
st.markdown("""
<style>
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    .typewriter {
        overflow: hidden;
        border-right: .15em solid orange;
        white-space: nowrap;
        margin: 0 auto;
        letter-spacing: .15em;
        animation: 
            typing 3.5s steps(40, end),
            blink-caret .75s step-end infinite;
    }
</style>
""", unsafe_allow_html=True)

def typewriter_text(text):
    container = st.empty()
    for i in range(len(text) + 1):
        container.markdown(f'<p class="typewriter">{text[:i]}</p>', unsafe_allow_html=True)
        time.sleep(0.05)
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

def main():
    st.title("Spotify Search Visualization")

    typewriter_text("Hi, I'm an A.I. trained to evaluate musical taste. To get started, I'll need to see your Spotify or Apple Music data.")

    query = st.text_input("Enter an artist or song name:")
    search_type = st.radio("Search for:", ("Artist", "Track"))

    if st.button("Search"):
        if query:
            with st.spinner("Searching Spotify..."):
                results = search_spotify(query, type=search_type.lower())
                
            if results:
                st.success("Search completed!")
                
                if search_type == "Artist":
                    artists = results['artists']['items']
                    if artists:
                        artist = artists[0]  # Get the first artist
                        artist_info, top_tracks = get_artist_profile(artist['id'])
                        
                        if artist_info:
                            st.subheader(f"🎤 {artist_info['name']}")
                            st.image(artist_info['images'][0]['url'] if artist_info['images'] else None, width=200)
                            st.write(f"Genres: {', '.join(artist_info['genres'])}")
                            st.write(f"Popularity: {artist_info['popularity']}/100")
                            st.write(f"Followers: {artist_info['followers']['total']:,}")
                            
                            st.subheader("Top Tracks:")
                            for track in top_tracks['tracks'][:5]:
                                st.write(f"🎵 {track['name']}")
                                st.audio(track['preview_url'])
                    else:
                        st.warning("No artists found. Try a different search term.")
                else:
                    tracks = results['tracks']['items']
                    for track in tracks:
                        st.subheader(f"🎵 {track['name']} by {track['artists'][0]['name']}")
                        st.image(track['album']['images'][0]['url'] if track['album']['images'] else None, width=200)
                        st.write(f"🎚️ Popularity: {track['popularity']}/100")
                        st.audio(track['preview_url'])
                        st.markdown("---")
            else:
                st.warning("No results found. Please try a different search term.")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    main()