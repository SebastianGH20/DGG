import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load your dataset
df = pd.read_csv("data\spotify_data.csv")

# Step 2: Create genre mapping functions
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

# Step 3: Apply genre mappings

improved_genre_mapping = create_improved_genre_mapping()
unified_genre_mapping = create_unified_genre_mapping()

df['mapped_genre'] = df['genre'].map(improved_genre_mapping)
df['unified_genre'] = df['genre'].map(unified_genre_mapping)

# Step 4: Remove rows with 'Unknown' or 'unknown'
df = df.dropna(subset=['mapped_genre', 'unified_genre'])

# Step 5: Scaling numeric features
numeric_features = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])


# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')


# Step 6: Drop unwanted columns
df = df.drop(columns=['Unnamed: 0', 'artist_name', 'track_name', 'track_id', 'year'])

# Step 7: Save the DataFrame to a new CSV file
df.to_csv("spotify_adapted.csv", index=False)

# Display the transformed DataFrame
print(df.head())
