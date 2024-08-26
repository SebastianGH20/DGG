import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict

# Define genre mapping
genre_mapping = {
    'rock': ['rock', 'alt-rock', 'hard-rock', 'psych-rock', 'punk-rock', 'rock-n-roll'],
    'metal': ['metal', 'black-metal', 'death-metal', 'heavy-metal', 'metalcore'],
    'pop': ['pop', 'indie-pop', 'k-pop', 'power-pop', 'cantopop'],
    'electronic': ['electronic', 'ambient', 'breakbeat', 'chill', 'drum-and-bass', 'dubstep', 'edm', 'electro', 'garage', 'hardstyle', 'minimal-techno', 'techno', 'trance', 'trip-hop'],
    'house': ['house', 'chicago-house', 'deep-house', 'progressive-house'],
    'hip-hop': ['hip-hop'],
    'jazz': ['jazz'],
    'classical': ['classical', 'opera'],
    'blues': ['blues'],
    'country': ['country'],
    'folk': ['folk'],
    'latin': ['salsa', 'samba', 'tango'],
    'funk_soul': ['funk', 'soul'],
    'reggae': ['reggae', 'dub', 'dancehall'],
    'world': ['afrobeat', 'indian', 'forro', 'sertanejo'],
    'other': ['acoustic', 'comedy', 'gospel', 'new-age', 'party', 'piano', 'show-tunes', 'sleep']
}

# Invert the mapping for easy lookup
genre_lookup = {}
for main_genre, sub_genres in genre_mapping.items():
    for genre in sub_genres:
        genre_lookup[genre] = main_genre

# Function to unify genres
def unify_genre(genre):
    return genre_lookup.get(genre, 'other')

def preprocess_and_unify_genres(file_path, output_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")

    # 1. Remove irrelevant columns
    columns_to_remove = ['Unnamed: 0', 'track_id', 'artist_name', 'track_name', 'year']
    df = df.drop(columns=columns_to_remove)
    print(f"Dataset shape after removing irrelevant columns: {df.shape}")

    # 2. Normalize numeric variables
    numeric_columns = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
                       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                       'valence', 'tempo', 'duration_ms', 'time_signature']
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    print("Numeric variables normalized.")

    # 3. Remove classes with little information
    classes_to_remove = ['sad', 'songwriter', 'chicago-house', 'detroit-techno', 'rock']
    le = LabelEncoder()
    le.fit(df['genre'])
    df = df[~df['genre'].isin(classes_to_remove)]
    print(f"Dataset shape after removing classes with little information: {df.shape}")

    # 4. Unify genres
    print("Unifying genres...")
    df['unified_genre'] = df['genre'].apply(unify_genre)

    # 5. Print class distribution
    class_distribution = df['unified_genre'].value_counts()
    print("\nUnified class distribution:")
    for genre, count in class_distribution.items():
        print(f"Class {genre}: {count}")

    # 6. Save the clean dataset
    df.to_csv(output_path, index=False)
    print(f"\nClean dataset saved to {output_path}")
    print(f"Final dataset shape: {df.shape}")

    # 7. Print genre distribution percentages
    print("\nGenre distribution percentages:")
    print(df['unified_genre'].value_counts(normalize=True) * 100)

    return df

# Main function
def main():
    input_file = 'data/spotify_data.csv'
    output_file = 'dataset_limpio_unificado.csv'
    preprocess_and_unify_genres(input_file, output_file)

if __name__ == "__main__":
    main()