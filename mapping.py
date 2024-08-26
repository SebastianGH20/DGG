import pandas as pd
import numpy as np

def create_genre_mapping():
    return {
        'electronic': 'Electronic/Dance', 'edm': 'Electronic/Dance', 'electro': 'Electronic/Dance',
        'dance': 'Electronic/Dance', 'club': 'Electronic/Dance', 'breakbeat': 'Electronic/Dance',
        'drum-and-bass': 'Electronic/Dance', 'dubstep': 'Electronic/Dance', 'garage': 'Electronic/Dance',
        'hardstyle': 'Electronic/Dance', 'house': 'Electronic/Dance', 'chicago-house': 'Electronic/Dance',
        'deep-house': 'Electronic/Dance', 'progressive-house': 'Electronic/Dance',
        'minimal-techno': 'Electronic/Dance', 'techno': 'Electronic/Dance',
        'detroit-techno': 'Electronic/Dance', 'trance': 'Electronic/Dance',
        
        'rock': 'Rock/Metal', 'alt-rock': 'Rock/Metal', 'hard-rock': 'Rock/Metal',
        'metal': 'Rock/Metal', 'heavy-metal': 'Rock/Metal', 'black-metal': 'Rock/Metal',
        'death-metal': 'Rock/Metal', 'metalcore': 'Rock/Metal', 'punk': 'Rock/Metal',
        'punk-rock': 'Rock/Metal', 'emo': 'Rock/Metal', 'goth': 'Rock/Metal',
        'grindcore': 'Rock/Metal', 'hardcore': 'Rock/Metal', 'industrial': 'Rock/Metal',
        
        'pop': 'Pop/Mainstream', 'indie-pop': 'Pop/Mainstream', 'k-pop': 'Pop/Mainstream',
        'power-pop': 'Pop/Mainstream', 'cantopop': 'Pop/Mainstream',
        
        'hip-hop': 'Hip-Hop/R&B', 'trip-hop': 'Hip-Hop/R&B', 'soul': 'Hip-Hop/R&B',
        
        'jazz': 'Jazz/Blues', 'blues': 'Jazz/Blues',
        
        'classical': 'Classical/Orchestral', 'opera': 'Classical/Orchestral',
        
        'folk': 'Folk/Acoustic', 'acoustic': 'Folk/Acoustic',
        'singer-songwriter': 'Folk/Acoustic', 'songwriter': 'Folk/Acoustic',
        
        'afrobeat': 'World Music', 'indian': 'World Music', 'spanish': 'World Music',
        'french': 'World Music', 'german': 'World Music', 'swedish': 'World Music',
        'forro': 'World Music', 'sertanejo': 'World Music',
        
        'salsa': 'Latin', 'samba': 'Latin', 'tango': 'Latin',
        
        'ambient': 'Ambient/Chill', 'chill': 'Ambient/Chill',
        
        'dub': 'Reggae/Ska', 'dancehall': 'Reggae/Ska', 'ska': 'Reggae/Ska',
        
        'country': 'Country/Americana', 'rock-n-roll': 'Country/Americana',
        
        'funk': 'Funk/Disco', 'disco': 'Funk/Disco',
        
        'piano': 'Instrumental', 'guitar': 'Instrumental',
        
        'romance': 'Miscellaneous', 'sad': 'Miscellaneous', 'pop-film': 'Miscellaneous'
    }

def map_genres(genre, genre_mapping):
    return genre_mapping.get(genre, 'Other')

def main():
    print("Loading data...")
    df = pd.read_csv('dataset_limpio_unificado.csv')
    print(f"Loaded {len(df)} rows of data.")

    print("\nCreating genre mapping...")
    genre_mapping = create_genre_mapping()

    print("\nApplying genre mapping...")
    df['mapped_genre'] = df['genre'].apply(lambda x: map_genres(x, genre_mapping))

    print("\nNew class distribution:")
    class_distribution = df['mapped_genre'].value_counts()
    total_samples = len(df)
    
    for genre, count in class_distribution.items():
        percentage = (count / total_samples) * 100
        print(f"Class {genre}: {count} ({percentage:.2f}%)")

    print("\nSaving results...")
    df.to_csv('mapped_dataset.csv', index=False)
    
    genre_mapping_df = pd.DataFrame(list(genre_mapping.items()), columns=['original_genre', 'mapped_genre'])
    genre_mapping_df.to_csv('genre_mapping.csv', index=False)

    print("Process complete. Results saved to 'mapped_dataset.csv' and 'genre_mapping.csv'.")

if __name__ == '__main__':
    main()