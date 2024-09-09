# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from collections import defaultdict

# # Define genre mapping
# genre_mapping = {
#     'rock': ['rock', 'alt-rock', 'hard-rock', 'psych-rock', 'punk-rock', 'rock-n-roll'],
#     'metal': ['metal', 'black-metal', 'death-metal', 'heavy-metal', 'metalcore'],
#     'pop': ['pop', 'indie-pop', 'k-pop', 'power-pop', 'cantopop'],
#     'electronic': ['electronic', 'ambient', 'breakbeat', 'chill', 'drum-and-bass', 'dubstep', 'edm', 'electro', 'garage', 'hardstyle', 'minimal-techno', 'techno', 'trance', 'trip-hop'],
#     'house': ['house', 'chicago-house', 'deep-house', 'progressive-house'],
#     'hip-hop': ['hip-hop'],
#     'jazz': ['jazz'],
#     'classical': ['classical', 'opera'],
#     'blues': ['blues'],
#     'country': ['country'],
#     'folk': ['folk'],
#     'latin': ['salsa', 'samba', 'tango'],
#     'funk_soul': ['funk', 'soul'],
#     'reggae': ['reggae', 'dub', 'dancehall'],
#     'world': ['afrobeat', 'indian', 'forro', 'sertanejo'],
#     'other': ['acoustic', 'comedy', 'gospel', 'new-age', 'party', 'piano', 'show-tunes', 'sleep']
# }

# # Invert the mapping for easy lookup
# genre_lookup = {}
# for main_genre, sub_genres in genre_mapping.items():
#     for genre in sub_genres:
#         genre_lookup[genre] = main_genre

# # Function to unify genres
# def unify_genre(genre):
#     return genre_lookup.get(genre, 'other')

# def preprocess_and_unify_genres(file_path, output_path):
#     print(f"Loading data from {file_path}...")
#     df = pd.read_csv(file_path)
#     print(f"Original dataset shape: {df.shape}")

#     # 1. Remove irrelevant columns
#     columns_to_remove = ['Unnamed: 0', 'track_id', 'artist_name', 'track_name', 'year']
#     df = df.drop(columns=columns_to_remove)
#     print(f"Dataset shape after removing irrelevant columns: {df.shape}")

#     # 2. Normalize numeric variables
#     numeric_columns = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
#                        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
#                        'valence', 'tempo', 'duration_ms', 'time_signature']
#     scaler = StandardScaler()
#     df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
#     print("Numeric variables normalized.")

#     # 3. Remove classes with little information
#     classes_to_remove = ['sad', 'songwriter', 'chicago-house', 'detroit-techno', 'rock']
#     le = LabelEncoder()
#     le.fit(df['genre'])
#     df = df[~df['genre'].isin(classes_to_remove)]
#     print(f"Dataset shape after removing classes with little information: {df.shape}")

#     # 4. Unify genres
#     print("Unifying genres...")
#     df['unified_genre'] = df['genre'].apply(unify_genre)

#     # 5. Print class distribution
#     class_distribution = df['unified_genre'].value_counts()
#     print("\nUnified class distribution:")
#     for genre, count in class_distribution.items():
#         print(f"Class {genre}: {count}")

#     # 6. Save the clean dataset
#     df.to_csv(output_path, index=False)
#     print(f"\nClean dataset saved to {output_path}")
#     print(f"Final dataset shape: {df.shape}")

#     # 7. Print genre distribution percentages
#     print("\nGenre distribution percentages:")
#     print(df['unified_genre'].value_counts(normalize=True) * 100)

#     return df

# # Main function
# def main():
#     input_file = 'data/spotify_data.csv'
#     output_file = 'dataset_limpio_unificado.csv'
#     preprocess_and_unify_genres(input_file, output_file)

# if __name__ == "__main__":
#     main()

# import pandas as pd

# # Load the dataset
# df = pd.read_csv('data/mapped_dataset.csv')

# # Drop the 'unified_genre' column
# df = df.drop(columns=['unified_genre'])

# # Save the updated dataset to a new CSV file
# df.to_csv('data/mapped_dataset_updated.csv', index=False)


# import pandas as pd
# import csv

# def generate_genre_mapping(input_file, output_file):
#     # Read the CSV file
#     df = pd.read_csv(input_file)
    
#     # Remove rows where mapped_genre is "Other" or "Miscellaneous"
#     df = df[~df['mapped_genre'].isin(["Other", "Miscellaneous"])]
    
#     # Extract unique mappings
#     unique_mappings = df[['genre', 'mapped_genre']].drop_duplicates()
    
#     # Add psych-rock to Rock/Metal
#     unique_mappings = pd.concat([unique_mappings, pd.DataFrame([{'genre': 'psych-rock', 'mapped_genre': 'Rock/Metal'}])], ignore_index=True)
    
#     # Sort the mappings for consistency
#     unique_mappings = unique_mappings.sort_values(['mapped_genre', 'genre'])
    
#     # Write the mappings to a new CSV file
#     with open(output_file, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Subgenre', 'Mapped Genre'])  # Write header
#         for _, row in unique_mappings.iterrows():
#             writer.writerow([row['genre'], row['mapped_genre']])
    
#     print(f"Genre mapping CSV file has been generated: {output_file}")
#     print(f"Total unique mappings: {len(unique_mappings)}")

# # Generate the mapping
# input_file = 'data/mapped_dataset.csv'
# output_file = 'genre_mapping.csv'
# generate_genre_mapping(input_file, output_file)

# # Display the first few rows of the generated mapping
# print("\nFirst few rows of the generated mapping:")
# with open(output_file, 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     for i, row in enumerate(reader):
#         if i == 0 or i > 5:  # Print header and first 5 data rows
#             break
#         print(row)

# # Display the Rock/Metal category
# print("\nRock/Metal category:")
# rock_metal = pd.read_csv(output_file)
# print(rock_metal[rock_metal['Mapped Genre'] == 'Rock/Metal'])

# # Display all unique Mapped Genres
# print("\nAll unique Mapped Genres:")
# print(rock_metal['Mapped Genre'].unique())

# import pandas as pd

# def clean_genre_csv(input_file, output_file):
#     # Read the CSV file
#     df = pd.read_csv(input_file)
    
#   # Change 'psych-rock' to 'Rock/Metal' in the 'mapped_genre' column
#     df.loc[df['genre'] == 'psych-rock', 'mapped_genre'] = 'Rock/Metal'
    
#     # Remove rows where mapped_genre is 'Other' or 'Miscellaneous'
#     df = df[~df['mapped_genre'].isin(['Other', 'Miscellaneous'])]

#     # Define the genres to be removed
#     genres_to_remove = ['comedy', 'gospel', 'groove', 'new-age', 'party', 'pop-film', 'romance', 'show-tunes', 'sleep']
    
#     # Filter out the removed genres
#     df = df[~df['genre'].isin(genres_to_remove)]
#     # Save the cleaned dataset
#     df.to_csv(output_file, index=False)
    
#     print(f"Cleaned CSV file has been generated: {output_file}")
#     print(f"Total rows after cleaning: {len(df)}")
    
#     # Display genre distribution
#     print("\nGenre distribution after cleaning:")
#     genre_distribution = df['mapped_genre'].value_counts()
#     for genre, count in genre_distribution.items():
#         print(f"{genre}: {count}")

# # Clean the CSV
# input_file = 'data/mapped_dataset.csv'  
# output_file = 'data/cleaned_mapped_dataset_final.csv'
# clean_genre_csv(input_file, output_file)

# # Display the first few rows of the cleaned dataset
# print("\nFirst few rows of the cleaned dataset:")
# df_cleaned = pd.read_csv(output_file)
# print(df_cleaned.head())

# # Display column names
# print("\nColumns in the cleaned dataset:")
# print(df_cleaned.columns.tolist())


import pandas as pd

def create_improved_genre_mapping():
    return {
        'ambient': {'mapped': 'Ambient/Chill', 'unified': 'electronic'},
        'chill': {'mapped': 'Ambient/Chill', 'unified': 'electronic'},
        'classical': {'mapped': 'Classical/Orchestral', 'unified': 'classical'},
        'opera': {'mapped': 'Classical/Orchestral', 'unified': 'classical'},
        'country': {'mapped': 'Country/Americana', 'unified': 'country'},
        'rock-n-roll': {'mapped': 'Country/Americana', 'unified': 'rock'},
        'breakbeat': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'club': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'dance': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'deep-house': {'mapped': 'Electronic/Dance', 'unified': 'house'},
        'drum-and-bass': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'dubstep': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'edm': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'electro': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'electronic': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'garage': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'hardstyle': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'house': {'mapped': 'Electronic/Dance', 'unified': 'house'},
        'minimal-techno': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'progressive-house': {'mapped': 'Electronic/Dance', 'unified': 'house'},
        'techno': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'trance': {'mapped': 'Electronic/Dance', 'unified': 'electronic'},
        'acoustic': {'mapped': 'Folk/Acoustic', 'unified': 'folk'},
        'folk': {'mapped': 'Folk/Acoustic', 'unified': 'folk'},
        'singer-songwriter': {'mapped': 'Folk/Acoustic', 'unified': 'folk'},
        'disco': {'mapped': 'Funk/Disco', 'unified': 'funk_soul'},
        'funk': {'mapped': 'Funk/Disco', 'unified': 'funk_soul'},
        'hip-hop': {'mapped': 'Hip-Hop/R&B', 'unified': 'hip-hop'},
        'soul': {'mapped': 'Hip-Hop/R&B', 'unified': 'funk_soul'},
        'trip-hop': {'mapped': 'Hip-Hop/R&B', 'unified': 'hip-hop'},
        'guitar': {'mapped': 'Instrumental', 'unified': 'rock'},
        'piano': {'mapped': 'Instrumental', 'unified': 'classical'},
        'blues': {'mapped': 'Jazz/Blues', 'unified': 'blues'},
        'jazz': {'mapped': 'Jazz/Blues', 'unified': 'jazz'},
        'salsa': {'mapped': 'Latin', 'unified': 'latin'},
        'samba': {'mapped': 'Latin', 'unified': 'latin'},
        'tango': {'mapped': 'Latin', 'unified': 'latin'},
        'cantopop': {'mapped': 'Pop/Mainstream', 'unified': 'pop'},
        'indie-pop': {'mapped': 'Pop/Mainstream', 'unified': 'pop'},
        'k-pop': {'mapped': 'Pop/Mainstream', 'unified': 'pop'},
        'pop': {'mapped': 'Pop/Mainstream', 'unified': 'pop'},
        'power-pop': {'mapped': 'Pop/Mainstream', 'unified': 'pop'},
        'dancehall': {'mapped': 'Reggae/Ska', 'unified': 'reggae'},
        'dub': {'mapped': 'Reggae/Ska', 'unified': 'reggae'},
        'ska': {'mapped': 'Reggae/Ska', 'unified': 'reggae'},
        'alt-rock': {'mapped': 'Rock/Metal', 'unified': 'rock'},
        'black-metal': {'mapped': 'Rock/Metal', 'unified': 'metal'},
        'death-metal': {'mapped': 'Rock/Metal', 'unified': 'metal'},
        'emo': {'mapped': 'Rock/Metal', 'unified': 'rock'},
        'goth': {'mapped': 'Rock/Metal', 'unified': 'rock'},
        'grindcore': {'mapped': 'Rock/Metal', 'unified': 'metal'},
        'hard-rock': {'mapped': 'Rock/Metal', 'unified': 'rock'},
        'hardcore': {'mapped': 'Rock/Metal', 'unified': 'metal'},
        'heavy-metal': {'mapped': 'Rock/Metal', 'unified': 'metal'},
        'industrial': {'mapped': 'Rock/Metal', 'unified': 'metal'},
        'metal': {'mapped': 'Rock/Metal', 'unified': 'metal'},
        'metalcore': {'mapped': 'Rock/Metal', 'unified': 'metal'},
        'psych-rock': {'mapped': 'Rock/Metal', 'unified': 'rock'},
        'punk': {'mapped': 'Rock/Metal', 'unified': 'rock'},
        'punk-rock': {'mapped': 'Rock/Metal', 'unified': 'rock'},
        'afrobeat': {'mapped': 'World Music', 'unified': 'world'},
        'forro': {'mapped': 'World Music', 'unified': 'world'},
        'french': {'mapped': 'World Music', 'unified': 'world'},
        'german': {'mapped': 'World Music', 'unified': 'world'},
        'indian': {'mapped': 'World Music', 'unified': 'world'},
        'sertanejo': {'mapped': 'World Music', 'unified': 'world'},
        'spanish': {'mapped': 'World Music', 'unified': 'world'},
        'swedish': {'mapped': 'World Music', 'unified': 'world'}
    }

genre_mapping = create_improved_genre_mapping()


def get_genres(csv_genre):
    csv_genre = csv_genre.lower().strip()
    
    for subgenre, genres in genre_mapping.items():
        if subgenre in csv_genre:
            return genres['mapped'], genres['unified']
    
    return 'Other', 'other'

def update_csv_genres(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Apply the get_genres function to update both 'mapped_genre' and 'unified_genre' columns
    df[['mapped_genre', 'unified_genre']] = df['genre'].apply(lambda x: pd.Series(get_genres(x)))
    
    # Save the updated DataFrame
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("Updated genre distributions:")
    print("\nMapped Genres:")
    print(df['mapped_genre'].value_counts())
    print("\nUnified Genres:")
    print(df['unified_genre'].value_counts())
    
    # Check for any remaining 'Other' genres
    other_genres = df[df['mapped_genre'] == 'Other']['genre'].unique()
    if len(other_genres) > 0:
        print("\nGenres still mapped to 'Other':")
        for genre in sorted(other_genres):
            print(f"- {genre}")
    else:
        print("\nNo genres are mapped to 'Other'.")

# Usage
input_file = 'data/cleaned_mapped_dataset_final.csv'  # Replace with your input CSV file name
output_file = 'updated_mapped_dataset.csv'  # Replace with your desired output CSV file name

update_csv_genres(input_file, output_file)
