import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_improved_genre_mapping():
    return {
        'electronic': 'Electronic/Dance', 'edm': 'Electronic/Dance', 'electro': 'Electronic/Dance',
        'dance': 'Electronic/Dance', 'club': 'Electronic/Dance', 'breakbeat': 'Electronic/Dance',
        'drum-and-bass': 'Electronic/Dance', 'dubstep': 'Electronic/Dance', 'garage': 'Electronic/Dance',
        'hardstyle': 'Electronic/Dance', 'house': 'Electronic/Dance', 'chicago-house': 'Electronic/Dance',
        'deep-house': 'Electronic/Dance', 'progressive-house': 'Electronic/Dance',
        'minimal-techno': 'Electronic/Dance', 'techno': 'Electronic/Dance',
        'detroit-techno': 'Electronic/Dance', 'trance': 'Electronic/Dance',
        'funk': 'Electronic/Dance', 'disco': 'Electronic/Dance',
        
        'rock': 'Rock/Metal', 'alt-rock': 'Rock/Metal', 'hard-rock': 'Rock/Metal',
        'metal': 'Rock/Metal', 'heavy-metal': 'Rock/Metal', 'black-metal': 'Rock/Metal',
        'death-metal': 'Rock/Metal', 'metalcore': 'Rock/Metal', 'punk': 'Rock/Metal',
        'punk-rock': 'Rock/Metal', 'emo': 'Rock/Metal', 'goth': 'Rock/Metal',
        'grindcore': 'Rock/Metal', 'hardcore': 'Rock/Metal', 'industrial': 'Rock/Metal',
        
        'pop': 'Pop/Mainstream', 'indie-pop': 'Pop/Mainstream', 'k-pop': 'Pop/Mainstream',
        'power-pop': 'Pop/Mainstream', 'cantopop': 'Pop/Mainstream',
        
        'hip-hop': 'Hip-Hop/R&B', 'trip-hop': 'Hip-Hop/R&B', 'soul': 'Hip-Hop/R&B',
        
        'jazz': 'Traditional', 'blues': 'Traditional',
        'classical': 'Traditional', 'opera': 'Traditional',
        
        'folk': 'Country/Folk', 'acoustic': 'Country/Folk',
        'singer-songwriter': 'Country/Folk', 'songwriter': 'Country/Folk',
        'country': 'Country/Folk', 'rock-n-roll': 'Country/Folk',
        
        'afrobeat': 'World Music', 'indian': 'World Music', 'spanish': 'World Music',
        'french': 'World Music', 'german': 'World Music', 'swedish': 'World Music',
        'forro': 'World Music', 'sertanejo': 'World Music',
        
        'salsa': 'Latin', 'samba': 'Latin', 'tango': 'Latin',
        
        'ambient': 'Ambient/Chill', 'chill': 'Ambient/Chill',
        
        'dub': 'Reggae/Ska', 'dancehall': 'Reggae/Ska', 'ska': 'Reggae/Ska',
        
        'piano': 'Instrumental', 'guitar': 'Instrumental',
        
        'romance': 'Miscellaneous', 'sad': 'Miscellaneous', 'pop-film': 'Miscellaneous'
    }

def investigate_other_category(df):
    other_genres = df[df['mapped_genre'] == 'Other']['genre'].value_counts()
    print("Top genres in 'Other' category:")
    print(other_genres.head(10))
    return other_genres

def handle_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def main():
    print("Loading data...")
    df = pd.read_csv('data/dataset_limpio_unificado.csv')
    print(f"Loaded {len(df)} rows of data.")

    print("\nApplying improved genre mapping...")
    genre_mapping = create_improved_genre_mapping()
    df['mapped_genre'] = df['genre'].map(genre_mapping).fillna('Other')

    print("\nInvestigating 'Other' category...")
    other_genres = investigate_other_category(df)

    print("\nNew class distribution:")
    class_distribution = df['mapped_genre'].value_counts()
    total_samples = len(df)
    
    for genre, count in class_distribution.items():
        percentage = (count / total_samples) * 100
        print(f"Class {genre}: {count} ({percentage:.2f}%)")

    print("\nPreparing features...")
    # Identify numeric and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    categorical_features = [col for col in categorical_features if col not in ['genre', 'mapped_genre']]

    # Create preprocessing steps
    numeric_transformer = 'passthrough'
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Prepare the features and target
    X = df.drop(['genre', 'mapped_genre'], axis=1)
    y = df['mapped_genre']

    # Fit and transform the features
    X_encoded = preprocessor.fit_transform(X)

    print("\nHandling class imbalance...")
    X_resampled, y_resampled = handle_imbalance(X_encoded, y)

    print("\nFinal balanced class distribution:")
    balanced_distribution = pd.Series(y_resampled).value_counts()
    total_balanced_samples = len(y_resampled)
    
    for genre, count in balanced_distribution.items():
        percentage = (count / total_balanced_samples) * 100
        print(f"Class {genre}: {count} ({percentage:.2f}%)")

    print("\nSplitting into train/validation/test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    print("\nSaving processed datasets...")
    np.save('X_train.npy', X_train.toarray() if hasattr(X_train, 'toarray') else X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val.toarray() if hasattr(X_val, 'toarray') else X_val)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test.toarray() if hasattr(X_test, 'toarray') else X_test)
    np.save('y_test.npy', y_test)

    print("Process complete. Processed datasets saved as .npy files.")

if __name__ == '__main__':
    main()