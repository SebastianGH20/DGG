# import pandas as pd
# import numpy as np
# import joblib
# from tensorflow.keras.models import load_model
# import logging
# import os
# from sklearn import __version__ as sklearn_version

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Suppress TensorFlow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def load_model_and_preprocessors():
#     try:
#         model = load_model('genre_classification_model.h5')
#         preprocessor = joblib.load('preprocessor.pkl')
#         label_encoder = joblib.load('label_encoder.pkl')
#         return model, preprocessor, label_encoder
#     except FileNotFoundError as e:
#         logging.error(f"Error loading model or preprocessors: {e}")
#         raise

# def load_sample_data(file_path, sample_size=5):
#     try:
#         df = pd.read_csv(file_path, nrows=sample_size)
#         return df
#     except FileNotFoundError:
#         logging.error(f"Data file not found: {file_path}")
#         raise
#     except pd.errors.EmptyDataError:
#         logging.error(f"No data in file: {file_path}")
#         raise

# def get_feature_names(column_transformer):
#     feature_names = []
#     for name, pipe, features in column_transformer.transformers_:
#         if name != 'remainder':
#             if hasattr(pipe, 'get_feature_names_out'):
#                 if isinstance(features, slice):
#                     feature_names.extend(pipe.get_feature_names_out())
#                 else:
#                     feature_names.extend(pipe.get_feature_names_out(features))
#             elif hasattr(pipe, 'get_feature_names'):
#                 if isinstance(features, slice):
#                     feature_names.extend(pipe.get_feature_names())
#                 else:
#                     feature_names.extend(pipe.get_feature_names(features))
#             else:
#                 feature_names.extend(features)
#     return feature_names

# def preprocess_data(df, preprocessor):
#     X = df.drop(['genre', 'mapped_genre'], axis=1)
#     X_processed = preprocessor.transform(X)
    
#     if hasattr(X_processed, 'toarray'):
#         X_processed = X_processed.toarray()
    
#     feature_names = get_feature_names(preprocessor)
    
#     return pd.DataFrame(X_processed, columns=feature_names), X_processed

# def predict_genre(model, X_processed, label_encoder):
#     prediction = model.predict(X_processed[:1])
#     predicted_class_index = prediction.argmax()
#     predicted_genre = label_encoder.inverse_transform([predicted_class_index])[0]
#     return predicted_genre, prediction[0]

# def main():
#     try:
#         # Load model and preprocessors
#         model, preprocessor, label_encoder = load_model_and_preprocessors()
        
#         # Load sample data
#         df = load_sample_data('data/mapped_dataset.csv')
        
#         # Preprocess data
#         processed_df, X_processed = preprocess_data(df, preprocessor)
        
#         # Print sample processed row
#         logging.info("Sample processed row:")
#         logging.info(processed_df.iloc[0])
        
#         # Print corresponding genre
#         logging.info("\nCorresponding genre:")
#         logging.info(df['mapped_genre'].iloc[0])
        
#         # Get model prediction
#         predicted_genre, prediction_probabilities = predict_genre(model, X_processed, label_encoder)
        
#         logging.info("\nModel prediction:")
#         logging.info(f"Predicted genre: {predicted_genre}")
#         logging.info(f"Prediction probabilities: {prediction_probabilities}")
        
#     except Exception as e:
#         logging.error(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()

import pandas as pd

# Load your dataset
# Replace 'your_dataset.csv' with the actual path to your dataset
df = pd.read_csv('data/cleaned_mapped_dataset_final.csv')

# Get the number of unique unified genres
num_unified_genres = df['unified_genre'].nunique()
num_mapped_genres = df['mapped_genre'].nunique()

# Get the list of unique unified genres
unique_unified_genres = df['unified_genre'].unique()
unique_mapped_genres = df['mapped_genre'].unique()
unique_mapped_genres = df['genre'].unique()

# Get the count of each unified genre
unified_genre_counts = df['unified_genre'].value_counts()
mapped_genre_counts = df['mapped_genre'].value_counts()

print(f"Number of unique unified genres: {num_unified_genres}")
print(f"Number of unique mapped genres: {num_mapped_genres}")
print("\nList of unique unified genres:")
print(unique_unified_genres)
print("\nCount of each unified genre:")
print(unified_genre_counts)
print("\nList of unique mapped genres:")
print(unique_mapped_genres)
print("\nCount of each mapped genre:")
print(mapped_genre_counts)

# Optionally, you can save this information to a file
with open('unified_genre_analysis.txt', 'w') as f:
    f.write(f"Number of unique unified genres: {num_unified_genres}\n\n")
    f.write("List of unique unified genres:\n")
    f.write(str(unique_unified_genres) + "\n\n")
    f.write("Count of each unified genre:\n")
    f.write(str(unified_genre_counts))

print("\nAnalysis has been saved to 'unified_genre_analysis.txt'")


# Function to get genres with 'other' unified genre
def get_other_genres(df):
    other_mask = df['unified_genre'] == 'other'
    other_genres = df.loc[other_mask, 'genre'].unique()
    return sorted(other_genres)

# Get the list of genres mapped to 'other'
other_genres = get_other_genres(df)

print("Genres mapped to 'other' unified genre:")
for genre in other_genres:
    print(f"- {genre}")

# Count of each genre mapped to 'other'
other_genre_counts = df[df['unified_genre'] == 'other']['genre'].value_counts()
