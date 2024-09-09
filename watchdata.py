import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Load preprocessor and label encoder
preprocessor = joblib.load('preprocessor.pkl')
le = joblib.load('label_encoder.pkl')

# Load the model
model = tf.keras.models.load_model('genre_classification_model.keras')

# Print the feature names used during training
def print_feature_names(preprocessor):
    transformers = preprocessor.transformers_
    feature_names = []

    for name, transformer, columns in transformers:
        if isinstance(transformer, StandardScaler):
            feature_names.extend(columns)
        elif isinstance(transformer, OneHotEncoder):
            feature_names.extend(transformer.get_feature_names_out(input_features=columns))

    logging.info(f"Feature names used during training: {feature_names}")
    return feature_names

# Get the feature names
feature_names = print_feature_names(preprocessor)

# Create mock input DataFrame
mock_input = pd.DataFrame({
    'popularity': [1.92734283362299],
    'danceability': [0.7239985210752148],
    'energy': [-0.3610706960996316],
    'key': [1.6067250937743145],
    'loudness': [0.2598200649954539],
    'mode': [-1.3180006869059344],
    'speechiness': [-0.2547662305200814],
    'acousticness': [-0.8711217725023787],
    'instrumentalness': [-0.6729309951206706],
    'liveness': [-0.487484820308363],
    'valence': [0.642176121578385],
    'tempo': [-0.9507517871299962],
    'duration_ms': [-0.2164065126616003],
    'time_signature': [0.2440057437544111],
    'unified_genre': ['rock']
})

# Map genres
genre_mapping = create_improved_genre_mapping()
mock_input['mapped_genre'] = mock_input['unified_genre'].map(genre_mapping).fillna('Other')

# Separate features
X_mock = mock_input.drop(['unified_genre', 'mapped_genre'], axis=1)

# Ensure the order of columns matches what the model expects
# Add missing columns with default values (e.g., zeros)
for col in feature_names:
    if col not in X_mock.columns:
        if 'unified_genre_' in col:
            X_mock[col] = 0  # Add missing one-hot encoded columns
        else:
            X_mock[col] = 0  # Add missing numeric columns

# Reorder columns to match feature_names
X_mock = X_mock[feature_names]

# Transform the features
try:
    X_mock_preprocessed = preprocessor.transform(X_mock)
except ValueError as e:
    logging.error(f"Error during preprocessing: {e}")
    raise

# Make predictions
y_pred = model.predict(X_mock_preprocessed)
y_pred_classes = np.argmax(y_pred, axis=1)

# Decode the predictions
predicted_genre = le.inverse_transform(y_pred_classes)

# Combine predictions with the original data
mock_input['predicted_genre'] = predicted_genre
print(mock_input[['unified_genre', 'predicted_genre']])
