import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load the preprocessor, model, and label encoder
preprocessor = joblib.load('preprocessor.pkl')
model = load_model('genre_classification_model.h5')
le = joblib.load('label_encoder.pkl')

# Create a mock input based on the provided data format
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
    'unified_genre': ['rock']  # Updated unified genre
})

# Ensure 'unified_genre' is handled correctly
mock_input['unified_genre'] = mock_input['unified_genre'].astype(str)  # Convert to string type

# Transform using the preprocessor
X_preprocessed = preprocessor.transform(mock_input)

# Verify the shape of the preprocessed input
print("Shape of preprocessed input:", X_preprocessed.shape)

# Make sure it has the expected number of features
if X_preprocessed.shape[1] != 30:
    raise ValueError(f"Expected 30 features, but got {X_preprocessed.shape[1]}")

# Make prediction
prediction = model.predict(X_preprocessed)
predicted_class_index = np.argmax(prediction, axis=1)
predicted_genre = le.inverse_transform(predicted_class_index)[0]

# Print results
print(f"Input unified genre: {mock_input['unified_genre'].iloc[0]}")
print(f"Predicted genre: {predicted_genre}")
print(f"Prediction probabilities: {prediction[0]}")
print(f"Class names: {le.classes_}")

# Optional: Compare with the actual mapped genre (if available)
actual_mapped_genre = "Rock/Metal"  # This is from the 'mapped_genre' column in your data
print(f"Actual mapped genre: {actual_mapped_genre}")
print(f"Prediction {'matches' if predicted_genre == actual_mapped_genre else 'does not match'} actual mapped genre.")
