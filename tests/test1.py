import pandas as pd
import tensorflow as tf
import joblib

# Load the saved components
model = tf.keras.models.load_model('genre_classification_model.keras')
preprocessor = joblib.load('preprocessor.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Example row data for prediction
new_data = pd.DataFrame({
    'popularity': [-0.1500183890296356],
    'danceability': [0.7587249459994932],
    'energy': [-0.9229914740719376],
    'key': [0.1760498788336178],
    'loudness': [-1.487338336122037],
    'mode': [0.7587249459994932],
    'speechiness': [-0.5196650955565505],
    'acousticness': [0.7787975199589121],
    'instrumentalness': [-0.6729309951206706],
    'liveness': [-0.5521387225194017],
    'valence': [-1.1193388400001902],
    'tempo': [-2.2871974824435],
    'duration_ms': [0.1030490574668306],  # Adjust as needed
    'time_signature': [0.2440057437544111],  # Adjust as needed
    'unified_genre': ['folk']
})

# # Example row data for prediction
# new_data = pd.DataFrame({
#     'popularity': [2.1791441939445204],
#     'danceability': [1.2026956794417198],
#     'energy': [-0.2517278088706602],
#     'key': [1.0441681065204325],
#     'loudness': [0.9861216580414566],
#     'mode': [-1.3180006869059344],
#     'speechiness': [-0.1964254090537162],
#     'acousticness': [-0.8719668737278155],
#     'instrumentalness': [-0.6906370464482766],
#     'liveness': [-0.6341994445564892],
#     'valence': [0.3144524077963246],
#     'tempo': [-0.950415988317133],
#     'duration_ms': [0.315950284082433],  # Adjust as needed
#     'time_signature': [0.2440057437544111],  # Adjust as needed
#     'unified_genre': ['metal']
# })

# # Example row data for prediction
# new_data = pd.DataFrame({
#     'popularity': [0.0388326312115121],
#     'danceability': [0.7442866237274701],
#     'energy': [-1.3304474736441552],
#     'key': [0.8972478260840465],
#     'loudness': [-0.6435028552412139],
#     'mode': [0.7587249459994932],
#     'speechiness': [-0.395099557831068],
#     'acousticness': [-0.9055776762650748],
#     'instrumentalness': [-0.6912214228390922],
#     'liveness': [1.7704283799848335],
#     'valence': [1.267830484253228],
#     'tempo': [-0.0829804949293986],
#     'duration_ms': [0.871589492651685],  # Adjust as needed
#     'time_signature': [0.2440057437544111],  # Adjust as needed
#     'unified_genre': ['rock']
# })

# Preprocess the new data
X_new = new_data.copy()  # Copy to avoid modifying original
print(X_new)

# The preprocessor should handle the encoding of 'unified_genre'
X_new_preprocessed = preprocessor.transform(X_new)
print(X_new_preprocessed)

# Make predictions
y_pred_probs = model.predict(X_new_preprocessed)
y_pred_classes = y_pred_probs.argmax(axis=1)  # Get class indices

# Convert class indices to class names
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

# Display predictions
predicted_genre = y_pred_labels[0]
print(f"Predicted genre: '{predicted_genre}'")


