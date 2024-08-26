import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

def prepare_data(file_path, top_n_genres=None):
    data = pd.read_csv(file_path)
    features = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'duration_ms', 'time_signature']
    X = data[features]
    y = data['genre']
    
    if top_n_genres:
        top_genres = y.value_counts().nlargest(top_n_genres).index
        mask = y.isin(top_genres)
        X = X[mask]
        y = y[mask]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le

def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),  # Use Input layer to define input shape
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(X_train, X_test, y_train, y_test, le, num_genres):
    input_shape = (X_train.shape[1],)
    model = create_model(input_shape, num_genres)
    
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy ({num_genres} genres): {test_accuracy:.4f}")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Convert classes to string to avoid potential errors
    genre_names = list(map(str, le.classes_))
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=genre_names))

# Main execution
if __name__ == "__main__":
    file_path = "data/dataset_limpio.csv"
    print("\nTraining with top 10 genres:")
    X_train_top10, X_test_top10, y_train_top10, y_test_top10, le_top10 = prepare_data(file_path, top_n_genres=10)
    train_and_evaluate(X_train_top10, X_test_top10, y_train_top10, y_test_top10, le_top10, 10)