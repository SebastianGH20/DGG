import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
import joblib
from time import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

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

genre_mapping = create_improved_genre_mapping()



def handle_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input

def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    logging.info("Loading data...")
    df = pd.read_csv('updated_mapped_dataset.csv')
    logging.info(f"Loaded {len(df)} rows of data.")

    logging.info("Applying improved genre mapping...")
    df['mapped_genre'] = df['genre'].map(genre_mapping).fillna('Other')

    logging.info("New class distribution:")
    class_distribution = df['mapped_genre'].value_counts()
    total_samples = len(df)
    for genre, count in class_distribution.items():
        percentage = (count / total_samples) * 100
        logging.info(f"Class {genre}: {count} ({percentage:.2f}%)")

    logging.info("Preparing features...")
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    categorical_features = [col for col in categorical_features if col not in ['genre', 'mapped_genre']]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    X = df.drop(['genre', 'mapped_genre'], axis=1)
    y = df['mapped_genre']

    X_encoded = preprocessor.fit_transform(X)

    logging.info("Handling class imbalance...")
    X_resampled, y_resampled = handle_imbalance(X_encoded, y)

    logging.info("Final balanced class distribution:")
    balanced_distribution = pd.Series(y_resampled).value_counts()
    total_balanced_samples = len(y_resampled)
    for genre, count in balanced_distribution.items():
        percentage = (count / total_balanced_samples) * 100
        logging.info(f"Class {genre}: {count} ({percentage:.2f}%)")

    logging.info("Splitting into train/validation/test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    logging.info(f"Train set size: {X_train.shape[0]}")
    logging.info(f"Validation set size: {X_val.shape[0]}")
    logging.info(f"Test set size: {X_test.shape[0]}")

    logging.info("Starting model training and evaluation...")
    
    try:
        # Handle string labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_val_encoded = le.transform(y_val)
        class_names = le.classes_
        
        logging.info(f"Unique classes: {class_names}")

        # Neural Network
        num_classes = len(class_names)
        model = create_model(X_train.shape[1], num_classes)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        
        class LoggingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 5 == 0:
                    logging.info(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

        start_time = time()
        history = model.fit(X_train, y_train_encoded,
                            validation_data=(X_val, y_val_encoded),
                            epochs=50,
                            batch_size=1024,
                            callbacks=[early_stopping, reduce_lr, LoggingCallback()],
                            verbose=0)
        end_time = time()
        
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")

        # Evaluate
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        logging.info("\nClassification Report:")
        from sklearn.metrics import classification_report
        print(classification_report(y_val_encoded, y_pred_classes, target_names=class_names))

        # Save the model and preprocessor
        # Save the model using the `SavedModel` format
        model.save('genre_classification_model.keras', save_format='h5')
        joblib.dump(preprocessor, 'preprocessor.pkl')
        joblib.dump(le, 'label_encoder.pkl')
        logging.info("Model saved as 'genre_classification_model.keras'")
        logging.info("Preprocessor saved as 'preprocessor.pkl'")
        logging.info("Label Encoder saved as 'label_encoder.pkl'")
        
    except Exception as e:
        logging.error(f"An error occurred during model training and evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()