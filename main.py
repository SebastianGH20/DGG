# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import classification_report
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.optimizers import Adam

# def prepare_data(file_path):
#     data = pd.read_csv(file_path)
    
#     features = ['popularity', 'year', 'danceability', 'energy', 'key', 'loudness', 'mode',
#                 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
#                 'tempo', 'duration_ms', 'time_signature']
    
#     X = data[features]
#     y = data['genre']
    
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)
#     y_categorical = to_categorical(y_encoded)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler

# def create_model(input_shape, num_classes):
#     model = Sequential([
#         Dense(256, activation='relu', input_shape=input_shape),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(128, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(64, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(num_classes, activation='softmax')
#     ])
    
#     model.compile(optimizer=Adam(learning_rate=0.001),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
#     history = model.fit(X_train, y_train, 
#                         epochs=epochs, 
#                         batch_size=batch_size,
#                         validation_data=(X_val, y_val),
#                         callbacks=[early_stopping, reduce_lr],
#                         verbose=1)
#     return history

# if __name__ == "__main__":
#     file_path = "data/spotify_data.csv"  # Replace with your actual file path
#     X_train, X_test, y_train, y_test, label_encoder, scaler = prepare_data(file_path)
    
#     # Split training data into train and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
#     input_shape = (X_train.shape[1],)
#     num_classes = y_train.shape[1]
    
#     model = create_model(input_shape, num_classes)
    
#     # Print model summary
#     model.summary()
    
#     # Print class distribution
#     print("\nClass distribution:")
#     for i, count in enumerate(np.sum(y_train, axis=0)):
#         print(f"Class {label_encoder.inverse_transform([i])[0]}: {count}")
    
#     history = train_model(model, X_train, y_train, X_val, y_val)
    
#     # Evaluate on test set
#     test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
#     print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
#     # Generate classification report
#     y_pred = model.predict(X_test)
#     y_pred_classes = np.argmax(y_pred, axis=1)
#     y_true_classes = np.argmax(y_test, axis=1)
#     class_names = label_encoder.classes_
#     print("\nClassification Report:")
#     print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
#     # Example prediction
#     example = X_test[0].reshape(1, -1)
#     prediction = model.predict(example)
#     predicted_genre = label_encoder.inverse_transform([np.argmax(prediction)])
#     print(f"\nPredicted genre for example: {predicted_genre[0]}")



# -------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, hamming_loss, jaccard_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import Sequence

# class SimpleDataGenerator(Sequence):
#     def __init__(self, X, y, batch_size=32):
#         self.X, self.y = X, y
#         self.batch_size = batch_size

#     def __len__(self):
#         return int(np.ceil(len(self.X) / float(self.batch_size)))

#     def __getitem__(self, idx):
#         batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
#         return batch_X, batch_y

# def prepare_data(file_path, sample_size=10000, min_samples_per_genre=50):
#     print(f"Reading data from {file_path}")
#     data = pd.read_csv(file_path)
#     if len(data) > sample_size:
#         data = data.sample(n=sample_size, random_state=42)
    
#     features = ['popularity', 'year', 'danceability', 'energy', 'key', 'loudness', 'mode',
#                 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
#                 'tempo', 'duration_ms', 'time_signature']
    
#     X = data[features]
#     genres = data['genre'].str.get_dummies(sep=',')
#     genres = genres.loc[:, genres.sum() >= min_samples_per_genre]
    
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     return train_test_split(X_scaled, genres.values, test_size=0.2, random_state=42)

# def create_model(input_shape, num_genres):
#     model = Sequential([
#         Dense(128, activation='relu', input_shape=input_shape),
#         Dropout(0.3),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(num_genres, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def train_model(model, train_generator, val_generator, epochs=5):
#     early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
#     history = model.fit(train_generator, epochs=epochs,
#                         validation_data=val_generator,
#                         callbacks=[early_stopping],
#                         verbose=1)
#     return history

# if __name__ == "__main__":
#     file_path = "data/spotify_data.csv"
#     X_train, X_test, y_train, y_test = prepare_data(file_path)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
#     input_shape = (X_train.shape[1],)
#     num_genres = y_train.shape[1]
#     batch_size = 32
    
#     train_generator = SimpleDataGenerator(X_train, y_train, batch_size)
#     val_generator = SimpleDataGenerator(X_val, y_val, batch_size)
#     test_generator = SimpleDataGenerator(X_test, y_test, batch_size)
    
#     model = create_model(input_shape, num_genres)
#     model.summary()
    
#     print("\nTraining model...")
#     history = train_model(model, train_generator, val_generator)
    
#     print("\nEvaluating model...")
#     test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
#     print(f"Test Accuracy: {test_accuracy:.4f}")
    
#     print("\nGenerating predictions...")
#     y_pred = model.predict(X_test)
#     y_pred_binary = (y_pred > 0.5).astype(int)
    
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred_binary))
    
#     print(f"\nHamming Loss: {hamming_loss(y_test, y_pred_binary):.4f}")
#     print(f"Jaccard Score: {jaccard_score(y_test, y_pred_binary, average='samples'):.4f}")
    
#     print("\nExample prediction:")
#     example = X_test[0:1]
#     prediction = model.predict(example)
#     predicted_genres = np.where(prediction[0] > 0.5)[0]
#     print(f"Predicted genre indices: {predicted_genres}")


# -------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tqdm.keras import TqdmCallback
from tqdm.auto import tqdm
import time

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Dataset shape: {data.shape}")
    
    # Separate features and target
    X = data.drop(['genre', 'popularity'], axis=1)
    y = data['genre']
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(y.unique())}")
    
    # Encode the target labels
    print("Encoding target labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Normalize the features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Data preprocessing completed.")
    return X_scaled, y_encoded, le.classes_

# Create class weights
def create_class_weights(y):
    print("Calculating class weights...")
    class_weights = {}
    total_samples = len(y)
    classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(classes, counts):
        class_weights[cls] = (1 / count) * (total_samples / len(classes))
    print(f"Number of classes: {len(class_weights)}")
    return class_weights

# Build the model
def build_model(input_shape, num_classes):
    print(f"Building model with input shape: {input_shape} and {num_classes} output classes...")
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    print("Model built successfully.")
    return model

# Feature importance analysis
def feature_importance_analysis(X, y, feature_names):
    print("Performing feature importance analysis...")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Feature importance ranking:")
    for f, idx in enumerate(indices):
        print("{0} - {1}: {2:.4f}".format(f + 1, feature_names[idx], importances[idx]))
    print("Feature importance analysis completed.")

# Custom callback for epoch progress
class EpochProgressCallback(TqdmCallback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_progress = tqdm(total=total_epochs, desc="Epochs", position=0)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_progress.update(1)

    def on_train_end(self, logs=None):
        self.epoch_progress.close()

# Main function
def main():
    start_time = time.time()
    print("Starting music genre classification process...")
    
    # Load and preprocess the data
    X, y, class_names = load_and_preprocess_data('data\dataset_limpio.csv')
    
    # Perform feature importance analysis
    feature_names = pd.read_csv('data\dataset_limpio.csv').drop(['genre', 'popularity'], axis=1).columns
    feature_importance_analysis(X, y, feature_names)
    
    # Split the data
    print("Splitting the data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Create class weights
    class_weights = create_class_weights(y_train)
    
    # Convert labels to categorical
    print("Converting labels to categorical...")
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    # Build the model
    model = build_model(X_train.shape[1], len(class_names))
    
    # Compile the model
    print("Compiling the model...")
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Define callbacks
    print("Setting up callbacks...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    
    # Set up progress bar callbacks
    epochs = 200
    epoch_progress = EpochProgressCallback(epochs)
    tqdm_callback = TqdmCallback(verbose=0)  # Disable default epoch progress bar
    
    # Train the model
    print(f"Starting model training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=epochs,
        batch_size=16,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr, epoch_progress, tqdm_callback],
        verbose=0  # Disable default verbosity
    )
    print("Model training completed.")
    
    # Evaluate the model
    print("\nEvaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))
    
    # Plot training history
    print("Generating training history plots...")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print("Music genre classification process completed.")

if __name__ == "__main__":
    main()