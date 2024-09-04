import os
import numpy as np
import pandas as pd
import joblib  # <-- NUEVA IMPORTACIÓN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
from time import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, input_shape=(input_shape,), activation='relu'),
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

def train_and_evaluate_model(X_train, X_val, y_train, y_val, class_names):
    num_classes = len(class_names)
    
    logging.info("Starting model training and evaluation...")
    
    try:
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        logging.info("Feature scaling completed.")

        # Guardar el scaler
        joblib.dump(scaler, 'scaler.pkl')  # <-- GUARDAR EL SCALER
        logging.info("Scaler saved as 'scaler.pkl'")

        # Neural Network
        model = create_model(X_train_scaled.shape[1], num_classes)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        
        # Custom callback for logging
        class LoggingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 5 == 0:
                    logging.info(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

        start_time = time()
        history = model.fit(X_train_scaled, y_train,
                            validation_data=(X_val_scaled, y_val),
                            epochs=50,
                            batch_size=1024,
                            callbacks=[early_stopping, reduce_lr, LoggingCallback()],
                            verbose=0)
        end_time = time()
        
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")

        # Evaluate
        y_pred = model.predict(X_val_scaled)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        logging.info("\nClassification Report:")
        print(classification_report(y_val, y_pred_classes, target_names=class_names))

        return model

    except Exception as e:
        logging.error(f"An error occurred during model training and evaluation: {str(e)}")
        raise

def main():
    try:
        logging.info("Loading preprocessed data...")
        start_time = time()
        X = np.load('X_train.npy', allow_pickle=True)
        y = np.load('y_train.npy', allow_pickle=True)
        end_time = time()
        
        logging.info(f"Loaded data shape: {X.shape}")
        logging.info(f"Data loading completed in {end_time - start_time:.2f} seconds.")
        
        # Handle string labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_
        
        logging.info(f"Unique classes: {class_names}")
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        model = train_and_evaluate_model(X_train, X_val, y_train, y_val, class_names)
        
        # Save the model
        model.save('genre_classification_model.h5')
        logging.info("Model saved as 'genre_classification_model.h5'")
        
    except FileNotFoundError as e:
        logging.error(f"Error: Required data file not found. {str(e)}")
    except ValueError as e:
        logging.error(f"Error: Invalid data format. {str(e)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()

       ##0.83




# import os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import classification_report
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import logging
# from time import time
# from tensorflow.keras.regularizers import l2

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Suppress TensorFlow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.get_logger().setLevel('ERROR')

# def create_model(input_shape, num_classes):
#     model = Sequential([
#         Dense(512, input_shape=(input_shape,), activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.4),
#         Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.4),
#         Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.4),
#         Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.4),
#         Dense(num_classes, activation='softmax')
#     ])
#     return model

# def train_and_evaluate_model(X_train, X_val, y_train, y_val, class_names):
#     num_classes = len(class_names)

#     logging.info("Starting model training and evaluation...")

#     try:
#         # Feature scaling
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_val_scaled = scaler.transform(X_val)
#         logging.info("Feature scaling completed.")

#         # Neural Network
#         model = create_model(X_train_scaled.shape[1], num_classes)
#         model.compile(optimizer=Adam(learning_rate=0.0005),
#                       loss='sparse_categorical_crossentropy',
#                       metrics=['accuracy'])

#         early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

#         # Custom callback for logging
#         class LoggingCallback(tf.keras.callbacks.Callback):
#             def on_epoch_end(self, epoch, logs=None):
#                 if (epoch + 1) % 5 == 0:
#                     logging.info(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

#         start_time = time()
#         history = model.fit(X_train_scaled, y_train,
#                             validation_data=(X_val_scaled, y_val),
#                             epochs=100,  # Increased from 50 to 100
#                             batch_size=512,  # Reduced from 1024 to 512
#                             callbacks=[early_stopping, reduce_lr, LoggingCallback()],
#                             verbose=0)
#         end_time = time()

#         logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")

#         # Evaluate
#         y_pred = model.predict(X_val_scaled)
#         y_pred_classes = np.argmax(y_pred, axis=1)

#         logging.info("\nClassification Report:")
#         print(classification_report(y_val, y_pred_classes, target_names=class_names))

#         return model, history

#     except Exception as e:
#         logging.error(f"An error occurred during model training and evaluation: {str(e)}")
#         raise

# def main():
#     try:
#         logging.info("Loading preprocessed data...")
#         start_time = time()
#         X = np.load('X_train.npy', allow_pickle=True)
#         y = np.load('y_train.npy', allow_pickle=True)
#         end_time = time()

#         logging.info(f"Loaded data shape: {X.shape}")
#         logging.info(f"Data loading completed in {end_time - start_time:.2f} seconds.")

#         # Handle string labels
#         le = LabelEncoder()
#         y_encoded = le.fit_transform(y)
#         class_names = le.classes_

#         logging.info(f"Unique classes: {class_names}")

#         # Split the data into training and validation sets
#         X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

#         model, history = train_and_evaluate_model(X_train, X_val, y_train, y_val, class_names)

#         # Plot training history
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(12, 4))
#         plt.subplot(1, 2, 1)
#         plt.plot(history.history['accuracy'], label='Train Accuracy')
#         plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#         plt.title('Model Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()

#         plt.subplot(1, 2, 2)
#         plt.plot(history.history['loss'], label='Train Loss')
#         plt.plot(history.history['val_loss'], label='Validation Loss')
#         plt.title('Model Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.tight_layout()
#         plt.savefig('training_history.png')
#         logging.info("Training history plot saved as 'training_history.png'")

#         # Save the model
#         model.save('improved_genre_classification_model.h5')
#         logging.info("Model saved as 'improved_genre_classification_model.h5'")

#     except FileNotFoundError as e:
#         logging.error(f"Error: Required data file not found. {str(e)}")
#     except ValueError as e:
#         logging.error(f"Error: Invalid data format. {str(e)}")
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()











# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import numpy as np
# import json

# def create_improved_model(input_shape, num_classes):
#     model = Sequential([
#         Dense(512, input_shape=(input_shape,), activation='elu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.4),
        
#         Dense(256, activation='elu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.4),
        
#         Dense(128, activation='elu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         Dense(64, activation='elu', kernel_regularizer=l2(0.001)),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         Dense(num_classes, activation='softmax')
#     ])
    
#     return model

# def train_improved_model(X_train, y_train, X_val, y_val, num_classes, epochs=100, batch_size=256):
#     model = create_improved_model(X_train.shape[1], num_classes)
    
#     optimizer = Adam(learning_rate=0.001)
#     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
#     early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=[early_stopping, reduce_lr],
#         verbose=1
#     )
    
#     # Save the model
#     model.save('improved_genre_classification_model_v2.h5')
    
#     # Save the training history
#     with open('training_history_v2.json', 'w') as f:
#         json.dump(history.history, f)
    
#     return model, history

# # Example usage:
# # Assuming X_train, y_train, X_val, y_val are your training and validation data
# # num_classes = 13  # Number of genre classes

# # model, history = train_improved_model(X_train, y_train, X_val, y_val, num_classes)

# # To evaluate the model:
# # test_loss, test_accuracy = model.evaluate(X_test, y_test)
# # print(f"Test accuracy: {test_accuracy}")

# # To make predictions:
# # predictions = model.predict(X_test)