# import numpy as np
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.regularizers import l2
# from tqdm.auto import tqdm

# def create_model(input_shape, num_classes):
#     model = Sequential([
#         Dense(512, input_shape=(input_shape,), kernel_regularizer=l2(0.0001)),
#         BatchNormalization(),
#         tf.keras.layers.ELU(),
#         Dropout(0.4),
#         Dense(256, kernel_regularizer=l2(0.0001)),
#         BatchNormalization(),
#         tf.keras.layers.ELU(),
#         Dropout(0.4),
#         Dense(128, kernel_regularizer=l2(0.0001)),
#         BatchNormalization(),
#         tf.keras.layers.ELU(),
#         Dropout(0.4),
#         Dense(64, kernel_regularizer=l2(0.0001)),
#         BatchNormalization(),
#         tf.keras.layers.ELU(),
#         Dropout(0.4),
#         Dense(num_classes, activation='softmax')
#     ])
#     return model

# def train_and_evaluate_model(X, y, class_names):
#     num_classes = len(class_names)
    
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     fold_accuracies = []
    
#     for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
#         print(f"\nTraining on fold {fold}")
#         X_train, X_val = X[train_index], X[val_index]
#         y_train, y_val = y[train_index], y[val_index]
        
#         # Neural Network
#         nn_model = create_model(X_train.shape[1], num_classes)
#         nn_model.compile(optimizer=Adam(learning_rate=0.001),
#                          loss='sparse_categorical_crossentropy',
#                          metrics=['accuracy'])
        
#         early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        
#         nn_model.fit(X_train, y_train, 
#                      validation_data=(X_val, y_val),
#                      epochs=100, 
#                      batch_size=512,
#                      callbacks=[early_stopping, reduce_lr],
#                      verbose=1)
        
#         # Random Forest
#         rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#         rf_model.fit(X_train, y_train)
        
#         # Gradient Boosting
#         gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
#         gb_model.fit(X_train, y_train)
        
#         # Voting Classifier
#         voting_model = VotingClassifier(
#             estimators=[('nn', nn_model), ('rf', rf_model), ('gb', gb_model)],
#             voting='soft'
#         )
#         voting_model.fit(X_train, y_train)
        
#         # Evaluate
#         accuracy = voting_model.score(X_val, y_val)
#         fold_accuracies.append(accuracy)
#         print(f"Fold {fold} Accuracy: {accuracy:.4f}")
        
#         y_pred = voting_model.predict(X_val)
#         print("\nClassification Report:")
#         print(classification_report(y_val, y_pred, target_names=class_names))
        
#         # Feature importance (using Random Forest)
#         importances = rf_model.feature_importances_
#         feature_imp = pd.DataFrame(sorted(zip(importances, range(X.shape[1])), reverse=True),
#                                    columns=['Importance', 'Feature Index'])
#         print("\nTop 10 Important Feature Indices:")
#         print(feature_imp.head(10))
    
#     print(f"\nMean Accuracy across folds: {np.mean(fold_accuracies):.4f}")
#     print(f"Standard Deviation of Accuracy: {np.std(fold_accuracies):.4f}")

# def main():
#     print("Loading preprocessed data...")
#     X_train = np.load('X_train.npy', allow_pickle=True)
#     y_train = np.load('y_train.npy', allow_pickle=True)
    
#     print(f"Loaded training data shape: {X_train.shape}")
    
#     # Encode labels
#     le = LabelEncoder()
#     y_train_encoded = le.fit_transform(y_train)
    
#     class_names = le.classes_
#     print(f"Classes: {class_names}")
    
#     train_and_evaluate_model(X_train, y_train_encoded, class_names)

# if __name__ == "__main__":
#     main()




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

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Suppress TensorFlow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.get_logger().setLevel('ERROR')

# def create_model(input_shape, num_classes):
#     model = Sequential([
#         Dense(256, input_shape=(input_shape,), activation='relu'),
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
#         model.compile(optimizer=Adam(learning_rate=0.001),
#                       loss='sparse_categorical_crossentropy',
#                       metrics=['accuracy'])
        
#         early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        
#         # Custom callback for logging
#         class LoggingCallback(tf.keras.callbacks.Callback):
#             def on_epoch_end(self, epoch, logs=None):
#                 if (epoch + 1) % 5 == 0:
#                     logging.info(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

#         start_time = time()
#         history = model.fit(X_train_scaled, y_train,
#                             validation_data=(X_val_scaled, y_val),
#                             epochs=50,
#                             batch_size=1024,
#                             callbacks=[early_stopping, reduce_lr, LoggingCallback()],
#                             verbose=0)
#         end_time = time()
        
#         logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")

#         # Evaluate
#         y_pred = model.predict(X_val_scaled)
#         y_pred_classes = np.argmax(y_pred, axis=1)
        
#         logging.info("\nClassification Report:")
#         print(classification_report(y_val, y_pred_classes, target_names=class_names))

#         return model

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
        
#         model = train_and_evaluate_model(X_train, X_val, y_train, y_val, class_names)
        
#         # Save the model
#         model.save('genre_classification_model.h5')
#         logging.info("Model saved as 'genre_classification_model.h5'")
        
#     except FileNotFoundError as e:
#         logging.error(f"Error: Required data file not found. {str(e)}")
#     except ValueError as e:
#         logging.error(f"Error: Invalid data format. {str(e)}")
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {str(e)}")

# if __name__ == "__main__":
#        ##0.83




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import logging
from time import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_improved_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(512, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(256, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(128, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(64, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        Dropout(0.4),
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

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        # Neural Network
        model = create_improved_model(X_train_scaled.shape[1], num_classes)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        
        # Custom callback for logging
        class LoggingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 5 == 0:
                    logging.info(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

        start_time = time()
        history = model.fit(X_train_scaled, y_train,
                            validation_data=(X_val_scaled, y_val),
                            epochs=100,  # Increased epochs
                            batch_size=512,  # Reduced batch size
                            callbacks=[early_stopping, reduce_lr, LoggingCallback()],
                            class_weight=class_weight_dict,  # Added class weights
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
        
        # Save the model in Keras format
        model.save('improved_genre_classification_model.keras')
        logging.info("Model saved as 'improved_genre_classification_model.keras'")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()

    ##0.82