import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tqdm.auto import tqdm

def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(512, input_shape=(input_shape,), kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        tf.keras.layers.ELU(),
        Dropout(0.4),
        Dense(256, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        tf.keras.layers.ELU(),
        Dropout(0.4),
        Dense(128, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        tf.keras.layers.ELU(),
        Dropout(0.4),
        Dense(64, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        tf.keras.layers.ELU(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_evaluate_model(X, y, class_names):
    num_classes = len(class_names)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"\nTraining on fold {fold}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Neural Network
        nn_model = create_model(X_train.shape[1], num_classes)
        nn_model.compile(optimizer=Adam(learning_rate=0.001),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        
        nn_model.fit(X_train, y_train, 
                     validation_data=(X_val, y_val),
                     epochs=100, 
                     batch_size=512,
                     callbacks=[early_stopping, reduce_lr],
                     verbose=1)
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        
        # Voting Classifier
        voting_model = VotingClassifier(
            estimators=[('nn', nn_model), ('rf', rf_model), ('gb', gb_model)],
            voting='soft'
        )
        voting_model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = voting_model.score(X_val, y_val)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold} Accuracy: {accuracy:.4f}")
        
        y_pred = voting_model.predict(X_val)
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=class_names))
        
        # Feature importance (using Random Forest)
        importances = rf_model.feature_importances_
        feature_imp = pd.DataFrame(sorted(zip(importances, range(X.shape[1])), reverse=True),
                                   columns=['Importance', 'Feature Index'])
        print("\nTop 10 Important Feature Indices:")
        print(feature_imp.head(10))
    
    print(f"\nMean Accuracy across folds: {np.mean(fold_accuracies):.4f}")
    print(f"Standard Deviation of Accuracy: {np.std(fold_accuracies):.4f}")

def main():
    print("Loading preprocessed data...")
    X_train = np.load('X_train.npy', allow_pickle=True)
    y_train = np.load('y_train.npy', allow_pickle=True)
    
    print(f"Loaded training data shape: {X_train.shape}")
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    class_names = le.classes_
    print(f"Classes: {class_names}")
    
    train_and_evaluate_model(X_train, y_train_encoded, class_names)

if __name__ == "__main__":
    main()