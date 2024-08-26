import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback

# Define intermediate genre mapping (excluding 'other')
intermediate_genre_mapping = {
    'rock_metal': ['rock', 'alt-rock', 'hard-rock', 'psych-rock', 'punk-rock', 'rock-n-roll', 
                   'metal', 'black-metal', 'death-metal', 'heavy-metal', 'metalcore'],
    'pop_indie': ['pop', 'indie-pop', 'k-pop', 'power-pop', 'cantopop'],
    'electronic_edm': ['electronic', 'ambient', 'breakbeat', 'chill', 'drum-and-bass', 'dubstep', 
                       'edm', 'electro', 'garage', 'hardstyle', 'minimal-techno', 'techno', 'trance', 'trip-hop'],
    'house': ['house', 'chicago-house', 'deep-house', 'progressive-house'],
    'hip_hop': ['hip-hop'],
    'jazz_blues': ['jazz', 'blues'],
    'classical': ['classical', 'opera'],
    'country_folk': ['country', 'folk'],
    'latin': ['salsa', 'samba', 'tango'],
    'funk_soul': ['funk', 'soul'],
    'reggae_dub': ['reggae', 'dub', 'dancehall'],
    'world': ['afrobeat', 'indian', 'forro', 'sertanejo'],
    'acoustic_instrumental': ['acoustic', 'piano']
}

def get_intermediate_genre(genre):
    for intermediate_genre, genres in intermediate_genre_mapping.items():
        if genre in genres:
            return intermediate_genre
    return None  # Return None for genres not in our mapping

def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path)
    
    # Create intermediate genre column and filter out rows with None (i.e., 'other' genres)
    df['intermediate_genre'] = df['genre'].apply(get_intermediate_genre)
    df = df.dropna(subset=['intermediate_genre'])
    
    X = df.drop(['unified_genre', 'genre', 'intermediate_genre'], axis=1)
    y = df['intermediate_genre']
   
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
   
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
        ],
        remainder='passthrough'
    )
   
    if len(categorical_features) > 0:
        preprocessor.transformers.append(
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        )
   
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=min(15, X_preprocessed.shape[1]))  # Select top 15 features or all if less than 15
    X_selected = selector.fit_transform(X_preprocessed, y)
    
    feature_names = list(numeric_features)
    if len(categorical_features) > 0:
        feature_names += list(
            preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        )
    selected_feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
   
    print(f"Shape after preprocessing and feature selection: {X_selected.shape}")
    print(f"Selected features: {selected_feature_names}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X_selected, y_encoded, selected_feature_names, le

def balance_dataset(X, y, sample_size=None):
    print("Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Shape after SMOTE: {X_resampled.shape}")
    
    if sample_size and sample_size < X_resampled.shape[0]:
        print(f"Sampling {sample_size} instances for training...")
        indices = np.random.choice(X_resampled.shape[0], sample_size, replace=False)
        X_sampled = X_resampled[indices]
        y_sampled = y_resampled[indices]
        print(f"Shape after sampling: {X_sampled.shape}")
        return X_sampled, y_sampled
    return X_resampled, y_resampled

def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, input_shape=(input_shape,), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(64, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(32, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model

class CustomTqdmCallback(TqdmCallback):
    def __init__(self, epochs, **kwargs):
        super().__init__(**kwargs)
        self.epochs = epochs
        self.overall_pbar = None

    def on_train_begin(self, logs=None):
        self.overall_pbar = tqdm(total=self.epochs, desc="Training Progress")

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.overall_pbar.update(1)

    def on_train_end(self, logs=None):
        self.overall_pbar.close()

def train_and_evaluate_model(X, y, feature_names, label_encoder):
    num_classes = len(label_encoder.classes_)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"\nTraining on fold {fold}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Train and evaluate neural network
        model = create_model(X_train.shape[1], num_classes)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        epochs = 50
        batch_size = 512
        custom_tqdm = CustomTqdmCallback(epochs=epochs, leave=True)
        
        history = model.fit(X_train, y_train, 
                            validation_data=(X_val, y_val),
                            epochs=epochs, 
                            batch_size=batch_size,
                            callbacks=[early_stopping, custom_tqdm],
                            verbose=0)
        
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(val_accuracy)
        print(f"Fold {fold} Validation Accuracy: {val_accuracy:.4f}")
        
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        print("\nNeural Network Classification Report:")
        print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_))
        
        # Train and evaluate logistic regression
        lr_model = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        lr_accuracy = lr_model.score(X_val, y_val)
        print(f"\nLogistic Regression Accuracy: {lr_accuracy:.4f}")
        
        # Feature importance analysis
        perm_importance = permutation_importance(lr_model, X_val, y_val, n_repeats=10, random_state=42)
        feature_importance = perm_importance.importances_mean
        feature_importance = feature_importance / np.sum(feature_importance)
        top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)
        print("\nFeature Importance:")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")
    
    print(f"\nMean Accuracy across folds: {np.mean(fold_accuracies):.4f}")
    print(f"Standard Deviation of Accuracy: {np.std(fold_accuracies):.4f}")

def main():
    file_path = 'dataset_limpio_unificado.csv'
    sample_size = 1000000  # Adjust based on your computational resources
    
    with tqdm(total=3, desc="Overall Progress") as pbar:
        X, y, feature_names, label_encoder = load_and_preprocess_data(file_path)
        pbar.update(1)
        
        X_resampled, y_resampled = balance_dataset(X, y, sample_size)
        pbar.update(1)
        
        train_and_evaluate_model(X_resampled, y_resampled, feature_names, label_encoder)
        pbar.update(1)

if __name__ == "__main__":
    main()