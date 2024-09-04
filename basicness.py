# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from tqdm import tqdm

# def load_and_preprocess_data(csv_file):
#     print("Loading and preprocessing data...")
#     df = pd.read_csv(csv_file)
    
#     features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
#                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
#                 'duration_ms', 'time_signature']
    
#     print("Calculating top 10% popular songs per genre...")
#     top_10_percent = df.groupby('unified_genre').apply(
#         lambda x: x.nlargest(max(int(len(x) * 0.1), 1), 'popularity')
#     ).reset_index(drop=True)
    
#     print("Calculating mean features for top songs per genre...")
#     genre_means = top_10_percent.groupby('unified_genre')[features].mean()
    
#     print("Calculating basicness scores...")
#     tqdm.pandas()
#     df['basicness'] = df.progress_apply(
#         lambda row: 1 - cosine_similarity(
#             [row[features]], 
#             [genre_means.loc[row['unified_genre']]]
#         )[0][0], 
#         axis=1
#     )
    
#     X = df[features + ['popularity']]
#     y = df['basicness']
    
#     return X, y, genre_means

# def create_model(input_shape):
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(input_shape,)),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
#     return model

# def train_model(X, y):
#     print("Preparing data for training...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     print("Creating and compiling the model...")
#     model = create_model(X_train_scaled.shape[1])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
#     print("Training the model...")
#     history = model.fit(
#         X_train_scaled, y_train,
#         validation_split=0.2,
#         epochs=50,
#         batch_size=64,
#         callbacks=[early_stopping],
#         verbose=1
#     )
    
#     print("Evaluating the model...")
#     test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
#     print(f"Test MAE: {test_mae}")
    
#     return model, scaler

# if __name__ == "__main__":
#     csv_file = 'data\mapped_dataset.csv' 
#     X, y, genre_means = load_and_preprocess_data(csv_file)
#     model, scaler = train_model(X, y)
    
#     print("Saving model, scaler, and genre means...")
#     model.save('basicness_prediction_model.h5')
#     import joblib
#     joblib.dump(scaler, 'basicness_scaler.joblib')
#     genre_means.to_csv('genre_means.csv')
    
#     print("Model, scaler, and genre means saved successfully.")





# pip install tensorflow scikit-learn pandas numpy matplotlib seaborn keras-tuner

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import l2
# from keras_tuner import RandomSearch
# from keras_tuner.engine.hyperparameters import HyperParameters
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# # Define genre mapping
# genre_mapping = {
#     'electronic': 'Electronic/Dance', 'edm': 'Electronic/Dance', 'electro': 'Electronic/Dance',
#     'dance': 'Electronic/Dance', 'club': 'Electronic/Dance', 'breakbeat': 'Electronic/Dance',
#     'drum-and-bass': 'Electronic/Dance', 'dubstep': 'Electronic/Dance', 'garage': 'Electronic/Dance',
#     'hardstyle': 'Electronic/Dance', 'house': 'Electronic/Dance', 'chicago-house': 'Electronic/Dance',
#     'deep-house': 'Electronic/Dance', 'progressive-house': 'Electronic/Dance',
#     'minimal-techno': 'Electronic/Dance', 'techno': 'Electronic/Dance', 'detroit-techno': 'Electronic/Dance',
#     'trance': 'Electronic/Dance',
#     'rock': 'Rock/Metal', 'alt-rock': 'Rock/Metal', 'hard-rock': 'Rock/Metal',
#     'metal': 'Rock/Metal', 'heavy-metal': 'Rock/Metal', 'black-metal': 'Rock/Metal',
#     'death-metal': 'Rock/Metal', 'metalcore': 'Rock/Metal', 'punk': 'Rock/Metal',
#     'punk-rock': 'Rock/Metal', 'emo': 'Rock/Metal', 'goth': 'Rock/Metal',
#     'grindcore': 'Rock/Metal', 'hardcore': 'Rock/Metal', 'industrial': 'Rock/Metal',
#     'pop': 'Pop/Mainstream', 'indie-pop': 'Pop/Mainstream', 'k-pop': 'Pop/Mainstream',
#     'power-pop': 'Pop/Mainstream', 'cantopop': 'Pop/Mainstream',
#     'hip-hop': 'Hip-Hop/R&B', 'trip-hop': 'Hip-Hop/R&B', 'soul': 'Hip-Hop/R&B',
#     'jazz': 'Jazz/Blues', 'blues': 'Jazz/Blues',
#     'classical': 'Classical/Orchestral', 'opera': 'Classical/Orchestral',
#     'folk': 'Folk/Acoustic', 'acoustic': 'Folk/Acoustic', 'singer-songwriter': 'Folk/Acoustic',
#     'songwriter': 'Folk/Acoustic',
#     'afrobeat': 'World Music', 'indian': 'World Music', 'spanish': 'World Music',
#     'french': 'World Music', 'german': 'World Music', 'swedish': 'World Music',
#     'forro': 'World Music', 'sertanejo': 'World Music',
#     'salsa': 'Latin', 'samba': 'Latin', 'tango': 'Latin',
#     'ambient': 'Ambient/Chill', 'chill': 'Ambient/Chill',
#     'dub': 'Reggae/Ska', 'dancehall': 'Reggae/Ska', 'ska': 'Reggae/Ska',
#     'country': 'Country/Americana', 'rock-n-roll': 'Country/Americana',
#     'funk': 'Funk/Disco', 'disco': 'Funk/Disco',
#     'piano': 'Instrumental', 'guitar': 'Instrumental',
#     'romance': 'Miscellaneous', 'sad': 'Miscellaneous', 'pop-film': 'Miscellaneous'
# }

# def load_and_preprocess_data(csv_file):
#     print("Loading and preprocessing data...")
#     df = pd.read_csv(csv_file)
    
#     # Print information about the dataframe
#     print(df.info())
#     print("\nTop 20 most common values in 'unified_genre':")
#     print(df['unified_genre'].value_counts(dropna=False).head(20))
    
#     # Map genres
#     df['mapped_genre'] = df['unified_genre'].map(lambda x: genre_mapping.get(x.lower(), x) if isinstance(x, str) else x)
    
#     # Handle any remaining NaN values in mapped_genre
#     nan_count = df['mapped_genre'].isna().sum()
#     if nan_count > 0:
#         print(f"\nWarning: Found {nan_count} NaN values in 'mapped_genre'. These will be labeled as 'Unknown'.")
#         df['mapped_genre'] = df['mapped_genre'].fillna('Unknown')
    
#     print("\nTop 20 most common values in 'mapped_genre':")
#     print(df['mapped_genre'].value_counts().head(20))
    
#     features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
#                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
#                 'duration_ms', 'time_signature', 'popularity']
    
#     print("\nCalculating top 10% popular songs per mapped genre...")
#     top_10_percent = df.groupby('mapped_genre', group_keys=False).apply(
#         lambda x: x.nlargest(max(int(len(x) * 0.1), 1), 'popularity')
#     )
    
#     print("Calculating mean features for top songs per mapped genre...")
#     genre_means = top_10_percent.groupby('mapped_genre')[features].mean()
    
#     print("Calculating basicness scores...")
#     tqdm.pandas()
#     df['basicness'] = df.progress_apply(
#         lambda row: 1 - cosine_similarity(
#             [row[features]], 
#             [genre_means.loc[row['mapped_genre']]]
#         )[0][0], 
#         axis=1
#     )
    
#     X = df[features]
#     y = df['basicness']
    
#     return X, y, genre_means

# def build_model(hp):
#     input_shape = (14,)  # Adjust this to match your feature count
#     model = Sequential([
#         Input(shape=input_shape),
#         Dense(units=hp.Int('units_1', min_value=32, max_value=512, step=32),
#               activation='relu',
#               kernel_regularizer=l2(hp.Float('l2_1', min_value=1e-5, max_value=1e-2, sampling='LOG'))),
#         BatchNormalization(),
#         Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1))
#     ])
    
#     for i in range(hp.Int('num_layers', 1, 3)):
#         model.add(Dense(units=hp.Int(f'units_{i+2}', min_value=32, max_value=512, step=32),
#                         activation='relu',
#                         kernel_regularizer=l2(hp.Float(f'l2_{i+2}', min_value=1e-5, max_value=1e-2, sampling='LOG'))))
#         model.add(BatchNormalization())
#         model.add(Dropout(hp.Float(f'dropout_{i+2}', min_value=0.0, max_value=0.5, step=0.1)))
    
#     model.add(Dense(1, activation='sigmoid'))
    
#     model.compile(
#         optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
#         loss='mse',
#         metrics=['mae']
#     )
#     return model

# def train_and_evaluate_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     tuner = RandomSearch(
#         build_model,
#         objective='val_mae',
#         max_trials=5,
#         executions_per_trial=1,
#         directory='keras_tuner',
#         project_name='basicness_prediction'
#     )
    
#     try:
#         tuner.search(X_train_scaled, y_train, epochs=50, validation_split=0.2, 
#                      callbacks=[EarlyStopping(patience=10)], verbose=1)
        
#         best_model = tuner.get_best_models(num_models=1)[0]
        
#         history = best_model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, 
#                                  callbacks=[EarlyStopping(patience=10)], verbose=1)
        
#         y_pred = best_model.predict(X_test_scaled).flatten()
        
#         mae = mean_absolute_error(y_test, y_pred)
#         mse = mean_squared_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
        
#         print(f"Mean Absolute Error: {mae}")
#         print(f"Mean Squared Error: {mse}")
#         print(f"R-squared Score: {r2}")
        
#         # Calculate accuracy (within 10% of true value)
#         accuracy = np.mean(np.abs(y_pred - y_test) <= 0.1)
#         print(f"Accuracy (within 10% of true value): {accuracy}")
        
#         plot_history(history)
#         plot_predictions(y_test, y_pred)
#         plot_feature_importance(best_model, X.columns)
        
#         return best_model, scaler
#     except Exception as e:
#         print(f"An error occurred during model training: {str(e)}")
#         return None, None

# def plot_history(history):
#     plt.figure(figsize=(12, 6))
#     plt.plot(history.history['mae'], label='Training MAE')
#     plt.plot(history.history['val_mae'], label='Validation MAE')
#     plt.title('Model Training History')
#     plt.xlabel('Epoch')
#     plt.ylabel('Mean Absolute Error')
#     plt.legend()
#     plt.savefig('training_history.png')
#     plt.close()

# def plot_predictions(y_true, y_pred):
#     plt.figure(figsize=(10, 10))
#     plt.scatter(y_true, y_pred, alpha=0.5)
#     plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
#     plt.xlabel('True Values')
#     plt.ylabel('Predictions')
#     plt.title('True Values vs Predictions')
#     plt.savefig('true_vs_pred.png')
#     plt.close()

# def plot_feature_importance(model, feature_names):
#     importances = np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
#     feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
#     feature_importance = feature_importance.sort_values('importance', ascending=False)
    
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='importance', y='feature', data=feature_importance)
#     plt.title('Feature Importance')
#     plt.savefig('feature_importance.png')
#     plt.close()

# if __name__ == "__main__":
#     csv_file = r'data\mapped_dataset.csv'
#     X, y, genre_means = load_and_preprocess_data(csv_file)
#     best_model, scaler = train_and_evaluate_model(X, y)
    
#     if best_model is not None and scaler is not None:
#         best_model.save('basicness_prediction_model.keras')
#         joblib.dump(scaler, 'basicness_scaler.joblib')
#         genre_means.to_csv('genre_means.csv')
#         print("Model, scaler, and genre means saved successfully.")
#     else:
#         print("Model training failed. No files were saved.")



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ... (keep the genre_mapping dictionary as it was) ...

def load_and_preprocess_data(csv_file):
    print("Loading and preprocessing data...")
    df = pd.read_csv(csv_file)
    
    # Map genres
    df['mapped_genre'] = df['unified_genre'].map(lambda x: genre_mapping.get(x.lower(), x) if isinstance(x, str) else x)
    
    features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                'duration_ms', 'time_signature', 'popularity']
    
    # Feature engineering
    df['energy_danceability_ratio'] = df['energy'] / df['danceability']
    df['loudness_energy_interaction'] = df['loudness'] * df['energy']
    df['tempo_bin'] = pd.qcut(df['tempo'], q=5, labels=['very_slow', 'slow', 'medium', 'fast', 'very_fast'])
    df['popularity_bin'] = pd.qcut(df['popularity'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['tempo_bin', 'popularity_bin', 'mapped_genre'])
    
    print("Calculating top 10% popular songs per mapped genre...")
    top_10_percent = df.groupby('mapped_genre', group_keys=False).apply(
        lambda x: x.nlargest(max(int(len(x) * 0.1), 1), 'popularity')
    )
    
    print("Calculating mean features for top songs per mapped genre...")
    genre_means = top_10_percent.groupby('mapped_genre')[features].mean()
    
    print("Calculating basicness scores...")
    tqdm.pandas()
    df['basicness'] = df.progress_apply(
        lambda row: 1 - cosine_similarity(
            [row[features]], 
            [genre_means.loc[row['mapped_genre']]]
        )[0][0], 
        axis=1
    )
    
    X = df.drop(['basicness', 'unified_genre'], axis=1)
    y = df['basicness']
    
    return X, y, genre_means

def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    
    # Neural Network
    nn_model = build_nn_model(X_train_scaled.shape[1])
    nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, 
                 validation_split=0.2, callbacks=[EarlyStopping(patience=10)], verbose=0)
    nn_pred = nn_model.predict(X_test_scaled).flatten()
    
    # Evaluate models
    models = {
        'Random Forest': (rf_model, rf_pred),
        'Gradient Boosting': (gb_model, gb_pred),
        'Neural Network': (nn_model, nn_pred)
    }
    
    for name, (model, pred) in models.items():
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        accuracy = np.mean(np.abs(pred - y_test) <= 0.1)
        
        print(f"\n{name} Results:")
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared Score: {r2}")
        print(f"Accuracy (within 10% of true value): {accuracy}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        print(f"Cross-validation MAE: {-cv_scores.mean()} (+/- {cv_scores.std() * 2})")
    
    # Error analysis
    best_model, best_pred = max(models.values(), key=lambda x: r2_score(y_test, x[1]))
    errors = np.abs(best_pred - y_test)
    worst_predictions = pd.DataFrame({'true': y_test, 'predicted': best_pred, 'error': errors})
    worst_predictions = worst_predictions.sort_values('error', ascending=False).head(10)
    
    print("\nWorst Predictions:")
    print(worst_predictions)
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': best_model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved as 'feature_importance.png'")
    
    return best_model, scaler

def build_nn_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    csv_file = r'data\mapped_dataset.csv'
    X, y, genre_means = load_and_preprocess_data(csv_file)
    best_model, scaler = train_and_evaluate_models(X, y)
    
    joblib.dump(best_model, 'best_basicness_model.joblib')
    joblib.dump(scaler, 'basicness_scaler.joblib')
    genre_means.to_csv('genre_means.csv')
    print("Best model, scaler, and genre means saved successfully.")