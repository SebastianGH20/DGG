# Music Genre Classification Project

This project implements a deep learning model to classify and predcit music stats based on various audio features extracted from Spotify data.

# Members
 - Sebastian
 - Pablo García del Moral


## Project Overview

The project consists of three main components:
1. Genre mapping
2. Data preprocessing and balancing
3. Model training and evaluation

## Setup

1. Clone the repository
2. Run the preprocessing scripts:
   - `python mapping.py`
   - `python balancing.py`
3. Run the model: `python genre_classification.py`
4. To run the visualization in local: `streamlit run main.py`

## Project Structure

- `data/`: Directory for storing the dataset and processed files
- `models/`: Directory for saving trained models
- `src/`: Source code for the project
  - `mapping.py`: Script for mapping specific genres to broader categories
  - `balancing.py`: Script for preprocessing and balancing the dataset
  - `genre_classification.py`: Main script for training and evaluating the model
- `tests/`: Unit tests (to be implemented)
- `requirements.txt`: List of Python dependencies

### Genre Mapping

The `mapping.py` script maps specific genres to broader categories, reducing the number of unique labels. Key features:
- Creates a manageable set of genre labels (13 genres)
- Outputs `mapped_dataset.csv` dataset with all number unless the names of the genres, and `genre_mapping.csv` displaying the subgenres later joined in the unique labels.

### Data Preprocessing and Balancing

The `balancing.py` script prepares the data for model training. Key steps:
- Applies genre mapping
- Handles class imbalance using SMOTE
- Encodes categorical features
- Splits data into train, validation, and test sets
- Saves processed datasets as numpy arrays

## Model

The genre classification model is implemented in `genre_classification.py`. It uses a deep neural network with the following characteristics:
- Multiple dense layers with L2 regularization
- Batch normalization and LeakyReLU activation
- Dropout for regularization
- Handles class imbalance through class weights

## Project Timeline and Results

- Initial data gathering: Used `spotify_data.csv`
- EDA: Normalized features, consolidated genres, transformed categorical features
- Early models: Achieved up to 30% accuracy
- Improved mapping: Reduced to 13 genre categories
- Final balanced dataset: 1140681 rows
- Current model performance: 83% accuracy 

## Future Work

- Visualizaton (eleccion de cancion y  artista parea calculo de genero y luego de prediccion)
- Api call
- Experiment with different model architectures
- connect api model and visualization

# Things we tried

  relu elu, distinta cantidad de neuronas, Feature scaling, distintos modelos a la vez que la red neuronal,
  k-fold no hemos probado por que tarda mucho, cambiado epoch y batch size tambien,
  callbacks=[early_stopping, reduce_lr, LoggingCallback()] 
  intentamos con convolucionales por una recomendacion. ReduceLROnPlateau
  cambiado el optimizados y las capas 

# Presentación final

El último día de clases, cada equipo presentará su proyecto final al resto de clase y a otras personas de 4Geeks Academy. Deben ser presentaciones enfocadas en el diseño, la tecnología y los retos y problemas superados, así que no tienes por qué preocuparte respecto a hacer una presentación con mucho protocolo. Muestra tu aplicación, explica lo que has hecho, qué cosas especiales has implementado y comparte la experiencia con el resto del curso.




Training Progress with Adam Optimizer:

  Your model was trained for 50 epochs using the Adam optimizer.
  The training completed successfully with the following results:
  Validation Loss: 0.4656
  Validation Accuracy: 82.64%
  The classification report shows reasonably good precision, recall, and F1-score values for most classes, suggesting the model performs well.


Training Metrics adamw

  Epochs: 50
  Final Training Loss: 0.5095
  Final Training Accuracy: 81.27%
  Final Validation Loss: 0.4654
  Final Validation Accuracy: 82.61%
  Training Time: 1037.71 seconds (approximately 17 minutes)
  

Training Metrics with SGD

  Epochs: 50
  Final Training Loss: 0.5365
  Final Training Accuracy: 80.36%
  Final Validation Loss: 0.4888
  Final Validation Accuracy: 81.79%
  Training Time: 1026.19 seconds (approximately 17 minutes) 
  
Training Metrics  rmsprop

  Epochs: 50
  Final Training Loss: 0.5140
  Final Training Accuracy: 81.16%
  Final Validation Loss: 0.4689
  Final Validation Accuracy: 82.52%
  Training Time: 973.78 seconds (approximately 16 minutes)


Feature names after preprocessing: ['num__popularity' 'num__danceability' 'num__energy' 'nu
 'num__loudness' 'num__mode' 'num__speechiness' 'num__acousticness'
 'num__instrumentalness' 'num__liveness' 'num__valence' 'num__tempo'
 'num__duration_ms' 'num__time_signature' 'cat__unified_genre_blues'
 'cat__unified_genre_classical' 'cat__unified_genre_country'
 'cat__unified_genre_electronic' 'cat__unified_genre_folk'
 'cat__unified_genre_funk_soul' 'cat__unified_genre_hip-hop'
 'cat__unified_genre_house' 'cat__unified_genre_jazz'
 'cat__unified_genre_latin' 'cat__unified_genre_metal'
 'cat__unified_genre_pop' 'cat__unified_genre_reggae'
 'cat__unified_genre_rock' 'cat__unified_genre_world']