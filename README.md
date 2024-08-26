# Music Genre Classification Project

This project implements a deep learning model to classify music genres based on various audio features extracted from Spotify data.

## Project Overview

The project consists of three main components:
1. Genre mapping
2. Data preprocessing and balancing
3. Model training and evaluation

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS and Linux: `source venv/bin/activate`
4. Install the required packages: `pip install -r requirements.txt`
5. Place your dataset (`spotify_data.csv`) in the `data/` directory
6. Run the preprocessing scripts:
   - `python mapping.py`
   - `python balancing.py`
7. Run the model: `python genre_classification.py`

## Project Structure

- `data/`: Directory for storing the dataset and processed files
- `models/`: Directory for saving trained models
- `src/`: Source code for the project
  - `mapping.py`: Script for mapping specific genres to broader categories
  - `balancing.py`: Script for preprocessing and balancing the dataset
  - `genre_classification.py`: Main script for training and evaluating the model
- `tests/`: Unit tests (to be implemented)
- `requirements.txt`: List of Python dependencies

## Data Processing

### Genre Mapping

The `mapping.py` script maps specific genres to broader categories, reducing the number of unique labels. Key features:
- Creates a manageable set of genre labels
- Outputs `mapped_dataset.csv` and `genre_mapping.csv`

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
- Final balanced dataset: 223,029 samples per class
- Current model performance: 83% accuracy 

## Future Work

- Implement unit tests
- Visualizaton
- Api call
- Experiment with different model architectures
- Feature engineering to improve model performance



