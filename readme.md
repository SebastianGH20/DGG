# Music Explorer and Genre Classification

## Project Overview

This project implements a deep learning model to classify and predict music genres based on various audio features extracted from Spotify data. It includes a web application for exploring music tracks, their features, and predicted genres.

## Team Members
- Sebastian Gonzalez 
- Pablo García del Moral

## Features

1. Genre mapping and classification
2. Data preprocessing and balancing
3. Deep learning model for genre prediction
4. Interactive web application for music exploration

## Project Structure

- `data/`: Directory for storing datasets and processed files
- `models/`: Directory for saving trained models
- `src/`: Source code for the project
  - `genre_model.py`: Main script for training and evaluating the genre classification model
  - `main.py`: Streamlit web application for music exploration
  - `dataset_generator.py`: Script for preprocessing the Spotify dataset
  - `test1.py`: Script for testing the trained model
- `requirements.txt`: List of Python dependencies

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Spotify API credentials in a `.env` file:
   ```
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   ```

## Usage

1. Preprocess the data:
   ```
   python dataset_generator.py
   ```
2. Train the genre classification model:
   ```
   python genre_model.py
   ```
3. Run the web application:
   ```
   streamlit run main.py
   ```

## Model Architecture

The genre classification model uses a deep neural network with the following characteristics:
- Multiple dense layers with BatchNormalization and Dropout
- ReLU activation functions
- Adam optimizer
- Early stopping and learning rate reduction on plateau

## Performance

The current model achieves approximately 95% accuracy on the test set.

## Future Work

- Implement additional music recommendation features
- Experiment with different model architectures
- Improve API integration and error handling
- Enhance visualization capabilities in the web application

## Acknowledgments

This project uses the Spotify API for retrieving track information and audio features.

