import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import json
import os

def plot_training_history(history=None, model_path='genre_classification_model.h5'):
    if history is None:
        # Try to load history from a JSON file if it exists
        history_path = 'training_history.json'
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            print("No saved history found. Only model summary will be displayed.")
            model = load_model(model_path)
            model.summary()
            return

    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history_plot.png')
    print("Plot saved as 'training_history_plot.png'")
    plt.show()

if __name__ == "__main__":
    plot_training_history()