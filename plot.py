import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import json
import os

def plot_training_history(history=None, model_path='.h5'):
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




#     python .\genre_classification.py
# 2024-08-30 20:54:41.195643: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2024-08-30 20:54:42.274827: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2024-08-30 20:54:44,813 - INFO - Loading preprocessed data...
# 2024-08-30 20:54:45,124 - INFO - Loaded data shape: (2029563, 30)
# 2024-08-30 20:54:45,124 - INFO - Data loading completed in 0.31 seconds.
# 2024-08-30 20:54:45,504 - INFO - Unique classes: ['Ambient/Chill' 'Country/Folk' 'Electronic/Dance' 'Hip-Hop/R&B'
#  'Instrumental' 'Latin' 'Miscellaneous' 'Other' 'Pop/Mainstream'
#  'Reggae/Ska' 'Rock/Metal' 'Traditional' 'World Music']
# 2024-08-30 20:54:46,383 - INFO - Starting model training and evaluation...
# 2024-08-30 20:54:46,992 - INFO - Feature scaling completed.
# C:\Users\paps_\anaconda3\Lib\site-packages\keras\src\layers\core\dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
#   super().__init__(activity_regularizer=activity_regularizer, **kwargs)
# 2024-08-30 20:54:46.997840: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.        
# To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2024-08-30 20:58:24,167 - INFO - Epoch 5/150 - loss: 0.6769 - accuracy: 0.7734 - val_loss: 0.6312 - val_accuracy: 0.7871
# 2024-08-30 21:02:01,228 - INFO - Epoch 10/150 - loss: 0.6677 - accuracy: 0.7754 - val_loss: 0.6284 - val_accuracy: 0.7870
# 2024-08-30 21:05:38,990 - INFO - Epoch 15/150 - loss: 0.6640 - accuracy: 0.7758 - val_loss: 0.6233 - val_accuracy: 0.7898
# 2024-08-30 21:09:12,540 - INFO - Epoch 20/150 - loss: 0.6149 - accuracy: 0.7850 - val_loss: 0.5581 - val_accuracy: 0.8018
# 2024-08-30 21:12:53,445 - INFO - Epoch 25/150 - loss: 0.5972 - accuracy: 0.7875 - val_loss: 0.5508 - val_accuracy: 0.8025
# 2024-08-30 21:16:29,093 - INFO - Epoch 30/150 - loss: 0.5950 - accuracy: 0.7882 - val_loss: 0.5486 - val_accuracy: 0.8029
# 2024-08-30 21:20:06,578 - INFO - Epoch 35/150 - loss: 0.5766 - accuracy: 0.7928 - val_loss: 0.5319 - val_accuracy: 0.8076
# 2024-08-30 21:23:39,397 - INFO - Epoch 40/150 - loss: 0.5737 - accuracy: 0.7929 - val_loss: 0.5288 - val_accuracy: 0.8080
# 2024-08-30 21:27:06,272 - INFO - Epoch 45/150 - loss: 0.5724 - accuracy: 0.7932 - val_loss: 0.5274 - val_accuracy: 0.8084
# 2024-08-30 21:30:32,444 - INFO - Epoch 50/150 - loss: 0.5721 - accuracy: 0.7932 - val_loss: 0.5269 - val_accuracy: 0.8081
# 2024-08-30 21:33:59,201 - INFO - Epoch 55/150 - loss: 0.5707 - accuracy: 0.7937 - val_loss: 0.5266 - val_accuracy: 0.8084
# 2024-08-30 21:37:25,655 - INFO - Epoch 60/150 - loss: 0.5702 - accuracy: 0.7937 - val_loss: 0.5249 - val_accuracy: 0.8089
# 2024-08-30 21:40:51,844 - INFO - Epoch 65/150 - loss: 0.5701 - accuracy: 0.7937 - val_loss: 0.5252 - val_accuracy: 0.8086
# 2024-08-30 21:44:18,764 - INFO - Epoch 70/150 - loss: 0.5662 - accuracy: 0.7951 - val_loss: 0.5220 - val_accuracy: 0.8095
# 2024-08-30 21:47:45,553 - INFO - Epoch 75/150 - loss: 0.5652 - accuracy: 0.7955 - val_loss: 0.5213 - val_accuracy: 0.8096
# 2024-08-30 21:51:11,491 - INFO - Epoch 80/150 - loss: 0.5644 - accuracy: 0.7954 - val_loss: 0.5207 - val_accuracy: 0.8099
# 2024-08-30 21:54:38,068 - INFO - Epoch 85/150 - loss: 0.5645 - accuracy: 0.7953 - val_loss: 0.5208 - val_accuracy: 0.8096
# 2024-08-30 21:58:04,448 - INFO - Epoch 90/150 - loss: 0.5640 - accuracy: 0.7955 - val_loss: 0.5198 - val_accuracy: 0.8100
# 2024-08-30 22:01:31,235 - INFO - Epoch 95/150 - loss: 0.5642 - accuracy: 0.7952 - val_loss: 0.5204 - val_accuracy: 0.8098
# 2024-08-30 22:04:58,562 - INFO - Epoch 100/150 - loss: 0.5635 - accuracy: 0.7956 - val_loss: 0.5198 - val_accuracy: 0.8102
# 2024-08-30 22:08:24,114 - INFO - Epoch 105/150 - loss: 0.5640 - accuracy: 0.7953 - val_loss: 0.5196 - val_accuracy: 0.8100
# 2024-08-30 22:11:50,809 - INFO - Epoch 110/150 - loss: 0.5636 - accuracy: 0.7952 - val_loss: 0.5193 - val_accuracy: 0.8099
# 2024-08-30 22:15:17,387 - INFO - Epoch 115/150 - loss: 0.5632 - accuracy: 0.7958 - val_loss: 0.5196 - val_accuracy: 0.8100
# 2024-08-30 22:18:43,953 - INFO - Epoch 120/150 - loss: 0.5632 - accuracy: 0.7957 - val_loss: 0.5188 - val_accuracy: 0.8103
# 2024-08-30 22:22:09,504 - INFO - Epoch 125/150 - loss: 0.5630 - accuracy: 0.7956 - val_loss: 0.5189 - val_accuracy: 0.8101
# 2024-08-30 22:25:35,102 - INFO - Epoch 130/150 - loss: 0.5627 - accuracy: 0.7956 - val_loss: 0.5184 - val_accuracy: 0.8104
# 2024-08-30 22:29:02,104 - INFO - Epoch 135/150 - loss: 0.5631 - accuracy: 0.7955 - val_loss: 0.5186 - val_accuracy: 0.8104
# 2024-08-30 22:31:05,495 - INFO - Training completed in 5778.41 seconds.
# 12685/12685 ━━━━━━━━━━━━━━━━━━━━ 13s 998us/step  
# 2024-08-30 22:31:22,117 - INFO - 
# Classification Report:
#                   precision    recall  f1-score   support

#    Ambient/Chill       0.82      0.84      0.83     31224
#     Country/Folk       0.72      0.76      0.74     31224
# Electronic/Dance       0.77      0.62      0.69     31224
#      Hip-Hop/R&B       0.82      0.86      0.84     31224
#     Instrumental       0.72      0.75      0.74     31224
#            Latin       1.00      1.00      1.00     31224
#    Miscellaneous       0.68      0.85      0.75     31224
#            Other       0.63      0.51      0.56     31224
#   Pop/Mainstream       1.00      1.00      1.00     31224
#       Reggae/Ska       0.84      0.90      0.87     31224
#       Rock/Metal       0.75      0.81      0.78     31224
#      Traditional       1.00      1.00      1.00     31225
#      World Music       0.77      0.64      0.70     31224

#         accuracy                           0.81    405913
#        macro avg       0.81      0.81      0.81    405913
#     weighted avg       0.81      0.81      0.81    405913

# 2024-08-30 22:31:23,160 - INFO - Training history plot saved as 'training_history.png'
# 2024-08-30 22:31:23,566 - INFO - Confusion matrix saved as 'confusion_matrix.png'
# 2024-08-30 22:31:23,566 - WARNING - You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# 2024-08-30 22:31:23,612 - INFO - Model saved as 'improved_genre_classification_model.h5'
# 2024-08-30 22:31:23,773 - INFO - Feature importance plot saved as 'feature_importance.png'