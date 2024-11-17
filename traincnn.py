# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:42:13 2024

@author: Admin
"""
import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models


def extract_mfcc_with_deltas(audio, sr=16000, n_mfcc=13):
    """
    Extract MFCCs, delta, and delta-delta features from an audio signal.
    """
    hop_length = int(0.01 * sr)  # 10 ms hop length
    win_length = int(0.025 * sr)  # 25 ms window length
    
    # Calculate MFCCs (including the 0th coefficient)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length, window='hamming')
    
    # Calculate log energy feature
    # energy = np.log(np.sum(librosa.feature.rms(y=audio, hop_length=hop_length)**2, axis=0))
    
    # Append log energy as the 0th feature
    # mfccs[0, :] = energy
    
    # Compute the deltas (first derivative)
    mfcc_delta = librosa.feature.delta(mfccs)
    
    # Compute delta-delta (second derivative)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Stack MFCCs, delta, and delta-delta features
    mfcc_combined = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
    
    return mfcc_combined


'''
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # 2D Convolution Layer
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Additional Conv Layers
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
'''

# Function to load data from folders
def load_data(data_dir):
    X = []
    y = []
    labels = os.listdir(data_dir)
    
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(label_dir):
            if file_name.endswith(".wav"):
                file_path = os.path.join(label_dir, file_name)
                # Load the audio file
                audio, sr = librosa.load(file_path, sr=None)
                features = extract_mfcc_with_deltas(audio,sr)
                X.append(features)
                y.append(label)
    
    return np.array(X), np.array(y)

# Load training and testing data
train_data_dir = r'C:\Users\USER\Downloads\SV_NCKH_audio_event\Train'
test_data_dir = r'C:\Users\USER\Downloads\SV_NCKH_audio_event\Test'


X_train, y_train = load_data(train_data_dir)
X_test, y_test = load_data(test_data_dir)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Reshape for CNN input (add channel dimension)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build CNN model
def create_cnn_model(input_shape, num_classes):
    model = tf.keras.Sequential()

    # 2D Convolution Layer
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Additional Conv Layers
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Fully Connected Layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    # Output Layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # Create the Adam optimizer with the custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Get input shape and number of classes
input_shape = X_train.shape[1:]  # shape of each input sample
num_classes = len(np.unique(y_train))  # number of class labels

# Create and train the CNN model
model = create_cnn_model(input_shape, num_classes)
model.summary()

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")