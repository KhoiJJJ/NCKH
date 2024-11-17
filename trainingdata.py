# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:01:45 2024

@author: Admin
"""

import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

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

def load_and_extract_features_from_folders(base_dir, sr=16000, n_mfcc=13):
    audio_features = []
    labels = []
    
    # Iterate over each folder in the base directory
    for class_label in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_label)
        
        # Check if it is a directory (i.e., class folder)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                
                # Check if it's an audio file
                if file_name.endswith('.wav') or file_name.endswith('.mp3'):
                    try:
                        # Load the audio file
                        audio, sr = librosa.load(file_path, sr=sr)
                        
                        # Extract MFCCs with delta and delta-delta
                        mfcc_features = extract_mfcc_with_deltas(audio, sr, n_mfcc)
                        
                        # Flatten the MFCC features for SVM input
                        mfcc_flattened = mfcc_features.flatten()
                        
                        # Append the MFCC features and label to lists
                        audio_features.append(mfcc_flattened)
                        labels.append(class_label)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    
    return np.array(audio_features), np.array(labels)

def train_and_test_svm(train_dir, test_dir, sr=16000, n_mfcc=13):
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_and_extract_features_from_folders(train_dir, sr, n_mfcc)
    
    # Load testing data
    print("Loading testing data...")
    X_test, y_test = load_and_extract_features_from_folders(test_dir, sr, n_mfcc)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train SVM classifier
    print("Training SVM model...")
    svm = SVC(kernel='rbf')  # You can change the kernel type and parameters
    svm.fit(X_train, y_train)
    
    # Make predictions on the test set
    print("Testing SVM model...")
    y_pred = svm.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

'''
def find_optimal_svm_parameters(train_dir, sr=16000, n_mfcc=12):
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_and_extract_features_from_folders(train_dir, sr, n_mfcc)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Define parameter grid for SVM with RBF kernel
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
        'kernel': ['rbf']  # Using RBF kernel
    }
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
    
    # Train SVM using grid search
    print("Finding optimal parameters for SVM with RBF kernel...")
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters
    print(f"Best parameters found: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def evaluate_svm_on_test(svm_model, test_dir, sr=16000, n_mfcc=12):
    # Load testing data
    print("Loading testing data...")
    X_test, y_test = load_and_extract_features_from_folders(test_dir, sr, n_mfcc)
    
    # Standardize the test features
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    
    # Make predictions on the test set
    print("Evaluating SVM model on test data...")
    y_pred = svm_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
'''

# Example usage:
train_directory = r'C:\Users\USER\Downloads\SV_NCKH_audio_event\Train'
test_directory = r'C:\Users\USER\Downloads\SV_NCKH_audio_event\Test'

# Find the optimal SVM model
#best_svm_model = find_optimal_svm_parameters(train_directory)

# Evaluate the best SVM model on the test data
#evaluate_svm_on_test(best_svm_model, test_directory)
train_and_test_svm(train_directory, test_directory, sr=16000, n_mfcc=13)