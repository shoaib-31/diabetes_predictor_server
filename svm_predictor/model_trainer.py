# svm_predictor/model_trainer.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from google.cloud import storage

def train_model(bucket_name: str, file_name: str) -> tuple[StandardScaler, SVC]:
    """
    Trains an SVM model using data from a Google Cloud Storage (GCS) bucket.
    Returns both the scaler and the trained model.

    Args:
        bucket_name (str): Name of the GCS bucket containing the dataset.
        file_name (str): Name of the CSV file containing the diabetes dataset.

    Returns:
        tuple[StandardScaler, SVC]: The fitted standard scaler and the trained SVM model.
    """

    try:
        # Create a GCS client
        client = storage.Client()

        # Download the data from GCS
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Download to a temporary file
        temp_file_name = "downloaded_data.csv"
        blob.download_to_filename(temp_file_name)

        # Load the data into a pandas DataFrame
        diabetes_dataset = pd.read_csv(temp_file_name, delimiter=',')

        # Split data into features and target variable
        X = diabetes_dataset.drop("Outcome", axis=1)
        y = diabetes_dataset["Outcome"]

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Create and train the SVM model
        svm_model = SVC(kernel="linear")
        svm_model.fit(X_train, y_train)

        # Make predictions on test data
        y_pred = svm_model.predict(X_test)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Clean up the temporary file
        import os
        os.remove(temp_file_name)

        return scaler, svm_model  # Return both scaler and model

    except Exception as e:
        print(f"Error accessing data from GCS: {e}")
        return None, None
