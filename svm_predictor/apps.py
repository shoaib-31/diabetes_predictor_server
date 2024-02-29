# svm_predictor/apps.py

from django.apps import AppConfig
from .model_trainer import train_model
import os
os.environ["GCLOUD_PROJECT"] = "test"
import pickle
import json

class SvmPredictorConfig(AppConfig):
    name = 'svm_predictor'

    def ready(self):
        # Call the train function when the server starts
        # Adjust the bucket name and file name accordingly
        bucket_name = "chisquarex-assignment"
        file_name = "diabetes.csv"
        scaler, svm_model = train_model(bucket_name, file_name)

        if scaler and svm_model:
            # Save the trained model and scaler using pickle
            with open('svm_model.pkl', 'wb') as f:
                pickle.dump(svm_model, f)
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            print({'message': 'Model trained successfully'})
        else:
            print({'error': 'Failed to train model'})