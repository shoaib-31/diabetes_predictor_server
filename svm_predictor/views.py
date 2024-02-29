# svm_predictor/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .model_trainer import train_model
import pickle

@csrf_exempt
# svm_predictor/views.py


@csrf_exempt
def train(request):
    if request.method == 'POST':
        try:
            # Assuming JSON input containing bucket_name and file_name
            data = json.loads(request.body)
            bucket_name = data.get('bucket_name')
            file_name = data.get('file_name')

            # Train the model
            scaler, svm_model = train_model(bucket_name, file_name)

            if scaler and svm_model:
                # Save the trained model and scaler using pickle
                with open('svm_model.pkl', 'wb') as f:
                    pickle.dump(svm_model, f)
                with open('scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)

                return JsonResponse({'message': 'Model trained successfully'}, status=200)
            else:
                return JsonResponse({'error': 'Failed to train model'}, status=500)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Load the trained model and scaler using pickle
            with open('svm_model.pkl', 'rb') as f:
                svm_model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            # Assuming JSON input containing patient data
            data = json.loads(request.body)
            input_data = data.get('input_data')

            # Perform prediction using the loaded model and scaler
            prediction = svm_model.predict(scaler.transform([input_data]))

            # Return prediction result
            if prediction:
                return JsonResponse({'prediction': 'The patient is diabetic'}, status=200)
            else:
                return JsonResponse({'prediction': 'The patient is not diabetic'}, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)
