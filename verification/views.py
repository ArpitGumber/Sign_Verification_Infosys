from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from PIL import Image
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from django.conf import settings
from django.contrib.auth.models import User
from .forms import LoginForm, UploadForm, RegistrationForm

# Load the BiRNN, CNN, and RNN models
birnn_model_path = os.path.join(settings.BASE_DIR, 'verification/models/bi_rnn_signature_verification_model.h5')
cnn_model_path = os.path.join(settings.BASE_DIR, 'verification/models/cnn_signature_verification_model.keras')
rnn_model_path = os.path.join(settings.BASE_DIR, 'verification/models/my_model.h5')

birnn_model = load_model(birnn_model_path)
cnn_model = load_model(cnn_model_path)
rnn_model = load_model(rnn_model_path)

# Preprocess image function for BiRNN
def preprocess_image_birnn(image, img_size=(128, 128), patch_size=(128, 128)):
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = cv2.resize(image, img_size)

    patches = []
    for i in range(0, image.shape[0], patch_size[0]):
        for j in range(0, image.shape[1], patch_size[1]):
            patch = image[i:i+patch_size[0], j:j+patch_size[1]].flatten()
            patches.append(patch)

    patches = np.array(patches)
    patches = np.expand_dims(patches, axis=0)
    return patches

# Preprocess image function for CNN
def preprocess_image_cnn(image, img_size=(128, 128)):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Ensure RGB mode
    image = np.array(image)
    image = cv2.resize(image, img_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

# Preprocess image function for RNN (adjust preprocessing as needed for the RNN model)
def preprocess_image_rnn(image, img_size=(128, 128)):
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = cv2.resize(image, img_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

# Interpret model prediction
def interpret_prediction(prediction):
    predicted_class = "Real" if prediction[0] > 0.5 else "Forged"
    confidence = prediction[0] if predicted_class == "Real" else 1 - prediction[0]
    return predicted_class, confidence

# Predict signature and return combined result
@csrf_exempt
def predict_signature(request):
    if request.method == 'POST' and request.FILES.get('image'):
        response_data = {}
        try:
            uploaded_image = request.FILES['image']
            image = Image.open(uploaded_image)

            # Initialize variables for combining predictions
            predictions = []
            accuracies = []

            # Process and predict with BiRNN
            try:
                processed_image_birnn = preprocess_image_birnn(image)
                birnn_prediction = birnn_model.predict(processed_image_birnn)[0]
                birnn_result, birnn_accuracy = interpret_prediction(birnn_prediction)
                predictions.append(birnn_result)
                accuracies.append(birnn_accuracy)
            except Exception as e:
                response_data['birnn'] = {'error': 'BiRNN model failed'}

            # Process and predict with CNN
            try:
                processed_image_cnn = preprocess_image_cnn(image)
                cnn_prediction = cnn_model.predict(processed_image_cnn)[0]
                cnn_result, cnn_accuracy = interpret_prediction(cnn_prediction)
                predictions.append(cnn_result)
                accuracies.append(cnn_accuracy)
            except Exception as e:
                response_data['cnn'] = {'error': 'CNN model failed'}

            # Process and predict with RNN
            try:
                processed_image_rnn = preprocess_image_rnn(image)
                rnn_prediction = rnn_model.predict(processed_image_rnn)[0]
                rnn_result, rnn_accuracy = interpret_prediction(rnn_prediction)
                predictions.append(rnn_result)
                accuracies.append(rnn_accuracy)
            except Exception as e:
                response_data['rnn'] = {'error': 'RNN model failed'}

            # Combine results
            final_prediction = "Real" if predictions.count("Real") > predictions.count("Forged") else "Forged"
            total_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

            # Add final prediction and accuracy to response
            response_data['final_prediction'] = final_prediction
            response_data['total_accuracy'] = f"{total_accuracy:.2%}"
            response_data['image_url'] = uploaded_image.url  # Ensure the image can be accessed in the template

            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({'error': f"Error processing the image: {str(e)}"}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

# Other views remain unchanged


# Index page (Login page)
def index(request):
    return render(request, 'index.html')

# Logout view
def logout_view(request):
    logout(request)  # Logs the user out
    return redirect('index')  # Redirect to the index or login page after logout

# User registration view
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from .forms import RegistrationForm

def registration(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            confirm_password = form.cleaned_data['confirm_password']

            # Validate password match
            if password != confirm_password:
                messages.error(request, "Passwords do not match.")
                return render(request, 'registration.html', {'form': form})

            # Check if the username or email is already taken
            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already exists.")
                return render(request, 'registration.html', {'form': form})

            if User.objects.filter(email=email).exists():
                messages.error(request, "Email is already taken.")
                return render(request, 'registration.html', {'form': form})

            # Create the user
            try:
                user = User.objects.create_user(username=username, email=email, password=password)
                user.save()
                messages.success(request, "Registration successful! Please log in.")
                return redirect('login')  # Redirect to login page after registration
            except Exception as e:
                messages.error(request, f"Error: {str(e)}")
                return render(request, 'registration.html', {'form': form})

    else:
        form = RegistrationForm()

    return render(request, 'registration.html', {'form': form})


# User login view
def login_view(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user:
                login(request, user)
                return redirect('upload')
            else:
                messages.error(request, "Invalid username or password.")
    else:
        form = LoginForm()

    return render(request, "login.html", {"form": form})

# Upload page (requires user to be logged in)

from django.shortcuts import render, redirect
from .forms import UploadForm
from django.contrib.auth.decorators import login_required
# Ensure only logged-in users can access this view
from django.http import JsonResponse

@login_required
def upload_view(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)  # Handle both form data and uploaded files
        if form.is_valid():
            # Process the image for prediction (use the same logic as in predict_signature)
            uploaded_image = form.cleaned_data['image']
            image = Image.open(uploaded_image)

            # Process and predict with BiRNN
            response_data = {}
            try:
                processed_image_birnn = preprocess_image_birnn(image)
                birnn_prediction = birnn_model.predict(processed_image_birnn)[0]
                birnn_result, birnn_accuracy = interpret_prediction(birnn_prediction)
                response_data['birnn'] = {'result': birnn_result, 'accuracy': f"{birnn_accuracy:.2%}"}
            except Exception as birnn_error:
                response_data['birnn'] = {'error': 'BiRNN model failed'}

            # Process and predict with CNN
            try:
                processed_image_cnn = preprocess_image_cnn(image)
                cnn_prediction = cnn_model.predict(processed_image_cnn)[0]
                cnn_result, cnn_accuracy = interpret_prediction(cnn_prediction)
                response_data['cnn'] = {'result': cnn_result, 'accuracy': f"{cnn_accuracy:.2%}"}
            except Exception as cnn_error:
                response_data['cnn'] = {'error': 'CNN model failed'}

            return JsonResponse(response_data)

        else:
            return render(request, 'upload.html', {'form': form})

    else:
        form = UploadForm()  # Empty form for GET request
    return render(request, 'upload.html', {'form': form})
