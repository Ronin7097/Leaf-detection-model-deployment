from flask import Flask, request, redirect, url_for, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict

app = Flask(__name__)

# Configure CORS properly
CORS(app, origins=['http://localhost:5173'], 
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

# Remove the manual CORS headers since flask-cors handles it
# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
#     return response

app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Model and Class Names ---
def get_model(num_classes=44):
    model = models.densenet169(weights=None)
    best_params = {
        'dropout': 0.3038298979358211,
        'activation': 'selu'
    }
    activation_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "selu": nn.SELU
    }
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(best_params["dropout"]),
        activation_map[best_params["activation"]](),
        nn.Linear(in_features, num_classes)
    )
    return model

# --- Load Model ---
model_path = 'densenet169_best_final.pth'
num_classes = 44
model = get_model(num_classes)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

# --- Class Names ---
class_names = [
    "Acacia_Gumefora",
    "Agave_Sisal",
    "Ajuga",
    "Allophylus",
    "Aloe_Ankoberenisis",
    "Aloe_Debrana",
    "Archirantus",
    "Bederjan",
    "Beresemma_Abysinica",
    "Biden",
    "Calpurnia",
    "Carissa_Spinanrum",
    "Chenopodium",
    "Chlorodundurum",
    "Climatis",
    "Clutea",
    "Cordia",
    "Crotun",
    "Dovianus",
    "Eberighia",
    "Echinopes_Kebericho",
    "Ficus_Sur",
    "Hagnia_Abbysinica",
    "Jesminium",
    "Laggeria",
    "Leonotes",
    "Leucas",
    "Linipia_Adonesisis",
    "Lobelia_Rehinopetanum",
    "Melitia",
    "Messa_Lanceolata",
    "Osirus",
    "Phytolleca",
    "Plantago",
    "Rumex_Abbysinica",
    "Rumix_Nervo",
    "Senecio",
    "Stephania_Abbysiniça",
    "Thymus_Schimperia",
    "Uritica",
    "Verbasucum",
    "Vernonia_Amag",
    "Vernonia_Leop",
    "Zeneria_Scabra"
]

# --- Image Transformation ---
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# --- Prediction ---
def predict(image_path, model, class_names):
    image_tensor = transform_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # Get the confidence and the predicted class
        confidence, predicted = torch.max(probabilities, 1)
    return class_names[predicted.item()], confidence.item()

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction, confidence = predict(filepath, model, class_names)
            
            # Clean up the uploaded file
            os.remove(filepath)

            return jsonify({
                'prediction': prediction, 
                'confidence': f'{confidence:.2%}'
            })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Plant classifier API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)