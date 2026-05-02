import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import onnxruntime as ort

app = Flask(__name__)

# Load models
SKIN_MODEL_PATH = "skin_cancer_model.onnx"
TUMOR_MODEL_PATH = "tumor_classifier.onnx"

skin_session = None
tumor_session = None

if os.path.exists(SKIN_MODEL_PATH):
    skin_session = ort.InferenceSession(SKIN_MODEL_PATH)
if os.path.exists(TUMOR_MODEL_PATH):
    tumor_session = ort.InferenceSession(TUMOR_MODEL_PATH)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_data = np.array(img).astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data - mean) / std
    
    # HWC to CHW
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # Add batch dimension
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    return img_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/newskin.html')
def skin():
    return render_template('newskin.html')

@app.route('/newtumour.html')
def tumor():
    return render_template('newtumour.html')

@app.route('/predict/skin', methods=['POST'])
def predict_skin():
    if not skin_session:
        return jsonify({'error': 'Skin model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    try:
        input_tensor = preprocess_image(file.read())
        input_name = skin_session.get_inputs()[0].name
        outputs = skin_session.run(None, {input_name: input_tensor})
        predictions = outputs[0][0]
        
        is_malignant = bool(predictions[1] > predictions[0])
        confidence = float(np.max(predictions) * 100)
        
        return jsonify({
            'is_malignant': is_malignant,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/tumor', methods=['POST'])
def predict_tumor():
    if not tumor_session:
        return jsonify({'error': 'Tumor model not loaded'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    try:
        input_tensor = preprocess_image(file.read())
        input_name = tumor_session.get_inputs()[0].name
        outputs = tumor_session.run(None, {input_name: input_tensor})
        predictions = outputs[0][0]
        
        is_malignant = bool(predictions[1] > predictions[0])
        confidence = float(np.max(predictions) * 100)
        
        return jsonify({
            'is_malignant': is_malignant,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
