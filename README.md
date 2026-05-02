# Medical Image Analysis Platform

This is an advanced AI-powered platform for detecting skin cancer and brain tumors from medical imaging (MRI and skin lesion images). 

It features:
- A responsive, animated, and modern frontend built with HTML, CSS, and vanilla JS.
- Deep Learning inference for classification utilizing ONNX models (`skin_cancer_model.onnx` and `tumor_classifier.onnx`).
- A scalable Flask backend API serving predictions securely and efficiently.

## Features

- **Brain Tumor Detection:** Upload brain MRI scans to check for potential tumors.
- **Skin Cancer Detection:** Analyze skin lesions to check for skin cancer.
- **Flask REST API:** Endpoint-based prediction enabling modular integration.
- **Sleek UI:** Gradient animated backgrounds, floating particles, and fully responsive layouts.

## Project Structure

```
├── app.py                      # Main Flask application and inference logic
├── skin_cancer_model.onnx      # Pre-trained skin cancer classification model
├── tumor_classifier.onnx       # Pre-trained brain tumor classification model
├── requirements.txt            # Python dependencies
├── test_app.py                 # Unit tests for the API
└── templates/                  # Frontend templates
    ├── index.html              # Landing page
    ├── newskin.html            # Skin cancer detection interface
    └── newtumour.html          # Brain tumor detection interface
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Tumour_cancer_det
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python app.py
   ```
   The application will be accessible at `http://127.0.0.1:5000/`.

## API Endpoints

- `POST /predict/skin`
  - Accepts a `multipart/form-data` with an `image` file.
  - Returns: `{"is_malignant": bool, "confidence": float}`
- `POST /predict/tumor`
  - Accepts a `multipart/form-data` with an `image` file.
  - Returns: `{"is_malignant": bool, "confidence": float}`

## Running Tests

We use `pytest` for unit testing. To run the tests, execute:
```bash
pytest test_app.py
```

## Disclaimer
This project is for educational purposes only. Not for clinical use. Always consult a healthcare professional for medical diagnosis.