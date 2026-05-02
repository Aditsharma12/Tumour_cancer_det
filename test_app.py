import io
import pytest
from PIL import Image
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def create_dummy_image():
    # Create a 224x224 RGB image
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200

def test_skin_page(client):
    response = client.get('/newskin.html')
    assert response.status_code == 200

def test_tumor_page(client):
    response = client.get('/newtumour.html')
    assert response.status_code == 200

def test_predict_skin_no_image(client):
    response = client.post('/predict/skin')
    assert response.status_code == 400
    assert response.get_json()['error'] == 'No image provided'

def test_predict_tumor_no_image(client):
    response = client.post('/predict/tumor')
    assert response.status_code == 400
    assert response.get_json()['error'] == 'No image provided'

def test_predict_skin_with_image(client):
    img_bytes = create_dummy_image()
    data = {'image': (img_bytes, 'test.jpg')}
    response = client.post('/predict/skin', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'is_malignant' in json_data
    assert 'confidence' in json_data

def test_predict_tumor_with_image(client):
    img_bytes = create_dummy_image()
    data = {'image': (img_bytes, 'test.jpg')}
    response = client.post('/predict/tumor', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'is_malignant' in json_data
    assert 'confidence' in json_data
