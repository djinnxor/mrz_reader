# MRZ Reader FastAPI

A FastAPI service for extracting MRZ (Machine Readable Zone) data from passport images using OCR technology.

## Features

- 🔍 **MRZ Extraction**: Extract passport information from MRZ zones
- 👤 **Face Detection**: Optional face detection and extraction
- 🖼️ **Base64 Support**: Accept images as base64 encoded strings
- 🚀 **FastAPI**: Modern, fast web framework with automatic API documentation
- 📋 **Structured Output**: JSON response with parsed passport data

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd mrz_reader_fastapi
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://localhost:5006`

### API Endpoints

#### 1. Health Check

```bash
GET /health
```

#### 2. Root Endpoint

```bash
GET /
```

#### 3. MRZ Prediction (Main Endpoint)

```bash
POST /predict
```

**Request Body**:

```json
{
  "image_base64": "base64_encoded_image_data",
  "do_facedetect": true,
  "preprocess_config": {
    "do_preprocess": true,
    "skewness": true,
    "delete_shadow": true,
    "clear_background": true
  }
}
```

**Response**:

```json
{
  "success": true,
  "mrz_data": {
    "document_type": "P",
    "country_code": "USA",
    "surname": "DOE",
    "given_names": "JOHN",
    "passport_number": "123456789",
    "nationality": "USA",
    "birth_date": "1990-01-01",
    "gender": "Male",
    "expiry_date": "2030-01-01"
  },
  "face_base64": "base64_encoded_face_image"
}
```

### Example using cURL

```bash
# Convert image to base64 first
BASE64_IMAGE=$(base64 -i ./test/OLD_PASSPORT.jpg)

# Send request
curl -X POST "http://localhost:5006/predict" \
     -H "Content-Type: application/json" \
     -d "{
       \"image_base64\": \"$BASE64_IMAGE\",
       \"do_facedetect\": true,
       \"preprocess_config\": {
         \"do_preprocess\": true,
         \"skewness\": true,
         \"delete_shadow\": true,
         \"clear_background\": true
       }
     }"
```

### Example using Python requests

```python
import base64
import requests

# Convert image to base64
with open("./test/OLD_PASSPORT.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Send request
response = requests.post("http://localhost:5006/predict", json={
    "image_base64": image_base64,
    "do_facedetect": True,
    "preprocess_config": {
        "do_preprocess": True,
        "skewness": True,
        "delete_shadow": True,
        "clear_background": True
    }
})

result = response.json()
print(result)
```

## API Documentation

Once the server is running, you can access:

- **Swagger UI**: http://localhost:5006/docs
- **ReDoc**: http://localhost:5006/redoc

## Configuration

### Environment Variables

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `5006`)

### Preprocessing Options

The `preprocess_config` parameter accepts:

- `do_preprocess`: Enable/disable preprocessing
- `skewness`: Correct image skewness
- `delete_shadow`: Remove shadows from image
- `clear_background`: Clear background noise

## Model Files

The application may require additional model files for full functionality:

- Face detection model: `./model/face_detector/`
- MRZ segmentation model: `./model/mrz_detector/`

If these files are missing, the application will fall back to basic OCR mode.

## Dependencies

- FastAPI: Web framework
- EasyOCR: OCR engine
- OpenCV: Image processing
- TensorFlow: Deep learning models
- NumPy, SciPy: Numerical computing

## Error Handling

The API handles various error cases:

- Invalid base64 image format
- Missing or corrupted image data
- OCR processing failures
- Model loading issues

All errors are returned in a structured JSON format with appropriate HTTP status codes.

## License

See LICENSE file for details.
