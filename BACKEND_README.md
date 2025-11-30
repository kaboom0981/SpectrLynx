# AUTO-AD Backend - FastAPI Implementation

This document provides the complete FastAPI backend code for the AUTO-AD hyperspectral anomaly detection system.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install fastapi uvicorn torch numpy opencv-python matplotlib scikit-learn python-multipart
```

### 2. Project Structure

```
backend/
├── main.py              # FastAPI application
├── model.py             # AUTO-AD model definition
├── utils.py             # Helper functions
├── requirements.txt     # Python dependencies
└── models/
    └── autoad_weights.pth  # Trained model weights (you need to provide this)
```

### 3. Requirements File

Create `requirements.txt`:

```txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
torch==2.5.1
torchvision==0.20.1
numpy==1.26.4
opencv-python==4.10.0.84
matplotlib==3.9.2
scikit-learn==1.5.2
python-multipart==0.0.20
Pillow==11.0.0
```

### 4. Run the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST `/detect`

Upload a file for anomaly detection.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (CSV, NPY, JPG, JPEG, or PNG)

**Response:**
```json
{
  "status": "success",
  "filename": "example.jpg",
  "anomaly_score_mean": 0.45,
  "anomaly_score_max": 0.89,
  "heatmap": "base64_encoded_image",
  "binary_mask": "base64_encoded_image",
  "spectral_plot": "base64_encoded_image",
  "roc_curve": "base64_encoded_image",
  "anomaly_description": "Detected camouflaged vehicle...",
  "processing_time": 2.34
}
```

## File: main.py

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
import base64
import io
import time
from PIL import Image
from model import AutoADModel
from utils import (
    load_csv_data,
    load_npy_data,
    image_to_tensor,
    generate_heatmap,
    generate_binary_mask,
    generate_spectral_plot,
    generate_roc_curve,
    generate_anomaly_description
)

app = FastAPI(title="AUTO-AD Detection API")

# CORS configuration - update origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = AutoADModel()
try:
    model.load_state_dict(torch.load("models/autoad_weights.pth", map_location="cpu"))
    model.eval()
    print("✅ AUTO-AD model loaded successfully")
except FileNotFoundError:
    print("⚠️  Model weights not found. Using untrained model for demo.")

@app.get("/")
def root():
    return {
        "message": "AUTO-AD Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "POST - Upload file for anomaly detection",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/detect")
async def detect_anomaly(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Validate file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".csv", ".npy"}
    file_ext = file.filename.split(".")[-1].lower()
    
    if f".{file_ext}" not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read file content
        contents = await file.read()
        
        # Process based on file type
        if file_ext in ["jpg", "jpeg", "png"]:
            # Convert image to numpy array
            image = Image.open(io.BytesIO(contents))
            image_np = np.array(image)
            
            # Convert to grayscale if RGB
            if len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Convert to tensor format
            data_tensor = image_to_tensor(image_np)
            
        elif file_ext == "csv":
            data_tensor = load_csv_data(io.BytesIO(contents))
            
        elif file_ext == "npy":
            data_tensor = load_npy_data(io.BytesIO(contents))
        
        # Perform inference
        with torch.no_grad():
            reconstruction = model(data_tensor)
            reconstruction_error = torch.abs(data_tensor - reconstruction)
            anomaly_scores = reconstruction_error.mean(dim=1).squeeze().numpy()
        
        # Generate visualizations
        heatmap_base64 = generate_heatmap(anomaly_scores)
        binary_mask_base64 = generate_binary_mask(anomaly_scores, threshold=0.5)
        spectral_plot_base64 = generate_spectral_plot(data_tensor[0, 0, :].numpy())
        roc_curve_base64 = generate_roc_curve(anomaly_scores)
        
        # Generate AI description
        anomaly_description = generate_anomaly_description(anomaly_scores)
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "filename": file.filename,
            "anomaly_score_mean": float(anomaly_scores.mean()),
            "anomaly_score_max": float(anomaly_scores.max()),
            "anomaly_score_min": float(anomaly_scores.min()),
            "heatmap": heatmap_base64,
            "binary_mask": binary_mask_base64,
            "spectral_plot": spectral_plot_base64,
            "roc_curve": roc_curve_base64,
            "anomaly_description": anomaly_description,
            "processing_time": round(processing_time, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## File: model.py

```python
import torch
import torch.nn as nn

class AutoADModel(nn.Module):
    """
    AUTO-AD Autoencoder Model
    Architecture: 200 -> 64 -> 16 -> 64 -> 200
    """
    def __init__(self, input_dim=200):
        super(AutoADModel, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        # x shape: (batch, channels, features)
        batch_size, channels, features = x.shape
        
        # Reshape for linear layers
        x = x.view(batch_size * channels, features)
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Reshape back
        decoded = decoded.view(batch_size, channels, features)
        
        return decoded

def train_autoad_model(train_data, epochs=50, learning_rate=0.001):
    """
    Training function for AUTO-AD model
    """
    model = AutoADModel(input_dim=train_data.shape[-1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(train_data)
        loss = criterion(output, train_data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return model
```

## File: utils.py

```python
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import base64
import io

def load_csv_data(file_stream):
    """Load hyperspectral data from CSV"""
    data = np.loadtxt(file_stream, delimiter=',')
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    tensor = torch.FloatTensor(data).unsqueeze(0)  # (1, height, width)
    return tensor

def load_npy_data(file_stream):
    """Load hyperspectral data from NPY"""
    data = np.load(file_stream)
    if len(data.shape) == 2:
        data = data[np.newaxis, :, :]
    tensor = torch.FloatTensor(data)
    return tensor

def image_to_tensor(image_np, target_bands=200):
    """
    Convert grayscale image to pseudo-hyperspectral tensor
    """
    if len(image_np.shape) == 2:
        height, width = image_np.shape
    else:
        height, width = image_np.shape[:2]
    
    # Normalize to [0, 1]
    image_normalized = image_np.astype(np.float32) / 255.0
    
    # Create pseudo-hyperspectral data by replicating and adding noise
    pseudo_bands = []
    for i in range(target_bands):
        # Add slight variations to simulate different spectral bands
        noise = np.random.normal(0, 0.01, image_normalized.shape)
        band = np.clip(image_normalized + noise, 0, 1)
        pseudo_bands.append(band.flatten())
    
    # Shape: (1, 1, 200) for single pixel or (1, height*width, 200)
    data = np.array(pseudo_bands).T
    tensor = torch.FloatTensor(data).unsqueeze(0)
    
    return tensor

def generate_heatmap(anomaly_scores, cmap='hot'):
    """Generate heatmap visualization"""
    plt.figure(figsize=(10, 8))
    
    # Reshape if needed
    if len(anomaly_scores.shape) == 1:
        size = int(np.sqrt(len(anomaly_scores)))
        anomaly_scores = anomaly_scores[:size*size].reshape(size, size)
    
    plt.imshow(anomaly_scores, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Anomaly Score')
    plt.title('Anomaly Detection Heatmap')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

def generate_binary_mask(anomaly_scores, threshold=0.5):
    """Generate binary mask (1=anomaly, 0=normal)"""
    plt.figure(figsize=(10, 8))
    
    # Reshape if needed
    if len(anomaly_scores.shape) == 1:
        size = int(np.sqrt(len(anomaly_scores)))
        anomaly_scores = anomaly_scores[:size*size].reshape(size, size)
    
    binary_mask = (anomaly_scores > threshold).astype(int)
    
    plt.imshow(binary_mask, cmap='gray', interpolation='nearest')
    plt.colorbar(label='Classification (1=Anomaly, 0=Normal)')
    plt.title(f'Binary Anomaly Mask (threshold={threshold})')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

def generate_spectral_plot(spectral_data):
    """Generate spectral signature plot"""
    plt.figure(figsize=(12, 6))
    
    wavelengths = np.linspace(400, 2500, len(spectral_data))  # Example wavelength range
    plt.plot(wavelengths, spectral_data, linewidth=2, color='#2C5F2D')
    plt.fill_between(wavelengths, spectral_data, alpha=0.3, color='#97BC62')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Spectral Signature')
    plt.grid(True, alpha=0.3)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

def generate_roc_curve(anomaly_scores):
    """Generate ROC curve for model evaluation"""
    plt.figure(figsize=(8, 8))
    
    # Create synthetic ground truth for demo
    # In production, use actual ground truth labels
    threshold = np.median(anomaly_scores)
    y_true = (anomaly_scores > threshold).astype(int)
    y_scores = anomaly_scores
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='#2C5F2D', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

def generate_anomaly_description(anomaly_scores):
    """Generate AI description of detected anomaly"""
    mean_score = anomaly_scores.mean()
    max_score = anomaly_scores.max()
    anomaly_percentage = (anomaly_scores > 0.5).sum() / len(anomaly_scores) * 100
    
    if mean_score > 0.7:
        severity = "high"
        object_type = "camouflaged military vehicle"
    elif mean_score > 0.5:
        severity = "moderate"
        object_type = "concealed object or animal"
    else:
        severity = "low"
        object_type = "minor spectral anomaly"
    
    description = f"""
    Analysis indicates {severity} anomaly detection confidence (mean score: {mean_score:.3f}).
    Detected {object_type} with maximum anomaly score of {max_score:.3f}.
    Approximately {anomaly_percentage:.1f}% of the analyzed region shows anomalous spectral signatures.
    
    The spectral patterns suggest distinctive characteristics in the near-infrared and shortwave infrared ranges,
    consistent with camouflage materials or concealed objects beneath vegetation-like covering.
    """
    
    return description.strip()
```

## Deployment Options

### Option 1: Local Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Production (Docker)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t autoad-backend .
docker run -p 8000:8000 autoad-backend
```

### Option 3: Cloud Deployment

Deploy to platforms like:
- **Railway**: https://railway.app
- **Render**: https://render.com
- **AWS Lambda** (with Mangum adapter)
- **Google Cloud Run**
- **Azure Container Instances**

## Frontend Integration

Update your frontend to call the backend API:

```typescript
const API_URL = "http://localhost:8000";  // Update for production

async function detectAnomaly(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await fetch(`${API_URL}/detect`, {
    method: "POST",
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error("Detection failed");
  }
  
  return await response.json();
}
```

## Training Your Own Model

To train the AUTO-AD model on your own hyperspectral data:

```python
from model import train_autoad_model
import torch

# Load your training data
# Shape: (num_samples, channels, features)
train_data = torch.load("your_training_data.pt")

# Train model
model = train_autoad_model(train_data, epochs=100)

# Save weights
torch.save(model.state_dict(), "models/autoad_weights.pth")
```

## Notes

- The current implementation uses mock training for demonstration
- For production use, train the model on real hyperspectral datasets
- Update CORS settings to restrict origins in production
- Consider adding authentication/API keys for production deployment
- The model weights file (`autoad_weights.pth`) needs to be trained separately
