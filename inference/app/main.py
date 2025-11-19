"""
FastAPI application for MNIST digit classification inference.

This application:
1. Loads a trained model on startup from persistent storage
2. Provides a web UI for image upload
3. Performs inference on uploaded images
4. Returns digit predictions with confidence scores
"""
import io
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from torchvision import transforms

from app.config import (
    MODEL_PATH,
    APP_NAME,
    APP_VERSION,
    MNIST_MEAN,
    MNIST_STD,
    IMAGE_SIZE
)
from app.model_def import load_model
from app.schemas import PredictionResponse, HealthResponse, ErrorResponse


# Initialize FastAPI app
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="MNIST digit classification inference service"
)

# Global model variable (loaded on startup)
model: Optional[torch.nn.Module] = None
model_loaded: bool = False

# Setup templates directory
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Setup static files directory (if it exists)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup_event():
    """
    Load the trained model on application startup.
    
    This ensures the model is loaded once and reused for all requests,
    improving performance and reducing memory usage.
    """
    global model, model_loaded
    
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(str(MODEL_PATH))
        model_loaded = True
        print(f"✓ {APP_NAME} v{APP_VERSION} started successfully")
        print(f"✓ Model ready for inference")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f"✗ Model not loaded - inference will not be available")
        model_loaded = False
    except Exception as e:
        print(f"✗ Unexpected error loading model: {e}")
        model_loaded = False


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for Kubernetes liveness and readiness probes.
    
    Returns:
        HealthResponse with status and model_loaded flag
    """
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded
    )


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Serve the main web UI for image upload and prediction.
    
    This endpoint will be implemented in Task 5.2.
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "app_name": APP_NAME, "version": APP_VERSION}
    )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess an uploaded image for MNIST inference.
    
    Steps:
    1. Convert to grayscale
    2. Resize to 28x28
    3. Convert to tensor
    4. Normalize using MNIST mean and std
    5. Add batch dimension
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed tensor of shape (1, 1, 28, 28)
    """
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),   # Resize to 28x28
        transforms.ToTensor(),                         # Convert to tensor [0, 1]
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))  # Normalize
    ])
    
    # Apply transforms
    tensor = transform(image)
    
    # Add batch dimension: (1, 28, 28) -> (1, 1, 28, 28)
    tensor = tensor.unsqueeze(0)
    
    return tensor


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Perform digit prediction on an uploaded image.
    
    This endpoint will be fully implemented in Task 5.3.
    
    Args:
        file: Uploaded image file (PNG, JPEG, GIF, etc.)
    
    Returns:
        PredictionResponse with predicted digit and probabilities
    
    Raises:
        HTTPException: If model not loaded or invalid image
    """
    # Check if model is loaded
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the training job has completed."
        )
    
    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Perform inference (no gradient calculation needed)
        with torch.no_grad():
            output = model(input_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert to Python types
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            probabilities_list = probabilities[0].tolist()
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities_list
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/info")
async def info():
    """
    Get application information.
    
    Returns:
        Application metadata including version and model status
    """
    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH)
    }


if __name__ == "__main__":
    import uvicorn
    from app.config import APP_HOST, APP_PORT
    
    uvicorn.run(
        "app.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True
    )
