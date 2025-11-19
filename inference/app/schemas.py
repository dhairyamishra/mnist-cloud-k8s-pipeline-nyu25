"""
Pydantic schemas for MNIST inference API.

Defines request and response models for type validation and API documentation.
"""
from typing import List
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    Response model for digit prediction.
    
    Contains the predicted digit class and probability distribution
    across all 10 digit classes (0-9).
    """
    predicted_class: int = Field(
        ...,
        ge=0,
        le=9,
        description="Predicted digit class (0-9)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted class (0.0-1.0)"
    )
    probabilities: List[float] = Field(
        ...,
        min_items=10,
        max_items=10,
        description="Probability distribution across all 10 digit classes"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": 7,
                "confidence": 0.9876,
                "probabilities": [
                    0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
                    0.0006, 0.0007, 0.9876, 0.0008, 0.0009
                ]
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Used by Kubernetes liveness and readiness probes.
    """
    status: str = Field(
        ...,
        description="Health status of the service"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded and ready for inference"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "model_loaded": True
            }
        }


class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    
    Provides structured error information to clients.
    """
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: str = Field(
        None,
        description="Additional error details (optional)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid image format",
                "detail": "Image must be in PNG, JPEG, or GIF format"
            }
        }
