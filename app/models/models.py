from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union
from enum import Enum
from datetime import datetime
import uuid


class Gender(str, Enum):
    """Enum for gender options."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class MeasurementUnit(str, Enum):
    """Enum for measurement units."""
    CM = "cm"
    INCHES = "inches"


class BodyMeasurements(BaseModel):
    """Schema for body measurements extracted from images/video."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    height: float
    weight: Optional[float] = None
    chest: float
    waist: float
    hips: float
    inseam: Optional[float] = None
    shoulder_width: Optional[float] = None
    arm_length: Optional[float] = None
    unit: MeasurementUnit = MeasurementUnit.CM
    confidence_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "height": 175.0,
                "weight": 70.0,
                "chest": 95.0,
                "waist": 80.0,
                "hips": 98.0,
                "inseam": 82.0,
                "shoulder_width": 42.0,
                "arm_length": 65.0,
                "unit": "cm",
                "confidence_score": 0.85
            }
        }


class UserProfile(BaseModel):
    """Schema for user profile information."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    shop_id: str
    gender: Optional[Gender] = None
    age: Optional[int] = None
    measurements: Optional[BodyMeasurements] = None
    fit_preference: Optional[str] = None  # e.g., "loose", "regular", "tight"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class SizeChartEntry(BaseModel):
    """Schema for a single entry in a size chart."""
    size_name: str  # e.g., "S", "M", "L", "XL", "32", "34", etc.
    measurements: Dict[str, Union[float, List[float]]]  # e.g., {"chest": 95.0, "waist": [75.0, 80.0]}


class ProductSizeChart(BaseModel):
    """Schema for product size charts."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    product_id: str
    gender: Optional[Gender] = None
    unit: MeasurementUnit = MeasurementUnit.CM
    size_entries: List[SizeChartEntry]
    measurement_points: List[str]  # List of measurement points used in this chart


class SizeRecommendation(BaseModel):
    """Schema for size recommendations."""
    user_id: str
    product_id: str
    recommended_size: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    fit_analysis: Dict[str, Dict[str, float]]  # e.g., {"chest": {"difference": 2.0, "fit": "good"}}
    alternative_sizes: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)


class MeasurementRequest(BaseModel):
    """Schema for measurement extraction requests."""
    image_urls: Optional[List[str]] = None
    video_url: Optional[str] = None
    height: float
    gender: Optional[Gender] = None
    unit: MeasurementUnit = MeasurementUnit.CM
    reference_object: Optional[Dict[str, Union[str, float]]] = None  # e.g., {"type": "credit_card", "width": 8.56}


class SizeRequest(BaseModel):
    """Schema for size recommendation requests."""
    user_id: Optional[str] = None
    measurements: BodyMeasurements
    product_id: str
    fit_preference: Optional[str] = None  # e.g., "loose", "regular", "tight"