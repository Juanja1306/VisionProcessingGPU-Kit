from pydantic import BaseModel, Field
from fastapi import UploadFile, File, Form, Depends
from typing import Optional


class CannyParams:
    """
    Parameters for Canny edge detection filter.
    """
    def __init__(self, image: UploadFile, low_threshold: Optional[float] = None, high_threshold: Optional[float] = None):
        self.image = image
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    @classmethod
    def as_dependency(cls):
        """
        Returns a dependency function for FastAPI to create CannyParams from form data.
        Usage: params: CannyParams = Depends(CannyParams.as_dependency())
        """
        async def get_canny_params(
            image: UploadFile = File(..., description="Image file to process"),
            low_threshold: Optional[float] = Form(None, description="Lower threshold for edge detection"),
            high_threshold: Optional[float] = Form(None, description="Upper threshold for edge detection")
        ) -> 'CannyParams':
            """
            Dependency function to create CannyParams from form data.
            """
            return cls(
                image=image,
                low_threshold=low_threshold,
                high_threshold=high_threshold
            )
        return get_canny_params


class CannyResponse(BaseModel):
    """
    Response schema for Canny edge detection processing.
    """
    message: str = Field(..., description="Processing status message")
    filename: Optional[str] = Field(None, description="Name of the processed image file")
    content_type: Optional[str] = Field(None, description="MIME type of the image")
    file_size: Optional[int] = Field(None, description="Size of the processed file in bytes")
    processed_image: str = Field(..., description="Processed image as base64 encoded string")

