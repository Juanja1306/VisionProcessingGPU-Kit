from fastapi import APIRouter, status, Depends
import base64
from app.schemas.canny import CannyParams, CannyResponse

router = APIRouter()


@router.post("/", status_code=status.HTTP_200_OK, response_model=CannyResponse, tags=["Canny"])
async def process_canny(params: CannyParams = Depends(CannyParams.as_dependency())):
    """
    Process an image with Canny edge detection filter.
    Accepts an image file upload and optional threshold parameters.
    Returns the processed image as base64 encoded string.
    """
    # Read the uploaded file
    contents = await params.image.read()
    
    # TODO: Process the image with Canny filter using params
    # For now, return the original image as base64 (replace with processed image)
    processed_image_base64 = base64.b64encode(contents).decode('utf-8')
    
    response = CannyResponse(
        message="Canny processing completed",
        filename=params.image.filename,
        content_type=params.image.content_type,
        file_size=len(contents),
        processed_image=processed_image_base64
    )
    
    return response