from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import numpy as np
import cv2
from app.filters.canny import aplicar_canny_cuda, convertir_a_grises

router = APIRouter()

@router.post("/api/canny")
async def canny_filter(
    file: UploadFile = File(...),
    kernel_size: int = Form(5),
    sigma: float = Form(1.4),
    low_threshold: str = Form(None), # Receive as string to handle "null" or empty
    high_threshold: str = Form(None),
    use_auto: bool = Form(False)
):
    """
    Aplica el filtro Canny Edge Detection usando CUDA.
    
    - Si use_auto = True -> se ignoran todos los parámetros y se usan valores
      recomendados según el tamaño de la imagen.
    - Si use_auto = False -> se usan los parámetros proporcionados.
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert to grayscale
        gray_image = convertir_a_grises(image)
        
        altura, ancho = gray_image.shape

        if use_auto:
            # Cálculo automático de parámetros basado en el tamaño de la imagen
            corto = min(altura, ancho)
            
            # Calcular kernel_size y sigma según resolución
            if corto <= 1080:
                # HD o menor
                kernel_size = 5
                sigma = 1.4
            elif corto <= 2160:
                # ~FullHD / 2K
                kernel_size = 7
                sigma = 2.0
            elif corto <= 4320:
                # ~4K
                kernel_size = 9
                sigma = 2.5
            else:
                # Imágenes enormes (8K+)
                kernel_size = 11
                sigma = 3.0
            
            # Los thresholds se calcularán automáticamente dentro de aplicar_canny_cuda
            low = None
            high = None
        else:
            # Parse thresholds manualmente
            low = float(low_threshold) if low_threshold and low_threshold != 'null' and low_threshold != '' else None
            high = float(high_threshold) if high_threshold and high_threshold != 'null' and high_threshold != '' else None

        # Apply Canny
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        result = aplicar_canny_cuda(
            gray_image, 
            tamanio_kernel=kernel_size, 
            sigma=sigma,
            low_threshold=low,
            high_threshold=high
        )

        # Encode result to image
        _, encoded_img = cv2.imencode('.png', result)
        
        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))