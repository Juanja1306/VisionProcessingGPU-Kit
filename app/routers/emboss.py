from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import numpy as np
import cv2
from ..filters.emboss import aplicar_emboss_cuda

router = APIRouter()

@router.post("/api/emboss")
async def apply_emboss(
    file: UploadFile = File(...),
    kernel_size: int = Form(3),
    bias_value: int = Form(128),
    use_auto: bool = Form(False)
):
    """
    Aplica el filtro Emboss usando CUDA.
    
    - Si use_auto = True -> se ignoran kernel_size y bias_value y se usan valores
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
            
        # Convert BGR to RGB (OpenCV uses BGR, our filter expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if use_auto:
            # Cálculo automático de parámetros basado en el tamaño de la imagen
            h, w, _ = image_rgb.shape
            corto = min(h, w)
            
            # Calcular kernel_size según resolución
            if corto <= 1080:
                # HD o menor
                kernel_size = 31
            elif corto <= 2160:
                # ~FullHD / 2K
                kernel_size = 51
            elif corto <= 4320:
                # ~4K
                kernel_size = 71
            else:
                # Imágenes enormes (8K+)
                kernel_size = 91
            
            # Bias estándar
            bias_value = 100
        # Apply Emboss
        processed_image = aplicar_emboss_cuda(
            image_rgb,
            kernel_size=kernel_size,
            bias=bias_value
        )
        
        # Convert back to BGR for encoding
        processed_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        
        # Encode result
        success, encoded_image = cv2.imencode(".png", processed_bgr)
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode image")
            
        return Response(content=encoded_image.tobytes(), media_type="image/png")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
