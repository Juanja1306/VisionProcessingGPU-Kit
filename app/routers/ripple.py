# app/routers/ripple.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import numpy as np
import cv2

from app.filters.ripple import aplicar_ripple_cuda

router = APIRouter()

@router.post("/api/ripple")
async def ripple_filter(
    file: UploadFile = File(...),
    edge_threshold: float = Form(100.0),
    color_levels: int = Form(8),
    saturation: float = Form(1.2)
) -> Response:
    """
    Aplica un efecto de "Comic Book" (Caricatura).
    Combina detección de bordes (trazado negro) con posterización de color.
    
    Args:
        file: Imagen a procesar.
        edge_threshold: Sensibilidad de los bordes (0-255).
        color_levels: Número de niveles de color (posterización).
        saturation: Saturación de color (1.0 = normal).
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Aplicar filtro
        result = aplicar_ripple_cuda(image, edge_threshold, color_levels, saturation)

        ok, encoded_img = cv2.imencode(".png", result)
        if not ok:
            raise HTTPException(status_code=500, detail="No se pudo codificar la imagen de salida")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
