# app/routers/watermark.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import numpy as np
import cv2
from pathlib import Path

from app.filters.watermark import aplicar_watermark_cuda

router = APIRouter()

# Ruta al logo
# Asumimos que este archivo está en app/routers/watermark.py
# y el logo en app/static/...
BASE_DIR = Path(__file__).resolve().parent.parent
LOGO_PATH = BASE_DIR / "static" / "UPS.png"

@router.post("/api/watermark")
async def watermark_filter(
    file: UploadFile = File(...),
    scale: float = Form(0.3),
    transparency: float = Form(0.3),
    spacing: float = Form(0.5)
) -> Response:
    """
    Aplica el filtro de marca de agua (logo universitario) usando CUDA.
    
    Args:
        file: Imagen a procesar.
        scale: Tamaño del logo relativo al ancho de la imagen (0.0 - 1.0).
        scale: Tamaño del logo relativo al ancho de la imagen (0.0 - 1.0).
        transparency: Opacidad del logo (0.0 - 1.0).
        spacing: Espaciado entre logos (fracción del tamaño del logo).
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Validar parámetros
        if scale <= 0 or scale > 1.0:
            raise HTTPException(status_code=400, detail="Scale must be between 0.0 and 1.0")
        if transparency < 0 or transparency > 1.0:
            raise HTTPException(status_code=400, detail="Transparency must be between 0.0 and 1.0")
        if spacing < 0:
            raise HTTPException(status_code=400, detail="Spacing must be non-negative")

        # Aplicar filtro
        result = aplicar_watermark_cuda(image, str(LOGO_PATH), scale, transparency, spacing)

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
