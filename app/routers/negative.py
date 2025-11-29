# app/routers/negative.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
import numpy as np
import cv2

from app.filters.negative import aplicar_negative_cuda

router = APIRouter()


@router.post("/api/negative")
async def negative_filter(
    file: UploadFile = File(...),
) -> Response:
    """
    Aplica el filtro negativo usando CUDA sobre la imagen subida.

    No recibe parámetros adicionales: simplemente invierte los canales de color.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # image: BGR uint8, 3 canales → es perfecto para el filtro negativo
        result = aplicar_negative_cuda(image)

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
