# app/routers/collage.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
import numpy as np
import cv2

from app.filters.collage import aplicar_collage_cuda

router = APIRouter()

@router.post("/api/collage")
async def collage_filter(
    file: UploadFile = File(...)
) -> Response:
    """
    Aplica un efecto de Collage dividiendo la imagen en 6 franjas diagonales.
    Cada franja muestra un filtro diferente:
    1. Canny (Bordes)
    2. Emboss (Relieve)
    3. Gaussian (Desenfoque)
    4. Negative (Invertido)
    5. Watermark (Marca de agua)
    6. Comic (Caricatura)
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Aplicar filtro
        result = aplicar_collage_cuda(image)

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
