# app/routers/gaussian.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import numpy as np
import cv2

from app.filters.gaussian import aplicar_gaussian_cuda
from app.schemas.gaussian import GaussianParameters

router = APIRouter()


@router.post("/api/gaussian")
async def gaussian_filter(
    file: UploadFile = File(...),
    kernel_size: int = Form(5),
    sigma: float = Form(1.4),
    use_auto: bool = Form(False),
) -> Response:
    """
    Aplica un filtro gaussiano usando CUDA sobre una imagen A COLOR.

    - Si use_auto = True -> se ignoran kernel_size y sigma y se usan valores
      recomendados según el tamaño de la imagen.
    - Si use_auto = False -> se usan kernel_size y sigma. Si alguno es <= 0,
      se trata como "no especificado" y se calcula un valor recomendado.

    El filtro se aplica por canal (B, G, R) para conservar el color.
    """
    try:
        # Leer la imagen subida (color BGR)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Parámetros desde schema (solo para mantener consistencia/tipado)
        params = GaussianParameters(kernel_size=kernel_size, sigma=sigma)

        if use_auto:
            # Dejamos que el filtro elija kernel_size y sigma según tamaño de imagen
            ks = None
            sg = None
        else:
            # Manual: lo que venga de los sliders (si 0/negativo, se auto–ajusta internamente)
            ks = params.kernel_size
            sg = params.sigma

        # ====== APLICAR GAUSSIAN POR CANAL (COLOR) ======
        # image: H x W x 3 (B, G, R)
        b_channel, g_channel, r_channel = cv2.split(image)

        b_blur = aplicar_gaussian_cuda(b_channel, kernel_size=ks, sigma=sg)
        g_blur = aplicar_gaussian_cuda(g_channel, kernel_size=ks, sigma=sg)
        r_blur = aplicar_gaussian_cuda(r_channel, kernel_size=ks, sigma=sg)

        # Volvemos a juntar en una sola imagen color
        result_color = cv2.merge([b_blur, g_blur, r_blur])

        ok, encoded_img = cv2.imencode(".png", result_color)
        if not ok:
            raise HTTPException(status_code=500, detail="No se pudo codificar la imagen de salida")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
