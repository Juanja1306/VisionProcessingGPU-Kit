# app/schemas/gaussian.py
from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


class GaussianParameters(BaseModel):
    """
    Parámetros para el filtro Gaussiano.

    Nota:
        - Si kernel_size <= 0, el backend calculará un tamaño recomendado.
        - Si sigma <= 0, el backend calculará un sigma recomendado.
    """
    kernel_size: int = 5
    sigma: float = 1.4
