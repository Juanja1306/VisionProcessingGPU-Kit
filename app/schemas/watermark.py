# app/schemas/watermark.py
from pydantic import BaseModel, Field

class WatermarkParameters(BaseModel):
    """
    Parámetros para el filtro de marca de agua.
    """
    scale: float = Field(0.2, ge=0.01, le=1.0, description="Escala del logo respecto al tamaño de la imagen (0.1 = 10%)")
    transparency: float = Field(0.3, ge=0.0, le=1.0, description="Nivel de transparencia del logo (0.0 = invisible, 1.0 = totalmente opaco)")
    spacing: float = Field(0.5, ge=0.0, description="Espaciado entre logos como fracción del tamaño del logo (0.5 = 50% de espacio)")
