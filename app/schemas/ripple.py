# app/schemas/ripple.py
from pydantic import BaseModel, Field

class RippleParameters(BaseModel):
    """
    Parámetros para el filtro creativo "Comic Book" (Cartoon).
    Combina detección de bordes y cuantización de color.
    """
    edge_threshold: float = Field(100.0, ge=0.0, le=255.0, description="Umbral para detectar bordes (menor = más líneas)")
    color_levels: int = Field(8, ge=2, le=32, description="Niveles de color para el efecto posterizado (menor = más caricaturesco)")
    saturation: float = Field(1.2, ge=0.0, le=3.0, description="Intensidad del color (1.0 = original)")
