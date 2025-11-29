# app/schemas/negative.py
from pydantic import BaseModel


class NegativeParameters(BaseModel):
    """
    Parámetros del filtro negativo.

    Por ahora el filtro negativo no necesita opciones adicionales, pero
    se define este esquema para mantener consistencia con el resto de la
    arquitectura (Canny, Gaussian, etc.) y dejar un punto de extensión
    para el futuro.

    Ejemplos de campos que podríamos añadir más adelante:
    - apply_red: bool = True
    - apply_green: bool = True
    - apply_blue: bool = True
    - intensity: float = 1.0
    """
    pass
