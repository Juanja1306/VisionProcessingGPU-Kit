# app/schemas/collage.py
from pydantic import BaseModel

class CollageParameters(BaseModel):
    """
    Parámetros para el filtro Collage (6 divisiones diagonales).
    No requiere parámetros complejos por ahora, pero dejamos la estructura lista.
    """
    pass
