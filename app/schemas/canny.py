from pydantic import BaseModel
from typing import Optional

class CannyParameters(BaseModel):
    kernel_size: int = 5
    sigma: float = 1.4
    low_threshold: Optional[float] = None
    high_threshold: Optional[float] = None
