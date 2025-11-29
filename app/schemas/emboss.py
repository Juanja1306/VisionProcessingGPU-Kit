from pydantic import BaseModel

class EmbossParameters(BaseModel):
    kernel_size: int = 3
    bias_value: int = 128