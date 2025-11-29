# Dependencies
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from PIL import Image
import numpy as np
from io import BytesIO

# Modules
from ..types.index import RGBImage


FormatImage = Literal["PNG", "JPG"]

@dataclass(frozen=True)
class SaveImageProps:
    pathImage   : str
    image       : RGBImage
    formatImage : FormatImage


class ProcessImage():
    
    def readImage(self, image_bytes: bytes) -> RGBImage:
        print("Reading image...")
        image_stream = BytesIO(image_bytes)
        
        with Image.open(image_stream) as pilImage:
            pilImage = pilImage.convert("RGB")
            imageArray: RGBImage = np.array(pilImage, dtype=np.uint8)
            
        return imageArray
    
    
    def getImageParamters(self, image: RGBImage):
        height, width, channels = image.shape       
        parameters = {
            'height'  : height,
            'width'   : width,
            'channels': channels
        }        
        
        print(f"Image parameters: Height={height}, Width={width}, Channels={channels}")        
        return parameters
    
