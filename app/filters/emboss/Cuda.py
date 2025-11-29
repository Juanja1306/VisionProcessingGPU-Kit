# Dependencies
from dataclasses import dataclass
import numpy as np
from fastapi import UploadFile
import time

# Modules
from .abstractEmboss import AplicationFilter
from .types.index import IntKernel
from .utils.processImage import ProcessImage
from .types.index import IntKernel, RGBImage

@dataclass(frozen=True)
class EmbossParams:
    kernel    : IntKernel    
    height    : int
    width     : int
    channels  : int


@dataclass (frozen=True)
class CudaProps:
    processImage : ProcessImage    
    imageFile    : UploadFile
    kernelSize   : int
    biasValue    : int
    
class CudaEmboss(AplicationFilter):
    def __init__(self, options: CudaProps):
        self.__imageFile    = options.imageFile        
        self.__processImage = options.processImage
        self.__kernelSize   = options.kernelSize
        self.__biasValue    = options.biasValue
        
    async def aplyFilter(self):
        print("Applying emboss filter...")        
        image_bytes = await self.__imageFile.read()
        
        image = self.__processImage.readImage(image_bytes) # Asumiendo método renombrado                
        image_params = self.__processImage.getImageParamters(image)
        
        emboss_params = self.__selectKernel(image_params)        
        
        image_processed = self.__embossFilter(image, emboss_params)
        
        return image_processed        
        
        
    
    def __embossFilter(self, image: RGBImage, embossParams: EmbossParams) -> RGBImage:
        print("Starting convolution process...")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must have shape (height, width, 3) in RGB")

        bias = self.__biasValue
        kernel = embossParams.kernel
        ksize = self.__kernelSize
        
        imageChannels = embossParams.channels
        imageHeight = embossParams.height
        imageWidth = embossParams.width

        # Padding (Asumiendo que el kernel es siempre cuadrado e impar)
        kHalf = ksize // 2
        paddedImage: RGBImage = np.pad(
            image, pad_width=((kHalf, kHalf), (kHalf, kHalf), (0, 0)), mode="edge"
        )

        embossedImage: RGBImage = np.zeros_like(image, dtype=np.uint8)

        # Medición de rendimiento (opcional pero bueno)
        start_time = time.time() 
        
        # Iteración sobre la imagen original
        for channelIndex in range(imageChannels):
            for rowIndex in range(imageHeight):
                for colIndex in range(imageWidth):
                    convolutionSum = 0
                    
                    # Convolución manual
                    for i in range(ksize):  # Kernel row index
                        for j in range(ksize):  # Kernel col index
                            
                            # Acceder al píxel en la imagen con padding
                            pixel_value = paddedImage[rowIndex + i, colIndex + j, channelIndex]
                            kernel_value = kernel[i, j]
                            convolutionSum += pixel_value * kernel_value

                    # Aplicar bias y clamp
                    valueWithBias = convolutionSum + bias
                    clampedValue = np.uint8(max(0, min(255, valueWithBias)))
                    
                    embossedImage[rowIndex, colIndex, channelIndex] = clampedValue
        
        end_time = time.time()
        print(f"Convolution time: {end_time - start_time:.4f} seconds")
            
        return embossedImage
    
    def __selectKernel(self, imageParams: dict[str, int]) -> EmbossParams:
        print("Selecting/Creating Kernel...")
        
        height = imageParams['height']
        width = imageParams['width']
        chanels = imageParams['channels']
        kernel_size = self.__kernelSize
        

        if kernel_size == 3:
            kernel = np.array(
                [[-2, -1, 0],
                 [-1,  1, 1],
                 [ 0,  1, 2]],
                dtype=np.int32
            )
        # Si el tamaño inyectado no es 3, podríamos usar una versión predefinida o lanzar un error
        elif kernel_size == 5:
             kernel = np.array(
                [
                    [ -2, -1,  0,  1,   2 ],
                    [ -1,  1,  1,  1,   1 ],
                    [  0,  1,  2,  1,   0 ],
                    [  1,  1,  1,  1,  -1 ],
                    [  2,  1,  0, -1,  -2 ]
                ],
                dtype=np.int32
            )
        # Agrega más casos para otros tamaños de kernel si es necesario (7, 9, etc.)
        elif kernel_size == 7:
            kernel = np.array(
                [
                    [ -2, -1,  0, 1, 2, 3, 4  ],
                    [ -1,  0,  1, 2, 3, 4, 5  ],
                    [  0,  1,  2, 3, 4, 5, 6  ],
                    [  1,  2,  3, 4, 5, 6, 7  ],
                    [  2,  3,  4, 5, 6, 7, 8  ],
                    [  3,  4,  5, 6, 7, 8, 9  ],
                    [  4,  5,  6, 7, 8, 9, 10 ]
                ],
                dtype=np.int32
            )
        elif kernel_size == 9:
            kernel = np.array(
                [
                    [ -3, -2, -1, 0, 1, 2,  3,  4,  5  ],
                    [ -2, -1,  0, 1, 2, 3,  4,  5,  6  ],
                    [ -1,  0,  1, 2, 3, 4,  5,  6,  7  ],
                    [  0,  1,  2, 3, 4, 5,  6,  7,  8  ],
                    [  1,  2,  3, 4, 5, 6,  7,  8,  9  ],
                    [  2,  3,  4, 5, 6, 7,  8,  9,  10 ],
                    [  3,  4,  5, 6, 7, 8,  9,  10, 11 ],
                    [  4,  5,  6, 7, 8, 9,  10, 11, 12 ],
                    [  5,  6,  7, 8, 9, 10, 11, 12, 13 ]
                ],
                dtype=np.int32
            )
        else:
            raise ValueError(f"Kernel size {kernel_size} not supported for Emboss filter.")


        embossParams = EmbossParams(
            kernel=kernel,
            height=height,
            width=width,
            channels=chanels,
        )
        
        print(f'Kernel size used: {kernel_size}, Bias: {self.__biasValue}')
        return embossParams