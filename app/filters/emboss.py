# Dependencies
from dataclasses import dataclass
import numpy as np
from fastapi import UploadFile
import time
from PIL import Image
from io import BytesIO
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from numpy.typing import NDArray


RGBImage = NDArray[np.uint8]
IntKernel = NDArray[np.int32]

@dataclass(frozen=True)
class EmbossParams:
    kernel    : IntKernel    
    height    : int
    width     : int
    channels  : int


@dataclass (frozen=True)
class CudaProps:    
    imageFile    : UploadFile
    kernelSize   : int
    biasValue    : int
    
class CudaEmboss():
    def __init__(self, options: CudaProps):
        self.__imageFile    = options.imageFile                
        self.__kernelSize   = options.kernelSize
        self.__biasValue    = options.biasValue
        
    async def aplyFilter(self):
        print("Applying emboss filter...")        
        image_bytes = await self.__imageFile.read()
        
        image = self.__readImage(image_bytes) # Asumiendo método renombrado                
        image_params = self.__getImageParamters(image)
        
        emboss_params = self.__selectKernel(image_params)        
        
        image_processed = self.__embossFilter(image, emboss_params)
        
        return image_processed        
        
    def __readImage(self, image_bytes: bytes) -> RGBImage:
        print("Reading image...")
        image_stream = BytesIO(image_bytes)
        
        with Image.open(image_stream) as pilImage:
            pilImage = pilImage.convert("RGB")
            imageArray: RGBImage = np.array(pilImage, dtype=np.uint8)
            
        return imageArray 
    
    
    def __getImageParamters(self, image: RGBImage):
        height, width, channels = image.shape       
        parameters = {
            'height'  : height,
            'width'   : width,
            'channels': channels
        }        
        
        print(f"Image parameters: Height={height}, Width={width}, Channels={channels}")        
        return parameters  
    
    def __embossFilter(self, image: RGBImage, embossParams: EmbossParams) -> RGBImage:
        print("Applying Emboss filter with CUDA...")

        # 1. Preparar Parámetros
        bias_value = self.__biasValue
        kernel = embossParams.kernel
        image_width = embossParams.width
        image_height = embossParams.height
        kernel_size = len(kernel) # Asumiendo kernel cuadrado


        
        # Medir el tiempo y la memoria de la GPU antes de comenzar
        start_time = time.time()  
        free_memory_before, total_memory_before = drv.mem_get_info()

        # 2. Transferir datos iniciales a la GPU (Host to Device - HtoD)
        image_on_gpu = np.array(image, dtype=np.uint8)
        kernel_on_gpu = np.array(kernel, dtype=np.int32)

        image_on_gpu_mem = drv.mem_alloc(image_on_gpu.nbytes)
        drv.memcpy_htod(image_on_gpu_mem, image_on_gpu)  

        kernel_on_gpu_mem = drv.mem_alloc(kernel_on_gpu.nbytes)
        drv.memcpy_htod(kernel_on_gpu_mem, kernel_on_gpu) 

        # Asignar memoria para la imagen procesada en la GPU
        embossed_image_on_gpu = np.zeros_like(image_on_gpu, dtype=np.uint8)
        embossed_image_on_gpu_mem = drv.mem_alloc(embossed_image_on_gpu.nbytes)

        # 3. Definir el kernel CUDA (El código C/C++)
        module = SourceModule("""
        __global__ void emboss_filter(unsigned char *image, int *kernel, unsigned char *output, int width, int height, int kernel_size, int bias) {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            int kHalf = kernel_size / 2;

            // Verificar límites para evitar errores de acceso (Excluye bordes sin padding)
            if (x >= kHalf && x < width - kHalf && y >= kHalf && y < height - kHalf) {
                int sum = 0;
                // Asume que la imagen NO tiene padding y que el kernel debe estar dentro de los límites.
                // Esta lógica de kernel CUDA es simplificada y espera una imagen sin padding
                // y asume un layout de memoria plano (solo para fines de demostración de CUDA).
                
                // NOTA: Tu código CUDA original solo funcionaba para imágenes en escala de grises
                // o si la imagen se procesaba plano. Estamos copiando la lógica original:
                
                // Iteración sobre los elementos del kernel
                for (int ky = -kHalf; ky <= kHalf; ky++) {
                    for (int kx = -kHalf; kx <= kHalf; kx++) {
                        // Acceso simple (asumiendo procesamiento por canal o imagen plana)
                        // Para el color, se requiere un bucle adicional o un diseño de memoria diferente.
                        int pixel_index = (y + ky) * width + (x + kx);
                        
                        // Verificar límites de píxeles
                        if (pixel_index >= 0 && pixel_index < width * height) {
                            int pixel_value = image[pixel_index];
                            int kernel_index = (ky + kHalf) * kernel_size + (kx + kHalf);
                            int kernel_value = kernel[kernel_index];
                            sum += pixel_value * kernel_value;
                        }
                    }
                }
                
                int result = sum + bias;
                // Clamp
                result = (result < 0) ? 0 : (result > 255) ? 255 : result;
                output[y * width + x] = (unsigned char)result;
            }
        }
        """)

        # 4. Configurar y Lanzar el kernel CUDA
        block_size = (16, 16, 1)
        # Calcular el tamaño de la cuadrícula
        grid_size = (int(image_width / block_size[0]) + 1, int(image_height / block_size[1]) + 1)
        emboss_filter_function = module.get_function("emboss_filter")

        # Ejecutar el kernel
        emboss_filter_function(image_on_gpu_mem, kernel_on_gpu_mem, embossed_image_on_gpu_mem, 
                               np.int32(image_width), np.int32(image_height), np.int32(kernel_size), np.int32(bias_value),
                               block=block_size, grid=grid_size)

        # 5. Copiar el resultado de vuelta a la memoria de la CPU (Device to Host - DtoH)
        drv.memcpy_dtoh(embossed_image_on_gpu, embossed_image_on_gpu_mem)

        # Medir el tiempo y la memoria después de terminar
        end_time = time.time()
        free_memory_after, total_memory_after = drv.mem_get_info()

        elapsed_time = end_time - start_time
        memory_used = total_memory_before - free_memory_after
        
        print(f"Execution time (CUDA): {elapsed_time:.4f} seconds")
        print(f"Memory used (CUDA): {memory_used / 1024 / 1024:.4f} MB")
        
        # 6. Devolver la imagen procesada
        return embossed_image_on_gpu
    
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