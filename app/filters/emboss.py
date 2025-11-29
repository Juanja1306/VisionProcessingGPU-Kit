import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

# Kernel CUDA para Emboss (Soporte RGB)
kernel_emboss_code = """
__global__ void emboss_filter(unsigned char *imagen, int *kernel, unsigned char *resultado, 
                              int ancho, int altura, int kernel_size, int bias) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    int kHalf = kernel_size / 2;
    int channels = 3; // Asumimos RGB
    
    // Verificar límites (evitar bordes)
    if (x < kHalf || x >= ancho - kHalf || y < kHalf || y >= altura - kHalf) {
        // Copiar pixel original en bordes
        int idx = (y * ancho + x) * channels;
        resultado[idx] = imagen[idx];
        resultado[idx + 1] = imagen[idx + 1];
        resultado[idx + 2] = imagen[idx + 2];
        return;
    }
    
    // Procesar cada canal
    for (int c = 0; c < channels; c++) {
        int sum = 0;
        
        for (int ky = -kHalf; ky <= kHalf; ky++) {
            for (int kx = -kHalf; kx <= kHalf; kx++) {
                int py = y + ky;
                int px = x + kx;
                
                int pixel_idx = (py * ancho + px) * channels + c;
                int kernel_idx = (ky + kHalf) * kernel_size + (kx + kHalf);
                
                sum += imagen[pixel_idx] * kernel[kernel_idx];
            }
        }
        
        int val = sum + bias;
        
        // Clamp
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        
        resultado[(y * ancho + x) * channels + c] = (unsigned char)val;
    }
}
"""

def obtener_kernel_numpy(kernel_size):
    if kernel_size == 3:
        return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.int32)
    elif kernel_size == 5:
        return np.array([
            [-2, -1, 0, 1, 2],
            [-1, 1, 1, 1, 1],
            [0, 1, 2, 1, 0],
            [1, 1, 1, 1, -1],
            [2, 1, 0, -1, -2]
        ], dtype=np.int32)
    elif kernel_size == 7:
        return np.array([
            [-2, -1, 0, 1, 2, 3, 4],
            [-1, 0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7, 8],
            [3, 4, 5, 6, 7, 8, 9],
            [4, 5, 6, 7, 8, 9, 10]
        ], dtype=np.int32)
    else:
        # Fallback a 3x3 si no está definido
        return np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.int32)

def aplicar_emboss_cuda(imagen, kernel_size=3, bias=128):
    """
    Aplica el filtro Emboss usando CUDA.
    
    Args:
        imagen: Imagen RGB (numpy array uint8)
        kernel_size: Tamaño del kernel (3, 5, 7, etc.)
        bias: Valor de bias a sumar
        
    Returns:
        numpy.ndarray: Imagen procesada
    """
    altura, ancho, canales = imagen.shape
    
    if canales != 3:
        # Si no es RGB (ej. RGBA), convertir o lanzar error. 
        # Por simplicidad asumimos que el router pasa RGB.
        pass

    # Compilar kernel
    mod = SourceModule(kernel_emboss_code)
    emboss_func = mod.get_function("emboss_filter")
    
    # Obtener kernel de convolución
    kernel_host = obtener_kernel_numpy(kernel_size)
    
    # Configurar bloques y grids
    block_size = (16, 16, 1)
    grid_size = (
        (ancho + block_size[0] - 1) // block_size[0],
        (altura + block_size[1] - 1) // block_size[1],
        1
    )
    
    # Transferir a GPU
    # Usamos flatten() para enviar como array 1D contiguo de bytes
    imagen_gpu = gpuarray.to_gpu(imagen.flatten().astype(np.uint8))
    kernel_gpu = gpuarray.to_gpu(kernel_host.flatten().astype(np.int32))
    resultado_gpu = gpuarray.empty_like(imagen_gpu)
    
    # Ejecutar kernel
    emboss_func(
        imagen_gpu, kernel_gpu, resultado_gpu,
        np.int32(ancho), np.int32(altura), np.int32(kernel_size), np.int32(bias),
        block=block_size, grid=grid_size
    )
    
    # Recuperar resultado
    resultado_flat = resultado_gpu.get()
    resultado = resultado_flat.reshape((altura, ancho, canales))
    
    return resultado