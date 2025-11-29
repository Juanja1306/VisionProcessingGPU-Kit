import numpy as np
import cv2
import os
import time

# Environment variables are now configured in app/core/cuda_config.py and initialized in app/main.py

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


def exp_manual(x):
    """Calcula e^x manualmente usando la serie de Taylor."""
    resultado = 1.0
    termino = 1.0
    
    for n in range(1, 50):
        termino *= x / n
        resultado += termino
        if abs(termino) < 1e-15:
            break
    
    return resultado


def generar_kernel_gaussiano(tamanio, sigma):
    """Genera un kernel gaussiano manualmente."""
    kernel = np.zeros((tamanio, tamanio), dtype=np.float32)
    centro = tamanio // 2
    
    suma_total = 0.0
    for y in range(tamanio):
        for x in range(tamanio):
            dx = x - centro
            dy = y - centro
            valor = exp_manual(-(dx*dx + dy*dy) / (2.0 * sigma * sigma))
            kernel[y, x] = valor
            suma_total += valor
    
    for y in range(tamanio):
        for x in range(tamanio):
            kernel[y, x] /= suma_total
    
    return kernel


def convertir_a_grises(imagen):
    """Convierte imagen a escala de grises."""
    if len(imagen.shape) == 3:
        b, g, r = cv2.split(imagen)
        grises = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    else:
        grises = imagen.astype(np.float32)
    return grises


# ==================== KERNELS CUDA ====================

# Kernel para suavizado gaussiano
kernel_suavizado = """
__global__ void convolve_gaussian(float *imagen, float *kernel, float *resultado, 
                                   int altura, int ancho, int tam_kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    int offset = tam_kernel / 2;
    float suma = 0.0f;
    
    for (int ky = 0; ky < tam_kernel; ky++) {
        for (int kx = 0; kx < tam_kernel; kx++) {
            int py = y + ky - offset;
            int px = x + kx - offset;
            
            // Manejar bordes
            if (py < 0) py = 0;
            if (py >= altura) py = altura - 1;
            if (px < 0) px = 0;
            if (px >= ancho) px = ancho - 1;
            
            suma += imagen[py * ancho + px] * kernel[ky * tam_kernel + kx];
        }
    }
    
    // Clamp entre 0 y 255
    if (suma < 0.0f) suma = 0.0f;
    if (suma > 255.0f) suma = 255.0f;
    
    resultado[y * ancho + x] = suma;
}
"""

# Kernel para cálculo de gradientes con Sobel
kernel_gradientes = """
__global__ void calcular_gradientes(float *imagen, float *magnitud, float *direccion,
                                      int altura, int ancho) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    // Bordes = 0
    if (x == 0 || x == ancho - 1 || y == 0 || y == altura - 1) {
        magnitud[y * ancho + x] = 0.0f;
        direccion[y * ancho + x] = 0.0f;
        return;
    }
    
    // Operadores Sobel
    float gx = 0.0f, gy = 0.0f;
    
    // Sobel X
    gx += -1.0f * imagen[(y-1) * ancho + (x-1)];
    gx +=  0.0f * imagen[(y-1) * ancho + x];
    gx +=  1.0f * imagen[(y-1) * ancho + (x+1)];
    gx += -2.0f * imagen[y * ancho + (x-1)];
    gx +=  0.0f * imagen[y * ancho + x];
    gx +=  2.0f * imagen[y * ancho + (x+1)];
    gx += -1.0f * imagen[(y+1) * ancho + (x-1)];
    gx +=  0.0f * imagen[(y+1) * ancho + x];
    gx +=  1.0f * imagen[(y+1) * ancho + (x+1)];
    
    // Sobel Y
    gy += -1.0f * imagen[(y-1) * ancho + (x-1)];
    gy += -2.0f * imagen[(y-1) * ancho + x];
    gy += -1.0f * imagen[(y-1) * ancho + (x+1)];
    gy +=  0.0f * imagen[y * ancho + (x-1)];
    gy +=  0.0f * imagen[y * ancho + x];
    gy +=  0.0f * imagen[y * ancho + (x+1)];
    gy +=  1.0f * imagen[(y+1) * ancho + (x-1)];
    gy +=  2.0f * imagen[(y+1) * ancho + x];
    gy +=  1.0f * imagen[(y+1) * ancho + (x+1)];
    
    magnitud[y * ancho + x] = sqrtf(gx * gx + gy * gy);
    direccion[y * ancho + x] = atan2f(gy, gx);
}
"""

# Kernel para supresión no-máxima
kernel_supresion = r"""
#define PI 3.14159265359f

__global__ void supresion_no_maxima(float *magnitud, float *direccion, float *resultado,
                                     int altura, int ancho) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    // Bordes = 0
    if (x == 0 || x == ancho - 1 || y == 0 || y == altura - 1) {
        resultado[y * ancho + x] = 0.0f;
        return;
    }
    
    float dir = direccion[y * ancho + x];
    float mag = magnitud[y * ancho + x];
    
    // Convertir a grados
    float angulo = dir * 180.0f / PI;
    if (angulo < 0.0f) angulo += 180.0f;
    
    float mag1, mag2;
    
    // Determinar dirección y vecinos
    if ((angulo >= 0.0f && angulo < 22.5f) || (angulo >= 157.5f && angulo <= 180.0f)) {
        // 0 grados (horizontal)
        mag1 = magnitud[y * ancho + (x-1)];
        mag2 = magnitud[y * ancho + (x+1)];
    } else if (angulo >= 22.5f && angulo < 67.5f) {
        // 45 grados (diagonal /)
        mag1 = magnitud[(y-1) * ancho + (x+1)];
        mag2 = magnitud[(y+1) * ancho + (x-1)];
    } else if (angulo >= 67.5f && angulo < 112.5f) {
        // 90 grados (vertical)
        mag1 = magnitud[(y-1) * ancho + x];
        mag2 = magnitud[(y+1) * ancho + x];
    } else {
        // 135 grados (diagonal backslash)
        mag1 = magnitud[(y-1) * ancho + (x-1)];
        mag2 = magnitud[(y+1) * ancho + (x+1)];
    }
    
    // Suprimir si no es máximo local
    if (mag >= mag1 && mag >= mag2) {
        resultado[y * ancho + x] = mag;
    } else {
        resultado[y * ancho + x] = 0.0f;
    }
}
"""

# Kernel para umbralización (primera fase)
kernel_umbralizacion = """
__global__ void umbralizar(float *magnitud, unsigned char *resultado,
                           float umbral_alto, float umbral_bajo,
                           int altura, int ancho) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    float mag = magnitud[y * ancho + x];
    
    if (mag >= umbral_alto) {
        resultado[y * ancho + x] = 255;  // Borde fuerte
    } else if (mag >= umbral_bajo) {
        resultado[y * ancho + x] = 128;  // Borde débil
    } else {
        resultado[y * ancho + x] = 0;    // No es borde
    }
}
"""

# Kernel para histéresis (conectar bordes)
kernel_histeresis = """
__global__ void conectar_bordes(unsigned char *imagen, unsigned char *resultado,
                                 int altura, int ancho, int *cambio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    // Bordes
    if (x == 0 || x == ancho - 1 || y == 0 || y == altura - 1) {
        resultado[y * ancho + x] = imagen[y * ancho + x];
        return;
    }
    
    unsigned char val = imagen[y * ancho + x];
    
    // Si es borde débil, verificar si está conectado a un borde fuerte
    if (val == 128) {
        bool conectado = false;
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (imagen[(y + dy) * ancho + (x + dx)] == 255) {
                    conectado = true;
                    break;
                }
            }
            if (conectado) break;
        }
        
        if (conectado) {
            resultado[y * ancho + x] = 255;
            *cambio = 1;  // Marcar que hubo cambio
        } else {
            resultado[y * ancho + x] = 128;
        }
    } else {
        resultado[y * ancho + x] = val;
    }
}
"""

# Kernel para limpieza final
kernel_limpieza = """
__global__ void limpiar_bordes_debiles(unsigned char *imagen, unsigned char *resultado,
                                        int altura, int ancho) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ancho || y >= altura) return;
    
    unsigned char val = imagen[y * ancho + x];
    
    if (val == 128) {
        resultado[y * ancho + x] = 0;  // Eliminar bordes débiles no conectados
    } else {
        resultado[y * ancho + x] = val;
    }
}
"""


def aplicar_canny_cuda(imagen_grises, tamanio_kernel=5, sigma=1.4, low_threshold=None, high_threshold=None):
    """
    Aplica el filtro Canny completo usando CUDA.
    
    Args:
        imagen_grises: Imagen en escala de grises (numpy array)
        tamanio_kernel: tamanio del kernel gaussiano (debe ser impar)
        sigma: Desviación estándar del filtro gaussiano
        low_threshold: Umbral bajo para histéresis (opcional)
        high_threshold: Umbral alto para histéresis (opcional)
    
    Returns:
        numpy.ndarray: Imagen con bordes detectados
    """
    altura, ancho = imagen_grises.shape
    
    # Compilar kernels
    mod_suavizado = SourceModule(kernel_suavizado)
    mod_gradientes = SourceModule(kernel_gradientes)
    mod_supresion = SourceModule(kernel_supresion)
    mod_umbralizacion = SourceModule(kernel_umbralizacion)
    mod_histeresis = SourceModule(kernel_histeresis)
    mod_limpieza = SourceModule(kernel_limpieza)
    
    convolve_func = mod_suavizado.get_function("convolve_gaussian")
    gradientes_func = mod_gradientes.get_function("calcular_gradientes")
    supresion_func = mod_supresion.get_function("supresion_no_maxima")
    umbralizar_func = mod_umbralizacion.get_function("umbralizar")
    conectar_func = mod_histeresis.get_function("conectar_bordes")
    limpiar_func = mod_limpieza.get_function("limpiar_bordes_debiles")
    
    # Configurar bloques y grids
    block_size = (16, 16, 1)
    grid_size = (
        (ancho + block_size[0] - 1) // block_size[0],
        (altura + block_size[1] - 1) // block_size[1],
        1
    )
    
    # print("  1) Generando kernel gaussiano...")
    kernel_gauss = generar_kernel_gaussiano(tamanio_kernel, sigma)
    tam_kernel = kernel_gauss.shape[0]
    
    # Transferir a GPU
    imagen_gpu = gpuarray.to_gpu(imagen_grises.astype(np.float32))
    kernel_gpu = gpuarray.to_gpu(kernel_gauss.flatten().astype(np.float32))
    resultado_gpu = gpuarray.empty((altura, ancho), dtype=np.float32)
    
    # Paso 1: Suavizado gaussiano
    # print("  2) Aplicando suavizado gaussiano (GPU)...")
    convolve_func(
        imagen_gpu, kernel_gpu, resultado_gpu,
        np.int32(altura), np.int32(ancho), np.int32(tam_kernel),
        block=block_size, grid=grid_size
    )
    
    suavizada_gpu = resultado_gpu
    
    # Paso 2: Cálculo de gradientes
    # print("  3) Calculando gradientes (Sobel) (GPU)...")
    magnitud_gpu = gpuarray.empty((altura, ancho), dtype=np.float32)
    direccion_gpu = gpuarray.empty((altura, ancho), dtype=np.float32)
    
    gradientes_func(
        suavizada_gpu, magnitud_gpu, direccion_gpu,
        np.int32(altura), np.int32(ancho),
        block=block_size, grid=grid_size
    )
    
    # Paso 3: Supresión no-máxima
    # print("  4) Aplicando supresión no-máxima (GPU)...")
    suprimida_gpu = gpuarray.empty((altura, ancho), dtype=np.float32)
    
    supresion_func(
        magnitud_gpu, direccion_gpu, suprimida_gpu,
        np.int32(altura), np.int32(ancho),
        block=block_size, grid=grid_size
    )
    
    # Paso 4: Calcular umbrales
    # print("  5) Calculando umbrales automáticos...")
    
    if low_threshold is None or high_threshold is None:
        suprimida_cpu = suprimida_gpu.get()
        
        # Calcular máximo manualmente
        max_magnitud = 0.0
        for y in range(altura):
            for x in range(ancho):
                if suprimida_cpu[y, x] > max_magnitud:
                    max_magnitud = suprimida_cpu[y, x]
        
        if high_threshold is None:
            high_threshold = max_magnitud * 0.15
        if low_threshold is None:
            low_threshold = high_threshold * 0.4
    
    # Paso 5: Umbralización
    # print("  6) Aplicando umbralización con histéresis (GPU)...")
    resultado_uint8_gpu = gpuarray.empty((altura, ancho), dtype=np.uint8)
    
    umbralizar_func(
        suprimida_gpu, resultado_uint8_gpu,
        np.float32(high_threshold), np.float32(low_threshold),
        np.int32(altura), np.int32(ancho),
        block=block_size, grid=grid_size
    )
    
    # Paso 6: Histéresis (conectar bordes) - iterativo
    # print("  7) Conectando bordes débiles a fuertes (GPU)...")
    temp_gpu = gpuarray.empty_like(resultado_uint8_gpu)
    cambio_gpu = gpuarray.zeros(1, dtype=np.int32)
    
    MAX_ITERACIONES = 100
    for iteracion in range(MAX_ITERACIONES):
        cambio_gpu.fill(0)
        
        conectar_func(
            resultado_uint8_gpu, temp_gpu,
            np.int32(altura), np.int32(ancho),
            cambio_gpu,
            block=block_size, grid=grid_size
        )
        
        # Copiar resultado temporal al principal
        resultado_uint8_gpu = temp_gpu.copy()
        
        # Verificar si hubo cambios
        if cambio_gpu.get()[0] == 0:
            break
    
    # Paso 7: Limpieza final
    # print("  8) Limpiando bordes débiles no conectados (GPU)...")
    resultado_final_gpu = gpuarray.empty_like(resultado_uint8_gpu)
    
    limpiar_func(
        resultado_uint8_gpu, resultado_final_gpu,
        np.int32(altura), np.int32(ancho),
        block=block_size, grid=grid_size
    )
    
    # Transferir resultado de vuelta a CPU
    resultado = resultado_final_gpu.get()
    
    return resultado
