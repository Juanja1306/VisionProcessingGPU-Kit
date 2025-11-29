# app/filters/gaussian.py
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

import pycuda.autoinit  # Inicializa el contexto CUDA
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from app.filters.canny import generar_kernel_gaussiano


CUDA_KERNEL_GAUSSIAN = """
__global__ void convolucion_gaussiana(float *imagen_in, float *imagen_out, float *kernel, 
                                      int altura, int ancho, int tam_kernel, int offset) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < ancho && y < altura) {
        float suma = 0.0f;

        for (int ky = 0; ky < tam_kernel; ++ky) {
            for (int kx = 0; kx < tam_kernel; ++kx) {
                int py = y + ky - offset;
                int px = x + kx - offset;

                if (py < 0) py = 0;
                if (py >= altura) py = altura - 1;
                if (px < 0) px = 0;
                if (px >= ancho) px = ancho - 1;

                suma += imagen_in[py * ancho + px] * kernel[ky * tam_kernel + kx];
            }
        }

        // Clamp [0,255]
        float resultado = fmaxf(fminf(suma, 255.0f), 0.0f);
        imagen_out[y * ancho + x] = resultado;
    }
}
""";

# Compilamos el kernel una sola vez
_mod_gaussian = SourceModule(CUDA_KERNEL_GAUSSIAN)
_convolucion_gaussiana = _mod_gaussian.get_function("convolucion_gaussiana")


def _seleccionar_parametros_gaussianos(
    altura: int,
    ancho: int,
    kernel_size: Optional[int],
    sigma: Optional[float],
) -> Tuple[int, float]:
    """
    Si kernel_size o sigma vienen como None o <= 0, se calculan valores
    recomendados en función del tamaño de la imagen.

    Versión agresiva: usa kernels grandes y sigma alto para que el desenfoque
    sea MUY evidente incluso en imágenes de resolución muy alta.
    """
    corto = min(altura, ancho)

    # Elegir kernel MUY grande según tamaño
    if kernel_size is None or kernel_size <= 0:
        if corto <= 1080:
            # HD o menor
            kernel_size = 15
        elif corto <= 2160:
            # ~FullHD / 2K
            kernel_size = 31
        elif corto <= 4320:
            # ~4K
            kernel_size = 41
        else:
            # Imágenes enormes (8K+, panorámicas tipo 10000x4000)
            kernel_size = 51

    # Aseguramos impar
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Sigma muy alto para que el efecto sea fuerte
    # (usar sigma ≈ tamaño del kernel da un blur bastante extremo)
    if sigma is None or sigma <= 0.0:
        sigma = float(kernel_size)

    return kernel_size, sigma

def aplicar_convolucion_cuda(
    imagen_grises: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Aplica una convolución 2D usando CUDA.

    Args:
        imagen_grises: Imagen 2D en escala de grises (alto x ancho).
        kernel: Kernel cuadrado (N x N) en float32.

    Returns:
        Imagen filtrada como np.uint8 (alto x ancho).
    """
    if imagen_grises.ndim != 2:
        raise ValueError("aplicar_convolucion_cuda espera una imagen 2D en escala de grises")

    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("El kernel debe ser una matriz cuadrada (N x N)")

    altura, ancho = imagen_grises.shape
    tam_kernel = kernel.shape[0]
    offset = tam_kernel // 2

    imagen_float = imagen_grises.astype(np.float32, copy=False)
    kernel_float = kernel.astype(np.float32, copy=False)

    # Pasar datos a GPU
    imagen_gpu = gpuarray.to_gpu(imagen_float)
    kernel_gpu = gpuarray.to_gpu(kernel_float.ravel())
    resultado_gpu = gpuarray.empty_like(imagen_gpu)

    block_size: Tuple[int, int, int] = (16, 16, 1)
    grid_size: Tuple[int, int, int] = (
        (ancho + block_size[0] - 1) // block_size[0],
        (altura + block_size[1] - 1) // block_size[1],
        1,
    )

    _convolucion_gaussiana(
        imagen_gpu,
        resultado_gpu,
        kernel_gpu,
        np.int32(altura),
        np.int32(ancho),
        np.int32(tam_kernel),
        np.int32(offset),
        block=block_size,
        grid=grid_size,
    )

    resultado = resultado_gpu.get()
    resultado = np.clip(resultado, 0, 255).astype(np.uint8)
    return resultado


def aplicar_gaussian_cuda(
    imagen_grises: np.ndarray,
    kernel_size: Optional[int] = None,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Aplica un filtro gaussiano usando CUDA sobre una imagen en escala de grises.

    Si kernel_size o sigma son None o <=0, se usan valores recomendados
    (dependientes del tamaño de la imagen).
    """
    altura, ancho = imagen_grises.shape

    ks, sg = _seleccionar_parametros_gaussianos(
        altura=altura,
        ancho=ancho,
        kernel_size=kernel_size,
        sigma=sigma,
    )

    kernel_gauss = generar_kernel_gaussiano(ks, sg).astype(np.float32)

    return aplicar_convolucion_cuda(imagen_grises, kernel_gauss)
