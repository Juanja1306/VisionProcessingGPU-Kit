# app/filters/negative.py
from __future__ import annotations

from typing import Tuple

import numpy as np

import pycuda.autoinit  # Inicializa el contexto CUDA
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


CUDA_KERNEL_NEGATIVE = r"""
extern "C"
__global__ void negative(
    unsigned char *input,
    unsigned char *output,
    int width, int height
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    output[idx]     = 255 - input[idx];
    output[idx + 1] = 255 - input[idx + 1];
    output[idx + 2] = 255 - input[idx + 2];
}
""";

_mod_negative = SourceModule(CUDA_KERNEL_NEGATIVE)
_negative_kernel = _mod_negative.get_function("negative")


def aplicar_negative_cuda(imagen_color: np.ndarray) -> np.ndarray:
    """
    Aplica el filtro negativo usando CUDA sobre una imagen a color (BGR/RGB).

    Args:
        imagen_color: np.ndarray de forma (alto, ancho, 3), dtype=uint8.

    Returns:
        np.ndarray con el negativo de la imagen, misma forma y dtype.
    """
    if imagen_color.ndim != 3:
        raise ValueError("Se esperaba una imagen de 3 dimensiones (H, W, C)")

    altura, ancho, canales = imagen_color.shape

    if canales == 4:
        # Si viene con alfa (BGRA / RGBA), ignoramos el canal alfa
        imagen_color = imagen_color[:, :, :3]
        canales = 3

    if canales != 3:
        raise ValueError("El filtro negativo CUDA está definido para imágenes con 3 canales")

    # Aseguramos tipo uint8 sin copiar innecesariamente
    img_uint8 = imagen_color.astype(np.uint8, copy=False)

    # Subir a GPU
    d_input = gpuarray.to_gpu(img_uint8)
    d_output = gpuarray.empty_like(d_input)

    block: Tuple[int, int, int] = (16, 16, 1)
    grid: Tuple[int, int, int] = (
        (ancho + block[0] - 1) // block[0],
        (altura + block[1] - 1) // block[1],
        1,
    )

    _negative_kernel(
        d_input,
        d_output,
        np.int32(ancho),
        np.int32(altura),
        block=block,
        grid=grid,
    )

    resultado = d_output.get()
    return resultado
