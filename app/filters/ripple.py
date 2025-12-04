# app/filters/ripple.py
from __future__ import annotations

from typing import Tuple
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# Definición del kernel CUDA
# Implementa efecto Comic: Bordes negros (Sobel) + Posterización de color
CUDA_KERNEL_COMIC = r"""
extern "C"
__global__ void comic_filter(
    unsigned char *input,
    unsigned char *output,
    int width, int height,
    float edge_threshold,
    int color_levels,
    float saturation
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    // 1. Detección de Bordes (Sobel simplificado)
    // Necesitamos vecinos. Si estamos en borde, asumimos valor del pixel central
    // Sobel Kernels:
    // Gx: -1 0 1   Gy: -1 -2 -1
    //     -2 0 2        0  0  0
    //     -1 0 1        1  2  1
    
    float gx_b = 0, gx_g = 0, gx_r = 0;
    float gy_b = 0, gy_g = 0, gy_r = 0;

    for(int j=-1; j<=1; j++){
        for(int i=-1; i<=1; i++){
            int nx = x + i;
            int ny = y + j;
            
            // Clamp coordinates
            if(nx < 0) nx = 0; if(nx >= width) nx = width - 1;
            if(ny < 0) ny = 0; if(ny >= height) ny = height - 1;
            
            int n_idx = (ny * width + nx) * 3;
            unsigned char b = input[n_idx];
            unsigned char g = input[n_idx+1];
            unsigned char r = input[n_idx+2];
            
            // Sobel X weights
            float wx = 0;
            if(i == -1) wx = (j == 0) ? -2 : -1;
            if(i == 1)  wx = (j == 0) ?  2 :  1;
            
            // Sobel Y weights
            float wy = 0;
            if(j == -1) wy = (i == 0) ? -2 : -1;
            if(j == 1)  wy = (i == 0) ?  2 :  1;
            
            gx_b += b * wx; gx_g += g * wx; gx_r += r * wx;
            gy_b += b * wy; gy_g += g * wy; gy_r += r * wy;
        }
    }

    float mag_b = sqrtf(gx_b*gx_b + gy_b*gy_b);
    float mag_g = sqrtf(gx_g*gx_g + gy_g*gy_g);
    float mag_r = sqrtf(gx_r*gx_r + gy_r*gy_r);
    
    float magnitude = (mag_b + mag_g + mag_r) / 3.0f;

    // Si es borde, pintar negro
    if (magnitude > edge_threshold) {
        output[idx]     = 0;
        output[idx + 1] = 0;
        output[idx + 2] = 0;
        return;
    }

    // 2. Posterización y Saturación
    unsigned char b = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char r = input[idx + 2];

    // Convertir a float
    float fb = (float)b;
    float fg = (float)g;
    float fr = (float)r;

    // Saturación simple
    // Luminancia aprox
    float gray = 0.114f * fb + 0.587f * fg + 0.299f * fr;
    
    fb = gray + (fb - gray) * saturation;
    fg = gray + (fg - gray) * saturation;
    fr = gray + (fr - gray) * saturation;

    // Clamp post-saturación
    if(fb < 0) fb = 0; if(fb > 255) fb = 255;
    if(fg < 0) fg = 0; if(fg > 255) fg = 255;
    if(fr < 0) fr = 0; if(fr > 255) fr = 255;

    // Posterización (Quantization)
    float step = 255.0f / (float)(color_levels - 1);
    
    fb = floorf(fb / step + 0.5f) * step;
    fg = floorf(fg / step + 0.5f) * step;
    fr = floorf(fr / step + 0.5f) * step;

    // Clamp final
    if(fb < 0) fb = 0; if(fb > 255) fb = 255;
    if(fg < 0) fg = 0; if(fg > 255) fg = 255;
    if(fr < 0) fr = 0; if(fr > 255) fr = 255;

    output[idx]     = (unsigned char)fb;
    output[idx + 1] = (unsigned char)fg;
    output[idx + 2] = (unsigned char)fr;
}
"""

_mod_comic = SourceModule(CUDA_KERNEL_COMIC)
_comic_kernel = _mod_comic.get_function("comic_filter")

def aplicar_ripple_cuda(imagen_color: np.ndarray, edge_threshold: float, color_levels: int, saturation: float) -> np.ndarray:
    """
    Aplica un efecto de Comic Book (Bordes + Posterización).
    
    Args:
        imagen_color: Imagen de entrada (H, W, 3) BGR.
        edge_threshold: Umbral para bordes (0-255).
        color_levels: Niveles de cuantización de color.
        saturation: Factor de saturación.
    """
    if imagen_color.ndim != 3:
        raise ValueError("Se esperaba una imagen de 3 dimensiones (H, W, C)")

    altura, ancho, canales = imagen_color.shape
    
    if canales == 4:
        imagen_color = imagen_color[:, :, :3]
        canales = 3

    # Preparar datos para GPU
    img_uint8 = imagen_color.astype(np.uint8, copy=False)
    
    d_input = gpuarray.to_gpu(img_uint8)
    d_output = gpuarray.empty_like(d_input)
    
    block: Tuple[int, int, int] = (16, 16, 1)
    grid: Tuple[int, int, int] = (
        (ancho + block[0] - 1) // block[0],
        (altura + block[1] - 1) // block[1],
        1,
    )
    
    _comic_kernel(
        d_input,
        d_output,
        np.int32(ancho),
        np.int32(altura),
        np.float32(edge_threshold),
        np.int32(color_levels),
        np.float32(saturation),
        block=block,
        grid=grid,
    )
    
    return d_output.get()
