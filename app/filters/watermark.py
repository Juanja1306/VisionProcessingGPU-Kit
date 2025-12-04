# app/filters/watermark.py
from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2
import os

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# Definición del kernel CUDA
CUDA_KERNEL_WATERMARK = r"""
extern "C"
__global__ void watermark(
    unsigned char *input,
    unsigned char *logo,
    unsigned char *output,
    int width, int height,
    int logo_width, int logo_height,
    int spacing_x, int spacing_y,
    float global_alpha
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    
    // Tamaño total de la celda (logo + espacio)
    int cell_width = logo_width + spacing_x;
    int cell_height = logo_height + spacing_y;
    
    // Coordenadas dentro de la celda
    int cell_x = x % cell_width;
    int cell_y = y % cell_height;
    
    // Verificar si estamos sobre el logo o sobre el espacio
    if (cell_x < logo_width && cell_y < logo_height) {
        // Estamos sobre el logo
        int logo_idx = (cell_y * logo_width + cell_x) * 4;

        unsigned char lb = logo[logo_idx];     // B
        unsigned char lg = logo[logo_idx + 1]; // G
        unsigned char lr = logo[logo_idx + 2]; // R
        unsigned char la = logo[logo_idx + 3]; // A

        unsigned char ib = input[idx];
        unsigned char ig = input[idx + 1];
        unsigned char ir = input[idx + 2];

        // Mezcla:
        // alpha efectivo = global_alpha * (alpha del pixel del logo / 255.0)
        float alpha = global_alpha * (la / 255.0f);

        // BGR
        output[idx]     = (unsigned char)(lb * alpha + ib * (1.0f - alpha));
        output[idx + 1] = (unsigned char)(lg * alpha + ig * (1.0f - alpha));
        output[idx + 2] = (unsigned char)(lr * alpha + ir * (1.0f - alpha));
    } else {
        // Estamos en el espacio, copiar pixel original
        output[idx]     = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
    }
}
"""

_mod_watermark = SourceModule(CUDA_KERNEL_WATERMARK)
_watermark_kernel = _mod_watermark.get_function("watermark")

def aplicar_watermark_cuda(imagen_color: np.ndarray, logo_path: str, scale: float, transparency: float, spacing: float = 0.5) -> np.ndarray:
    """
    Aplica una marca de agua (logo) repetida en cuadrícula sobre la imagen con espaciado.
    
    Args:
        imagen_color: Imagen de entrada (H, W, 3) BGR.
        logo_path: Ruta al archivo de imagen del logo.
        scale: Escala del logo respecto al ancho de la imagen (0.0 - 1.0).
        transparency: Nivel de opacidad del logo (0.0 - 1.0).
        spacing: Espaciado entre logos como fracción del tamaño del logo (0.0+).
    """
    if imagen_color.ndim != 3:
        raise ValueError("Se esperaba una imagen de 3 dimensiones (H, W, C)")

    altura, ancho, canales = imagen_color.shape
    
    # Cargar logo
    if not os.path.exists(logo_path):
        raise FileNotFoundError(f"No se encontró el logo en: {logo_path}")
        
    # Leer con alpha (IMREAD_UNCHANGED) para obtener RGBA/BGRA
    logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo_img is None:
        raise ValueError("Error al leer el archivo del logo")

    # Asegurar que tenga 4 canales (BGRA)
    if logo_img.ndim == 2:
        # Grayscale -> BGRA
        logo_img = cv2.cvtColor(logo_img, cv2.COLOR_GRAY2BGRA)
    elif logo_img.shape[2] == 3:
        # BGR -> BGRA
        logo_img = cv2.cvtColor(logo_img, cv2.COLOR_BGR2BGRA)
    
    # Calcular nuevo tamaño del logo
    target_width = int(ancho * scale)
    if target_width < 1: target_width = 1
    
    aspect_ratio = logo_img.shape[0] / logo_img.shape[1]
    target_height = int(target_width * aspect_ratio)
    if target_height < 1: target_height = 1
    
    # Redimensionar logo
    logo_resized = cv2.resize(logo_img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    logo_h, logo_w = logo_resized.shape[:2]
    
    # Calcular espaciado en píxeles
    spacing_x = int(logo_w * spacing)
    spacing_y = int(logo_h * spacing)
    
    # Preparar datos para GPU
    img_uint8 = imagen_color.astype(np.uint8, copy=False)
    logo_uint8 = logo_resized.astype(np.uint8, copy=False)
    
    d_input = gpuarray.to_gpu(img_uint8)
    d_logo = gpuarray.to_gpu(logo_uint8)
    d_output = gpuarray.empty_like(d_input)
    
    block: Tuple[int, int, int] = (16, 16, 1)
    grid: Tuple[int, int, int] = (
        (ancho + block[0] - 1) // block[0],
        (altura + block[1] - 1) // block[1],
        1,
    )
    
    _watermark_kernel(
        d_input,
        d_logo,
        d_output,
        np.int32(ancho),
        np.int32(altura),
        np.int32(logo_w),
        np.int32(logo_h),
        np.int32(spacing_x),
        np.int32(spacing_y),
        np.float32(transparency),
        block=block,
        grid=grid,
    )
    
    return d_output.get()
