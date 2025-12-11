# app/filters/collage.py
from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2
from pathlib import Path

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# Importar los otros filtros
from app.filters.canny import aplicar_canny_cuda
from app.filters.emboss import aplicar_emboss_cuda
from app.filters.gaussian import aplicar_gaussian_cuda
from app.filters.negative import aplicar_negative_cuda
from app.filters.watermark import aplicar_watermark_cuda
from app.filters.ripple import aplicar_ripple_cuda # Este es el Comic

# Definición del kernel Compositor
CUDA_KERNEL_COLLAGE = r"""
extern "C"
__global__ void collage_compositor(
    unsigned char *out,
    unsigned char *img0, // Canny
    unsigned char *img1, // Emboss
    unsigned char *img2, // Gaussian
    unsigned char *img3, // Negative
    unsigned char *img4, // Watermark
    unsigned char *img5, // Comic
    int width, int height
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    // Calcular posición diagonal usando coordenadas normalizadas
    // Esto distribuye el área de manera más equitativa
    float norm_x = (float)x / (float)width;
    float norm_y = (float)y / (float)height;
    
    // Posición diagonal de 0.0 a 1.0
    float diagonal_pos = (norm_x + norm_y) / 2.0f;
    
    // Convertir a sección (0-5)
    int section = (int)(diagonal_pos * 6.0f);
    
    // Clamp por seguridad
    if (section < 0) section = 0;
    if (section > 5) section = 5;

    unsigned char b, g, r;

    if (section == 0) {
        b = img0[idx]; g = img0[idx+1]; r = img0[idx+2];
    } else if (section == 1) {
        b = img1[idx]; g = img1[idx+1]; r = img1[idx+2];
    } else if (section == 2) {
        b = img2[idx]; g = img2[idx+1]; r = img2[idx+2];
    } else if (section == 3) {
        b = img3[idx]; g = img3[idx+1]; r = img3[idx+2];
    } else if (section == 4) {
        b = img4[idx]; g = img4[idx+1]; r = img4[idx+2];
    } else {
        b = img5[idx]; g = img5[idx+1]; r = img5[idx+2];
    }

    // Dibujar líneas negras en los bordes entre secciones
    float exact_pos = diagonal_pos * 6.0f;
    float dist_to_border = exact_pos - (float)section;
    
    // Si estamos cerca del borde de la siguiente sección (>0.97)
    if (dist_to_border > 0.97f && section < 5) {
        b = 0; g = 0; r = 0;
    }

    out[idx]     = b;
    out[idx + 1] = g;
    out[idx + 2] = r;
}
"""

_mod_collage = SourceModule(CUDA_KERNEL_COLLAGE)
_collage_kernel = _mod_collage.get_function("collage_compositor")

def aplicar_collage_cuda(image: np.ndarray) -> np.ndarray:
    """
    Genera un collage con 6 filtros en franjas diagonales.
    Orden: Canny, Emboss, Gaussian, Negative, Watermark, Comic.
    """
    altura, ancho = image.shape[:2]
    
    # --- 1. Generar las 6 imágenes ---
    
    # 1. Canny
    # Canny devuelve 1 canal, necesitamos convertir a 3 canales BGR
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_gray = aplicar_canny_cuda(gray_img, tamanio_kernel=5, sigma=1.4)
    img_canny = cv2.cvtColor(canny_gray, cv2.COLOR_GRAY2BGR)

    # 2. Emboss
    img_emboss = aplicar_emboss_cuda(image, kernel_size=3, bias=128)

    # 3. Gaussian
    # Gaussian también necesita escala de grises
    gaussian_gray = aplicar_gaussian_cuda(gray_img, kernel_size=15, sigma=5.0)
    img_gaussian = cv2.cvtColor(gaussian_gray, cv2.COLOR_GRAY2BGR)

    # 4. Negative
    img_negative = aplicar_negative_cuda(image)

    # 5. Watermark
    # Necesitamos el path del logo. Asumimos ubicación relativa fija o lo buscamos.
    # Hack: Construir ruta relativa al archivo actual
    base_dir = Path(__file__).resolve().parent.parent
    logo_path = base_dir / "static" / "UPS.png"
    img_watermark = aplicar_watermark_cuda(image, str(logo_path), scale=0.3, transparency=0.3, spacing=0.5)

    # 6. Comic (Ripple)
    img_comic = aplicar_ripple_cuda(image, edge_threshold=100.0, color_levels=4, saturation=1.2)

    # --- 2. Preparar GPU ---
    
    # Asegurar uint8 y contiguous
    imgs = [img_canny, img_emboss, img_gaussian, img_negative, img_watermark, img_comic]
    gpu_ptrs = []
    
    for img in imgs:
        if img.shape != (altura, ancho, 3):
            # Resize por seguridad si algún filtro cambió tamaño (no debería)
            img = cv2.resize(img, (ancho, altura))
        
        img_u8 = img.astype(np.uint8, copy=False)
        gpu_ptrs.append(gpuarray.to_gpu(img_u8))

    d_out = gpuarray.empty_like(gpu_ptrs[0])

    block: Tuple[int, int, int] = (16, 16, 1)
    grid: Tuple[int, int, int] = (
        (ancho + block[0] - 1) // block[0],
        (altura + block[1] - 1) // block[1],
        1,
    )

    _collage_kernel(
        d_out,
        gpu_ptrs[0], # Canny
        gpu_ptrs[1], # Emboss
        gpu_ptrs[2], # Gaussian
        gpu_ptrs[3], # Negative
        gpu_ptrs[4], # Watermark
        gpu_ptrs[5], # Comic
        np.int32(ancho),
        np.int32(altura),
        block=block,
        grid=grid
    )

    return d_out.get()
