# VisionProcessingGPU-Kit

A GPU-accelerated image processing microservice built with **FastAPI**, **PyCUDA**, and fully containerized with **Docker** for seamless deployment in any environment. This project provides high-performance GPU filters executed directly on the GPU to deliver significantly improved performance over CPU-based approaches.

## 🚀 Features

- **GPU-Accelerated Processing**: Leverages NVIDIA CUDA for parallel image processing
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Modern Web UI**: Premium, responsive interface with real-time parameter adjustments
- **Auto Mode**: Intelligent parameter selection based on image resolution
- **Docker Support**: Fully containerized with NVIDIA GPU support
- **Cross-Platform**: Works on both Windows and Linux environments

## 🎨 Available Filters

### ✅ Implemented

- **Canny Edge Detection**: Full GPU implementation with automatic threshold detection
  - Gaussian blur smoothing
  - Sobel gradient calculation
  - Non-maximum suppression
  - Hysteresis edge tracking
  - **Auto mode**: Automatically calculates kernel size, sigma, and thresholds based on image resolution

- **Gaussian Blur**
  - Works directly on **color images (BGR)**.
  - Configurable **kernel size** (odd values, up to large kernels).
  - Configurable **sigma** for blur intensity.
  - **Auto mode**: chooses kernel size and sigma based on image size.

- **Negative Filter**
  - Full-color inversion on the GPU.
  - No parameters: upload an image and invert all RGB channels.

- **Emboss Filter**
  - Creates a 3D embossing effect on RGB images
  - Configurable **kernel size** (3x3, 5x5, 7x7, 9x9)
  - Configurable **bias value** for brightness adjustment (0-255)
  - **Auto mode**: Automatically selects kernel size based on image resolution
  - Processes all color channels simultaneously

- **Collage Filter**
  - Combines 6 different filters in diagonal strips.
  - Includes: Canny, Emboss, Gaussian, Negative, Watermark, and Comic.
  - **Auto mode**: Automatically adjusts parameters for each sub-filter.

- **Ripple (Comic) Filter**
  - Creates a comic book style effect.
  - **Edge detection**: Highlights edges using Sobel operator.
  - **Posterization**: Reduces color levels for a cartoon look.
  - **Saturation**: Boosts color vibrancy.

- **Watermark Filter**
  - Overlays a logo in a repeated grid pattern.
  - Configurable **scale**, **transparency**, and **spacing**.

## 🏗️ Architecture

```
VisionProcessingGPU-Kit/
├── app/
│   ├── core/
│   │   └── cuda_config.py        # Cross-platform CUDA environment setup
│   ├── filters/
│   │   ├── canny.py              # CUDA kernels and Canny implementation
│   │   ├── collage.py            # Collage filter implementation
│   │   ├── emboss.py             # Emboss filter CUDA implementation (RGB)
│   │   ├── gaussian.py           # Gaussian blur CUDA implementation (color)
│   │   ├── negative.py           # Negative filter CUDA implementation (color)
│   │   ├── ripple.py             # Ripple/Comic filter CUDA implementation
│   │   └── watermark.py          # Watermark filter CUDA implementation
│   ├── routers/
│   │   ├── canny.py              # FastAPI endpoint for Canny filter
│   │   ├── collage.py            # FastAPI endpoint for Collage filter
│   │   ├── emboss.py             # FastAPI endpoint for Emboss filter
│   │   ├── gaussian.py           # FastAPI endpoint for Gaussian blur
│   │   ├── negative.py           # FastAPI endpoint for Negative filter
│   │   ├── ripple.py             # FastAPI endpoint for Ripple filter
│   │   └── watermark.py          # FastAPI endpoint for Watermark filter
│   ├── schemas/
│   │   ├── canny.py              # Pydantic models for Canny parameters
│   │   ├── collage.py            # Pydantic models for Collage parameters
│   │   ├── emboss.py             # Pydantic model for Emboss parameters
│   │   ├── gaussian.py           # Pydantic models for Gaussian parameters
│   │   ├── negative.py           # Pydantic model for Negative filter
│   │   ├── ripple.py             # Pydantic models for Ripple parameters
│   │   └── watermark.py          # Pydantic models for Watermark parameters
│   ├── static/
│   │   └── index.html            # Premium web UI (all filters with auto mode)
│   └── main.py                   # FastAPI application entry point
├── dockerfile                    # Docker configuration with CUDA base image
├── requirements.txt              # Python dependencies
└── README.md
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.12
- **GPU Computing**: PyCUDA, NVIDIA CUDA 12.6
- **Image Processing**: OpenCV, NumPy
- **Frontend**: Vanilla HTML/CSS/JavaScript with modern design
- **Containerization**: Docker with NVIDIA GPU support

## 📋 Requirements

### Local Development

- Python 3.12+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.6
- Visual Studio Build Tools (Windows only)

### Docker Deployment

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with compatible drivers

## 🚀 Getting Started

### Local Development

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/VisionProcessingGPU-Kit.git
   cd VisionProcessingGPU-Kit
   ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

5. **Access the web UI**

   ```
   http://localhost:8000
   ```

### Docker Deployment

1. **Build the Docker image**

   ```bash
   docker build -t gpu-vision-kit .
   ```

2. **Run with GPU support**

   ```bash
   docker run --gpus all -p 8000:8000 gpu-vision-kit
   ```

3. **Access the application**

   ```
   http://localhost:8000
   ```

### Subir a Docker Hub

Para subir la imagen a Docker Hub y evitar tener que compilarla cada vez:

1. **Crear cuenta en Docker Hub** 
   - Visitar https://hub.docker.com y crear una cuenta

2. **Iniciar sesión desde la terminal**

   ```bash
   docker login
   ```
   - Ingresar el username y password de Docker Hub

3. **Taggear la imagen con tu username de Docker Hub**

   ```bash
   docker tag gpu-vision-kit juanja/gpu-vision-kit:latest
   ```

4. **Subir la imagen a Docker Hub**

   ```bash
   docker push juanja/gpu-vision-kit:latest
   ```

5. **Usar la imagen desde Docker Hub (sin compilar)**

   ```bash
   docker build -t gpu-vision-kit .
   docker pull juanja/gpu-vision-kit:latest
   docker run --gpus all -p 8000:8000 juanja/gpu-vision-kit:latest
   ```

## 📡 API Endpoints

### Canny Edge Detection

```text
POST /api/canny
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- kernel_size: Gaussian kernel size (default: 5, must be odd)
- sigma: Gaussian sigma (default: 1.4)
- low_threshold: Low threshold for hysteresis (optional, auto if not provided)
- high_threshold: High threshold for hysteresis (optional, auto if not provided)
- use_auto: Enable automatic parameter calculation based on image size (default: false)

Response: PNG image with detected edges
```

**Auto Mode Behavior** (when `use_auto=true`):
- ≤ 1080px (HD): `kernel_size=5`, `sigma=1.4`
- ≤ 2160px (2K): `kernel_size=7`, `sigma=2.0`
- ≤ 4320px (4K): `kernel_size=9`, `sigma=2.5`
- > 4320px (8K+): `kernel_size=11`, `sigma=3.0`
- Thresholds calculated automatically

### Gaussian Blur

```text
POST /api/gaussian
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- kernel_size: Gaussian kernel size (default: 15)
- sigma: Gaussian sigma (default: 5)
- use_auto: Enable automatic parameter calculation based on image size (default: false)

Response: PNG image with applied blur
```

**Auto Mode Behavior** (when `use_auto=true`):
- ≤ 1080px (HD): `kernel_size=15`
- ≤ 2160px (2K): `kernel_size=31`
- ≤ 4320px (4K): `kernel_size=41`
- > 4320px (8K+): `kernel_size=51`
- Sigma automatically set to kernel size for strong effect

### Negative Filter

```text
POST /api/negative
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)

Response: PNG image with inverted colors
```

### Emboss Filter

```text
POST /api/emboss
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- kernel_size: Emboss kernel size (default: 3, options: 3, 5, 7, 9)
- bias_value: Brightness bias (default: 128, range: 0-255)
- use_auto: Enable automatic parameter calculation based on image size (default: false)

Response: PNG image with emboss effect
```

**Auto Mode Behavior** (when `use_auto=true`):
- ≤ 1080px (HD): `kernel_size=3`
- ≤ 2160px (2K): `kernel_size=5`
- ≤ 4320px (4K): `kernel_size=7`
- > 4320px (8K+): `kernel_size=9`
- Bias always set to 128

### Collage Filter

```text
POST /api/collage
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)

Response: PNG image with the collage effect
```

### Ripple (Comic) Filter

```text
POST /api/ripple
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- edge_threshold: Threshold for edge detection (default: 100.0)
- color_levels: Number of color levels for posterization (default: 8)
- saturation: Color saturation factor (default: 1.2)

Response: PNG image with comic effect
```

### Watermark Filter

```text
POST /api/watermark
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- scale: Logo scale relative to image width (default: 0.3)
- transparency: Logo opacity (default: 0.3)
- spacing: Spacing between logos (default: 0.5)

Response: PNG image with watermark pattern
```

### Health Check

```http
GET /health

Response: {"service": "GPU-Processing", "status": "healthy"}
```

## 🎯 CUDA Implementation Details

### Canny Edge Detection

The Canny edge detection filter uses custom CUDA kernels for:

1. **Gaussian Convolution**: Parallel smoothing with configurable kernel
2. **Sobel Gradients**: Simultaneous X and Y gradient calculation
3. **Non-Maximum Suppression**: Edge thinning based on gradient direction
4. **Hysteresis Thresholding**: Iterative edge connection on GPU
5. **Edge Cleanup**: Final pass to remove weak edges

All operations are performed on the GPU, minimizing CPU-GPU data transfers.

**Auto Mode**: Automatically selects optimal parameters based on image resolution (min of width/height):
- Kernel size and sigma scale with image size
- Thresholds calculated from gradient magnitudes
- Optimized for edge detection quality across different resolutions

### Gaussian Blur

- Generates a 2D Gaussian kernel on the CPU (normalized).
- Transfers kernel + image to GPU.
- Applies convolution in parallel across the image.
- Operates on all color channels to preserve the color structure of the image.

Supports both manual parameters and an **auto mode** that adapts to the image resolution for consistent blur effect regardless of image size.

### Negative Filter

Simple, highly parallel kernel:

- One thread per pixel (per channel group).
- Performs `255 - value` on each color channel.

Extremely fast due to low arithmetic intensity and fully parallel execution.

### Emboss Filter

GPU-accelerated embossing with RGB support:

- **Custom CUDA kernel** applies directional convolution to create 3D relief effect
- **Multi-kernel support**: Predefined kernels for 3x3, 5x5, 7x7, and 9x9 sizes
- **Parallel processing**: One thread per pixel, processes all RGB channels simultaneously
- **Border handling**: Preserves original pixels at image borders to avoid artifacts
- **Bias adjustment**: Configurable brightness offset (0-255) for fine-tuning

**Auto Mode**: Adapts kernel size to image resolution:
- Larger images use larger kernels for better embossing effect
- Maintains visual consistency across different image sizes
- Bias fixed at optimal value (128) for balanced results

### Collage Filter

- **Compositor Kernel**: A custom CUDA kernel that divides the image into 6 diagonal sections.
- **Parallel Execution**: Runs all 6 underlying filters (Canny, Emboss, Gaussian, Negative, Watermark, Comic) and composites them on the GPU.
- **Efficient Memory**: Manages GPU memory to handle multiple intermediate images.

### Ripple (Comic) Filter

- **Single Pass Kernel**: Performs edge detection and color processing in one go.
- **Sobel Edge Detection**: Calculates gradient magnitude to identify and darken edges.
- **Color Quantization**: Reduces color palette to create a posterized look.
- **Saturation Boost**: Enhances colors in the HSV-like color space (simplified).

### Watermark Filter

- **Grid Blending Kernel**: Calculates logo positions based on a grid layout.
- **Alpha Blending**: Blends the logo with the background image using the alpha channel.
- **Seamless Tiling**: Handles spacing and repetition efficiently on the GPU.

## 🌐 Web Interface

The web UI features:

- **Drag & drop** image upload
- **Real-time parameter** adjustment with live value display
- **Auto mode toggle** for Canny, Gaussian, and Emboss filters
- **Side-by-side comparison** of original and processed images
- **Modern glassmorphism** design with smooth animations
- **Responsive layout** for various screen sizes
- **Dynamic controls** that adapt to each filter's requirements

## 🔧 Configuration

### CUDA Environment

The application automatically configures CUDA paths based on the operating system:

- **Windows**: Sets Visual Studio compiler paths
- **Linux**: Configures CUDA bin paths

Configuration is handled in `app/core/cuda_config.py` and initialized at application startup.

## 📝 License

MIT License - see LICENSE file for details
