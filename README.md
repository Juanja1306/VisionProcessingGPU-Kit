# VisionProcessingGPU-Kit

A GPU-accelerated image processing microservice built with **FastAPI**, **PyCUDA**, and fully containerized with **Docker** for seamless deployment in any environment. This project provides high-performance GPU filters executed directly on the GPU to deliver significantly improved performance over CPU-based approaches.

## 🚀 Features

- **GPU-Accelerated Processing**: Leverages NVIDIA CUDA for parallel image processing
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Modern Web UI**: Premium, responsive interface with real-time parameter adjustments
- **Docker Support**: Fully containerized with NVIDIA GPU support
- **Cross-Platform**: Works on both Windows and Linux environments

## 🎨 Available Filters

### ✅ Implemented

- **Canny Edge Detection**: Full GPU implementation with automatic threshold detection
  - Gaussian blur smoothing
  - Sobel gradient calculation
  - Non-maximum suppression
  - Hysteresis edge tracking

- **Gaussian Blur** (GPU)
  - Works directly on **color images (BGR)**.
  - Configurable **kernel size** (odd values, up to large kernels).
  - Configurable **sigma** for blur intensity.
  - Optional **auto mode**: chooses kernel size and sigma based on image size.

- **Negative Filter** (GPU)
  - Full-color inversion on the GPU.
  - No parameters: upload an image and invert all RGB channels.

### 🔜 Coming Soon

The following filters have UI placeholders and will be implemented:

- **Emboss**: 3D embossing effect

## 🏗️ Architecture

```
VisionProcessingGPU-Kit/
├── app/
│   ├── core/
│   │   └── cuda_config.py        # Cross-platform CUDA environment setup
│   ├── filters/
│   │   ├── canny.py              # CUDA kernels and Canny implementation
│   │   ├── gaussian.py           # Gaussian blur CUDA implementation (color)
│   │   └── negative.py           # Negative filter CUDA implementation (color)
│   ├── routers/
│   │   ├── canny.py              # FastAPI endpoint for Canny filter
│   │   ├── gaussian.py           # FastAPI endpoint for Gaussian blur
│   │   └── negative.py           # FastAPI endpoint for Negative filter
│   ├── schemas/
│   │   ├── canny.py              # Pydantic models for Canny parameters
│   │   ├── gaussian.py           # Pydantic models for Gaussian parameters
│   │   └── negative.py           # Pydantic model (placeholder) for Negative filter
│   ├── static/
│   │   └── index.html            # Premium web UI (Canny / Gaussian / Negative)
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

Response: PNG image with detected edges
```

### Gaussian Blur

```text
POST /api/gaussian
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- kernel_size: Gaussian kernel size (default: 15)
- sigma: Gaussian sigma (default: 5)
- auto: Use auto mode (default: false)

Response: PNG image with applied blur
```

### Negative Filter

```text
POST /api/negative
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)

Response: PNG image with inverted colors
```

### Health Check

```http
GET /health

Response: {"service": "GPU-Processing", "status": "healthy"}
```

## 🎯 CUDA Implementation Details

The Canny edge detection filter uses custom CUDA kernels for:

1. **Gaussian Convolution**: Parallel smoothing with configurable kernel
2. **Sobel Gradients**: Simultaneous X and Y gradient calculation
3. **Non-Maximum Suppression**: Edge thinning based on gradient direction
4. **Hysteresis Thresholding**: Iterative edge connection on GPU
5. **Edge Cleanup**: Final pass to remove weak edges

All operations are performed on the GPU, minimizing CPU-GPU data transfers.

### Gaussian Blur

- Generates a 2D Gaussian kernel on the CPU (normalized).

- Transfers kernel + image to GPU.

- Applies convolution in parallel across the image.

- Operates on all color channels to preserve the color structure of the image.

Supports both manual parameters and an “auto” mode that adapts to the image resolution.

### Negative Filter

Simple, highly parallel kernel:

- One thread per pixel (per channel group).

- Performs 255 - value on each color channel.

Extremely fast due to low arithmetic intensity and fully parallel execution.

## 🌐 Web Interface

The web UI features:

- **Drag & drop** image upload
- **Real-time parameter** adjustment with live value display
- **Side-by-side comparison** of original and processed images
- **Modern glassmorphism** design with smooth animations
- **Responsive layout** for various screen sizes

## 🔧 Configuration

### CUDA Environment

The application automatically configures CUDA paths based on the operating system:

- **Windows**: Sets Visual Studio compiler paths
- **Linux**: Configures CUDA bin paths

Configuration is handled in `app/core/cuda_config.py` and initialized at application startup.

## 📝 License

MIT License - see LICENSE file for details
