from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from contextlib import asynccontextmanager

from .routers import canny


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación:
    - startup: Se ejecuta al iniciar
    - shutdown: Se ejecuta al cerrar
    """
    
    yield  # La aplicación corre aquí
    
    # Shutdown
    print("Cerrando aplicación...")
    
    print("Aplicación cerrada correctamente")


app = FastAPI(
    title="GPU Image Processing",
    lifespan=lifespan
)

# CORS para permitir pruebas desde navegador si fuera necesario
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


app.include_router(canny.router, prefix="/canny", tags=["Canny"])

@app.get("/health", status_code=200, tags=["Health"])
def health_check():
    return {"service": "GPU-Processing", "status": "healthy"}
