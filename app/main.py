from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle:
    - startup: Executes on startup
    - shutdown: Executes on shutdown
    """
    
    yield  # Application runs here
    
    # Shutdown
    print("Closing application...")
    
    print("Application closed successfully")


app = FastAPI(
    title="GPU Image Processing",
    lifespan=lifespan
)

# CORS to allow testing from browser if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


#app.include_router(canny.router, tags=["Canny"])

@app.get("/", tags=["UI"])
async def read_index():
    return FileResponse(static_dir / "index.html")

@app.get("/emboss", tags=["UI"])
async def read_emboss():
    return FileResponse(static_dir / "emboss" / "emboss.html")


@app.get("/health", status_code=200, tags=["Health"])
def health_check():
    return {"service": "GPU-Processing", "status": "healthy"}
    