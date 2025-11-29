from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from PIL import Image
from io import BytesIO

from ..filters.emboss.Sequential import Sequential, SequentialProps
from ..filters.emboss.Cuda import CudaProps, CudaEmboss
from ..filters.emboss.utils.processImage import ProcessImage
from ..filters.emboss.emboss import Emboss, EmbossProps



router = APIRouter()

@router.post("/api/emboss")
async def applyEmboss(
    file: UploadFile = File(...),              # La imagen (FormData.append('file', ...))
    kernelSize: int = Form(3),                 # Parámetro 1 (FormData.append('kernelSize', ...))
    biasValue: int = Form(128)
):
    
    try:       
        processImage = ProcessImage()
        sequentialService = Sequential(SequentialProps(
            processImage=processImage,
            imageFile=file,
            kernelSize=kernelSize,
            biasValue=biasValue
        ))
        
        
        service_implementation = sequentialService
        
        emboss = Emboss(EmbossProps(
            serviceEmboss=service_implementation
        ))
        
        
        
        processed_image_np = await emboss.aplyFilter()       
        
        
        pil_image_out = Image.fromarray(processed_image_np)
        
        # Guardar la imagen en un buffer de bytes en memoria (PNG)
        byte_io = BytesIO()
        pil_image_out.save(byte_io, format="PNG")
        byte_io.seek(0)
        
        # Devolver el resultado como una respuesta binaria
        return Response(content=byte_io.read(), media_type="image/png")

    except Exception as e:
        # Manejo de errores
        print(f"Error procesando imagen: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno durante el procesamiento del filtro: {e.__class__.__name__}: {str(e)}"
        )