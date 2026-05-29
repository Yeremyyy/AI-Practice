"""
API REST con FastAPI - Production Ready
Deployment listo con validación, logging, y manejo de errores
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import uvicorn
import logging
import json
from datetime import datetime
import os

from ai_engine import AIEngine, ModelConfig
from rag_system import RAGSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="AI Engine API",
    description="Motor IA gratuito production-ready con arquitectura neural",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Inicializar motor IA (una sola instancia)
config = ModelConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    quantize=True,
    temperature=0.7
)

logger.info("Inicializando AIEngine...")
engine = AIEngine(config)
rag = RAGSystem(engine)
logger.info("✓ AIEngine inicializado")


# ==================== MODELOS PYDANTIC ====================

class GenerateRequest(BaseModel):
    """Solicitud de generación de texto"""
    prompt: str = Field(..., min_length=1, max_length=5000, description="Texto de entrada")
    max_tokens: int = Field(512, ge=1, le=2048, description="Máximo de tokens a generar")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Control de creatividad")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling")
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0, description="Penalidad por repetición")
    
    @validator('prompt')
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt no puede estar vacío")
        return v


class GenerateResponse(BaseModel):
    """Respuesta de generación"""
    response: str
    tokens_generated: int
    time_elapsed: float
    tokens_per_second: float


class RAGRequest(BaseModel):
    """Solicitud con contexto RAG"""
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(3, ge=1, le=10)
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class RAGResponse(BaseModel):
    """Respuesta con contexto"""
    response: str
    sources: List[str]
    context_used: int
    tokens_generated: int
    time_elapsed: float


class DocumentRequest(BaseModel):
    """Solicitud para añadir documento"""
    doc_id: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=10, max_length=50000)
    metadata: Optional[Dict] = Field(None)


class RetrieveRequest(BaseModel):
    """Solicitud para recuperar documentos"""
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(3, ge=1, le=10)
    threshold: float = Field(0.3, ge=0.0, le=1.0)


# ==================== ENDPOINTS ====================

@app.get("/", tags=["Info"])
async def root():
    """Información general de la API"""
    return {
        "name": "AI Engine API",
        "version": "1.0.0",
        "description": "Motor IA gratuito con arquitectura neural completa",
        "endpoints": {
            "generate": "/generate - Generar texto",
            "rag": "/rag - Generar con contexto",
            "documents": "/documents - Gestionar base de conocimiento",
            "health": "/health - Estado del servicio",
            "docs": "/docs - Documentación interactiva"
        }
    }


@app.get("/health", tags=["Monitor"])
async def health_check():
    """Health check del servicio"""
    try:
        health_status = {
            "status": "healthy",
            "device": str(engine.device),
            "model": engine.config.model_name,
            "timestamp": datetime.now().isoformat(),
            "rag_documents": len(rag.knowledge_base)
        }
        
        if engine.device.type == "cuda":
            import torch
            health_status["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1e9
        
        return health_status
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(request: GenerateRequest):
    """
    Genera texto basado en un prompt
    
    Parámetros:
    - prompt: Texto de entrada
    - max_tokens: Máximo de tokens (1-2048)
    - temperature: Creatividad (0-2, 0.7 recomendado)
    - top_p: Nucleus sampling (0-1)
    - top_k: Top-k sampling
    - repetition_penalty: Penalidad por repetición
    """
    try:
        logger.info(f"Generación solicitada: {request.prompt[:50]}...")
        
        result = engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        logger.info(f"✓ Generación completada ({result['tokens_generated']} tokens)")
        
        return GenerateResponse(
            response=result["response"],
            tokens_generated=result["tokens_generated"],
            time_elapsed=result["time_elapsed"],
            tokens_per_second=result["tokens_per_second"]
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error en generación: {str(e)}")


@app.post("/rag", response_model=RAGResponse, tags=["RAG"])
async def rag_generate(request: RAGRequest):
    """
    Genera respuesta usando contexto RAG
    
    El sistema busca documentos relevantes y los usa como contexto
    """
    try:
        logger.info(f"RAG generation solicitado: {request.query[:50]}...")
        
        result = rag.generate_with_context(
            query=request.query,
            top_k=request.top_k,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        logger.info(f"✓ RAG generación completada (fuentes: {len(result['sources'])})")
        
        return RAGResponse(
            response=result["response"],
            sources=result["sources"],
            context_used=result["context_used"],
            tokens_generated=result["tokens_generated"],
            time_elapsed=result["time_elapsed"]
        )
    
    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=f"Error en RAG: {str(e)}")


@app.post("/documents", tags=["Documents"])
async def add_document(request: DocumentRequest):
    """Añade un documento a la base de conocimiento"""
    try:
        logger.info(f"Documento solicitado: {request.doc_id}")
        
        rag.add_document(
            doc_id=request.doc_id,
            content=request.content,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "message": f"Documento {request.doc_id} añadido exitosamente",
            "doc_id": request.doc_id,
            "tokens": len(request.content.split())
        }
    
    except ValueError as e:
        logger.error(f"Document validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", tags=["Documents"])
async def retrieve_documents(request: RetrieveRequest):
    """Recupera documentos relevantes para una query"""
    try:
        results = rag.retrieve(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return {
            "query": request.query,
            "documents_found": len(results),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Retrieve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", tags=["Documents"])
async def list_documents():
    """Lista todos los documentos en la base de conocimiento"""
    try:
        documents = rag.list_documents()
        stats = rag.get_statistics()
        
        return {
            "total_documents": len(documents),
            "statistics": stats,
            "documents": documents
        }
    
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Elimina un documento de la base"""
    try:
        success = rag.remove_document(doc_id)
        
        if success:
            return {"status": "success", "message": f"Documento {doc_id} eliminado"}
        else:
            raise HTTPException(status_code=404, detail=f"Documento {doc_id} no encontrado")
    
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-cache", tags=["Monitor"])
async def clear_cache():
    """Limpia la caché GPU"""
    try:
        engine.clear_cache()
        return {"status": "success", "message": "Caché GPU limpiado"}
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ERROR HANDLERS ====================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handler para errores de validación"""
    logger.error(f"ValueError: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handler general para excepciones"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Evento de inicio"""
    logger.info("="*60)
    logger.info("API iniciada correctamente")
    logger.info("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre"""
    logger.info("API cerrando...")
    engine.clear_cache()
    logger.info("✓ API cerrada")


# ==================== MAIN ====================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Iniciando servidor en {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
