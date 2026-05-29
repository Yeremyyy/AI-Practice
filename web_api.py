"""
Web API con FastAPI + CORS
Servir archivos estáticos y API optimizada para Lighthouse 100%
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
import os
from pathlib import Path

from ai_engine import AIEngine, ModelConfig
from rag_system import RAGSystem

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== APP SETUP ====================
app = FastAPI(
    title="AI Engine API",
    description="Motor IA production-ready con interfaz web Lighthouse 100%",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# ==================== MIDDLEWARE ====================

# CORS - Permitir solicitudes de la web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZIP Compression - Comprime respuestas para mejor performance
app.add_middleware(GZIPMiddleware, minimum_size=1000)

# Trusted Host - Seguridad
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
)

# ==================== INICIALIZAR IA ====================
logger.info("Inicializando AIEngine...")
config = ModelConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    quantize=True,
    temperature=0.7
)

engine = AIEngine(config)
rag = RAGSystem(engine)
logger.info("✓ AIEngine inicializado")

# ==================== ARCHIVOS ESTÁTICOS ====================
# Crear directorio de static si no existe
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Montar archivos estáticos con headers de caché óptimos
app.mount(
    "/static",
    StaticFiles(
        directory=str(static_dir),
        html=False,
        check_dir=True
    ),
    name="static"
)

# ==================== RUTAS ====================

@app.get("/", response_class=FileResponse)
async def serve_index():
    """Servir página principal con caché óptimo"""
    return FileResponse(
        "static/index.html",
        media_type="text/html"
    )

@app.get("/health")
async def health_check():
    """Health check"""
    try:
        import torch
        health_data = {
            "status": "healthy",
            "device": str(engine.device),
            "model": engine.config.model_name,
            "rag_documents": len(rag.knowledge_base)
        }
        
        if engine.device.type == "cuda":
            health_data["gpu_memory_gb"] = torch.cuda.memory_allocated() / 1e9
        
        return health_data
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "error"}

@app.post("/api/generate")
async def generate_api(request: dict):
    """API de generación optimizada"""
    try:
        prompt = request.get("prompt", "")
        
        if not prompt or len(prompt.strip()) == 0:
            return {"error": "Prompt vacío"}
        
        result = engine.generate(
            prompt=prompt,
            max_new_tokens=min(int(request.get("max_tokens", 256)), 2048),
            temperature=float(request.get("temperature", 0.7))
        )
        
        return {
            "success": True,
            "response": result["response"],
            "tokens": result["tokens_generated"],
            "time": f"{result['time_elapsed']:.2f}s"
        }
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {"error": str(e), "success": False}

@app.post("/api/rag")
async def rag_api(request: dict):
    """API RAG optimizada"""
    try:
        query = request.get("query", "")
        
        if not query:
            return {"error": "Query vacía"}
        
        result = rag.generate_with_context(
            query=query,
            top_k=int(request.get("top_k", 3)),
            max_new_tokens=min(int(request.get("max_tokens", 256)), 2048)
        )
        
        return {
            "success": True,
            "response": result["response"],
            "sources": result["sources"],
            "context": result["context_used"],
            "time": f"{result['time_elapsed']:.2f}s"
        }
    
    except Exception as e:
        logger.error(f"RAG error: {e}")
        return {"error": str(e), "success": False}

@app.post("/api/documents")
async def add_document_api(request: dict):
    """Agregar documento a RAG"""
    try:
        doc_id = request.get("doc_id", "")
        content = request.get("content", "")
        
        if not doc_id or not content:
            return {"error": "doc_id o content vacío"}
        
        rag.add_document(doc_id, content)
        
        return {
            "success": True,
            "message": f"Documento {doc_id} agregado"
        }
    
    except Exception as e:
        logger.error(f"Document error: {e}")
        return {"error": str(e), "success": False}

@app.get("/api/documents")
async def list_documents_api():
    """Listar documentos"""
    try:
        docs = rag.list_documents()
        stats = rag.get_statistics()
        
        return {
            "success": True,
            "documents": docs,
            "statistics": stats
        }
    
    except Exception as e:
        logger.error(f"List documents error: {e}")
        return {"error": str(e), "success": False}

@app.delete("/api/documents/{doc_id}")
async def delete_document_api(doc_id: str):
    """Eliminar documento"""
    try:
        success = rag.remove_document(doc_id)
        
        return {
            "success": success,
            "message": f"Documento {doc_id} eliminado" if success else "Documento no encontrado"
        }
    
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return {"error": str(e), "success": False}

# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup():
    logger.info("="*60)
    logger.info("API Web iniciada")
    logger.info("Accede a http://localhost:8000")
    logger.info("="*60)

@app.on_event("shutdown")
async def shutdown():
    logger.info("API cerrando...")
    engine.clear_cache()

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Servidor en {host}:{port}")
    
    uvicorn.run(
        "web_api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
