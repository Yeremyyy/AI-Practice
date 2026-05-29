"""
Sistema RAG (Retrieval-Augmented Generation)
Mejora precisión del IA incorporando conocimiento externo
Usa embeddings semánticos para búsqueda eficiente
"""

import numpy as np
import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Sistema de Retrieval-Augmented Generation
    Combina búsqueda de documentos con generación de texto
    """
    
    def __init__(self, ai_engine, embedding_dim: int = 4096):
        """
        Inicializa RAG System
        
        Args:
            ai_engine: Instancia del AIEngine
            embedding_dim: Dimensión de los embeddings
        """
        self.ai = ai_engine
        self.knowledge_base: List[Dict] = []
        self.embedding_dim = embedding_dim
        
        logger.info("RAGSystem inicializado")
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict = None
    ) -> None:
        """
        Añade documento a la base de conocimiento
        
        Args:
            doc_id: ID único del documento
            content: Contenido del documento
            metadata: Metadatos adicionales (autor, fecha, etc)
        """
        try:
            # Validación
            if not doc_id or not content:
                raise ValueError("doc_id y content no pueden estar vacíos")
            
            # Evitar duplicados
            if any(doc["id"] == doc_id for doc in self.knowledge_base):
                logger.warning(f"Documento {doc_id} ya existe, actualizando...")
                self.knowledge_base = [d for d in self.knowledge_base if d["id"] != doc_id]
            
            logger.info(f"Generando embeddings para documento {doc_id}...")
            
            # Generar embeddings
            embeddings = self.ai.get_embeddings(content)
            
            # Normalizar embeddings para búsqueda coseno
            embeddings_norm = embeddings / np.linalg.norm(embeddings)
            
            # Crear documento
            doc = {
                "id": doc_id,
                "content": content,
                "embeddings": embeddings_norm,
                "metadata": metadata or {},
                "tokens": len(content.split()),
                "content_preview": content[:100] + "..." if len(content) > 100 else content
            }
            
            self.knowledge_base.append(doc)
            logger.info(f"✓ Documento {doc_id} añadido ({len(content.split())} palabras)")
            
        except Exception as e:
            logger.error(f"❌ Error añadiendo documento: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        Recupera documentos relevantes basado en similaridad semántica
        
        Args:
            query: Consulta del usuario
            top_k: Número de documentos a retornar
            threshold: Umbral mínimo de similaridad
            
        Returns:
            Lista de documentos relevantes con scores
        """
        try:
            if not self.knowledge_base:
                logger.warning("Base de conocimiento vacía")
                return []
            
            logger.info(f"Buscando documentos relevantes para: '{query[:50]}...'")
            
            # Generar embeddings de la query
            query_embeddings = self.ai.get_embeddings(query)
            query_embeddings_norm = query_embeddings / np.linalg.norm(query_embeddings)
            
            # Calcular similaridad coseno
            similarities = []
            for doc in self.knowledge_base:
                # Similaridad coseno = dot(a, b) / (norm(a) * norm(b))
                # Ya normalizamos, así que es solo el dot product
                similarity = np.dot(query_embeddings_norm[0], doc["embeddings"][0])
                similarities.append((doc, similarity))
            
            # Ordenar por similaridad descendente
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Filtrar por threshold y tomar top_k
            results = [
                {
                    "id": doc["id"],
                    "content": doc["content"],
                    "similarity": float(similarity),
                    "metadata": doc["metadata"],
                    "preview": doc["content_preview"]
                }
                for doc, similarity in similarities[:top_k]
                if similarity >= threshold
            ]
            
            logger.info(f"✓ {len(results)} documentos relevantes encontrados")
            for i, r in enumerate(results, 1):
                logger.info(f"  {i}. {r['id']} (similaridad: {r['similarity']:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en retrieval: {e}")
            return []
    
    def generate_with_context(
        self,
        query: str,
        top_k: int = 3,
        max_new_tokens: int = 512,
        **kwargs
    ) -> Dict:
        """
        Genera respuesta usando contexto de documentos relevantes
        
        Args:
            query: Pregunta del usuario
            top_k: Documentos a usar como contexto
            max_new_tokens: Tokens a generar
            **kwargs: Argumentos adicionales para generation
            
        Returns:
            Dict con respuesta, contexto usado, y fuentes
        """
        try:
            logger.info("Iniciando generación con contexto RAG...")
            
            # Recuperar documentos relevantes
            relevant_docs = self.retrieve(query, top_k=top_k)
            
            if not relevant_docs:
                logger.warning("No se encontraron documentos relevantes, generando sin contexto")
                context = ""
            else:
                # Crear prompt con contexto
                context = "\n\n".join([
                    f"[Fuente: {doc['id']}]\n{doc['content'][:500]}"
                    for doc in relevant_docs
                ])
            
            # Crear prompt aumentado
            if context:
                augmented_prompt = f"""Basándose en el siguiente contexto, responda la pregunta:

CONTEXTO:
{context}

PREGUNTA: {query}

RESPUESTA:"""
            else:
                augmented_prompt = f"PREGUNTA: {query}\n\nRESPUESTA:"
            
            logger.info(f"Prompt aumentado creado ({len(augmented_prompt)} caracteres)")
            
            # Generar respuesta
            result = self.ai.generate(
                augmented_prompt,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            
            return {
                "response": result["response"],
                "tokens_generated": result["tokens_generated"],
                "time_elapsed": result["time_elapsed"],
                "context_used": len(relevant_docs),
                "sources": [doc["id"] for doc in relevant_docs],
                "context_documents": relevant_docs
            }
            
        except Exception as e:
            logger.error(f"❌ Error en generación con contexto: {e}")
            raise
    
    def save_knowledge_base(self, filepath: str) -> None:
        """Guarda la base de conocimiento a disco"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Convertir embeddings a lista para serialización JSON
            serializable_kb = []
            for doc in self.knowledge_base:
                doc_copy = doc.copy()
                doc_copy["embeddings"] = doc["embeddings"].tolist()
                serializable_kb.append(doc_copy)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_kb, f)
            
            logger.info(f"✓ Base de conocimiento guardada en {filepath}")
        except Exception as e:
            logger.error(f"❌ Error guardando base: {e}")
    
    def load_knowledge_base(self, filepath: str) -> None:
        """Carga base de conocimiento desde disco"""
        try:
            with open(filepath, 'r') as f:
                serializable_kb = json.load(f)
            
            # Convertir embeddings de vuelta a numpy arrays
            self.knowledge_base = []
            for doc in serializable_kb:
                doc["embeddings"] = np.array(doc["embeddings"])
                self.knowledge_base.append(doc)
            
            logger.info(f"✓ Base de conocimiento cargada ({len(self.knowledge_base)} documentos)")
        except Exception as e:
            logger.error(f"❌ Error cargando base: {e}")
    
    def list_documents(self) -> List[Dict]:
        """Lista todos los documentos en la base"""
        return [
            {
                "id": doc["id"],
                "tokens": doc["tokens"],
                "preview": doc["content_preview"],
                "metadata": doc["metadata"]
            }
            for doc in self.knowledge_base
        ]
    
    def remove_document(self, doc_id: str) -> bool:
        """Elimina un documento de la base"""
        try:
            initial_len = len(self.knowledge_base)
            self.knowledge_base = [d for d in self.knowledge_base if d["id"] != doc_id]
            
            if len(self.knowledge_base) < initial_len:
                logger.info(f"✓ Documento {doc_id} eliminado")
                return True
            else:
                logger.warning(f"Documento {doc_id} no encontrado")
                return False
        except Exception as e:
            logger.error(f"❌ Error eliminando documento: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de la base de conocimiento"""
        if not self.knowledge_base:
            return {"total_documents": 0, "total_tokens": 0}
        
        total_tokens = sum(doc["tokens"] for doc in self.knowledge_base)
        total_chars = sum(len(doc["content"]) for doc in self.knowledge_base)
        
        return {
            "total_documents": len(self.knowledge_base),
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "avg_tokens_per_doc": total_tokens / len(self.knowledge_base),
            "document_ids": [doc["id"] for doc in self.knowledge_base]
        }


# Ejemplo de uso
if __name__ == "__main__":
    from ai_engine import AIEngine, ModelConfig
    
    logger.info("Iniciando demo de RAGSystem")
    
    # Inicializar AIEngine
    config = ModelConfig(quantize=True)
    engine = AIEngine(config)
    
    # Crear RAG System
    rag = RAGSystem(engine)
    
    # Agregar documentos de ejemplo
    rag.add_document(
        "doc1",
        "El aprendizaje profundo es un subcampo del aprendizaje automático que utiliza redes neuronales con múltiples capas."
    )
    rag.add_document(
        "doc2",
        "Las redes neuronales convolucionales (CNN) son especialmente útiles para tareas de visión por computadora como clasificación de imágenes."
    )
    rag.add_document(
        "doc3",
        "Los transformers revolucionaron el procesamiento del lenguaje natural. Utilizan mecanismo de atención para procesar secuencias de manera paralela."
    )
    
    # Mostrar estadísticas
    stats = rag.get_statistics()
    print("\nEstadísticas de la base de conocimiento:")
    print(json.dumps(stats, indent=2))
    
    # Realizar búsqueda y generación con contexto
    print("\n" + "="*60)
    print("GENERACIÓN CON CONTEXTO RAG")
    print("="*60)
    
    result = rag.generate_with_context(
        "¿Cómo funcionan los transformers?",
        top_k=2,
        max_new_tokens=256
    )
    
    print(f"\nRespuesta:\n{result['response']}")
    print(f"\nFuentes utilizadas: {result['sources']}")
    print(f"Documentos de contexto: {result['context_used']}")
    print(f"Tiempo: {result['time_elapsed']:.2f}s")
