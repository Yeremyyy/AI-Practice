"""
Ejemplos de uso del AI Engine
Demostraciones de todas las funcionalidades
"""

import logging
from ai_engine import AIEngine, ModelConfig, NeuralActivations
from rag_system import RAGSystem
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_generation():
    """Ejemplo 1: Generación básica de texto"""
    print("\n" + "="*60)
    print("EJEMPLO 1: Generación Básica")
    print("="*60)
    
    config = ModelConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        quantize=True
    )
    
    engine = AIEngine(config)
    
    prompt = "Explica qué es el aprendizaje profundo en 3 líneas"
    
    result = engine.generate(
        prompt=prompt,
        max_new_tokens=256,
        temperature=0.7
    )
    
    print(f"\nPrompt: {prompt}")
    print(f"\nRespuesta:\n{result['response']}")
    print(f"\nTokens: {result['tokens_generated']} | Tiempo: {result['time_elapsed']:.2f}s")


def example_2_temperature_control():
    """Ejemplo 2: Control de temperatura (creatividad)"""
    print("\n" + "="*60)
    print("EJEMPLO 2: Control de Temperatura")
    print("="*60)
    
    config = ModelConfig(quantize=True)
    engine = AIEngine(config)
    
    prompt = "Cuéntame una historia corta sobre IA"
    
    temperatures = [0.3, 0.7, 1.5]
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        result = engine.generate(
            prompt=prompt,
            max_new_tokens=200,
            temperature=temp
        )
        print(f"Respuesta: {result['response'][:150]}...")


def example_3_embeddings():
    """Ejemplo 3: Obtener embeddings para búsqueda semántica"""
    print("\n" + "="*60)
    print("EJEMPLO 3: Embeddings y Búsqueda Semántica")
    print("="*60)
    
    config = ModelConfig(quantize=True)
    engine = AIEngine(config)
    
    texts = [
        "El gato duerme en la cama",
        "Los perros juegan en el parque",
        "La inteligencia artificial es revolucionaria"
    ]
    
    print("\nGenerando embeddings...")
    embeddings = [engine.get_embeddings(text) for text in texts]
    
    print(f"Dimensión de embeddings: {embeddings[0].shape}")
    
    # Calcular similaridad entre el primer y segundo texto
    from numpy import dot
    from numpy.linalg import norm
    
    sim = dot(embeddings[0][0], embeddings[1][0]) / (norm(embeddings[0][0]) * norm(embeddings[1][0]))
    print(f"\nSimilaridad entre texto 1 y 2: {sim:.3f}")


def example_4_rag_system():
    """Ejemplo 4: Sistema RAG con base de conocimiento"""
    print("\n" + "="*60)
    print("EJEMPLO 4: Sistema RAG")
    print("="*60)
    
    config = ModelConfig(quantize=True)
    engine = AIEngine(config)
    rag = RAGSystem(engine)
    
    # Agregar documentos
    print("\nAgregando documentos a la base de conocimiento...")
    
    rag.add_document(
        "doc_redes_neuronales",
        """
        Las redes neuronales artificiales están inspiradas en las redes neuronales del cerebro biológico.
        Consisten en capas de neuronas conectadas que procesan información.
        Cada conexión tiene un peso que se ajusta durante el entrenamiento.
        """
    )
    
    rag.add_document(
        "doc_backpropagation",
        """
        Backpropagation es el algoritmo de entrenamiento más común para redes neuronales.
        Calcula gradientes de la función de pérdida con respecto a los pesos.
        Usa la regla de la cadena para propagar errores hacia atrás en la red.
        """
    )
    
    rag.add_document(
        "doc_transformers",
        """
        Los Transformers revolucionaron el procesamiento del lenguaje natural.
        Utilizan mecanismo de atención para procesar secuencias en paralelo.
        Modelos como BERT, GPT y Claude se basan en arquitectura Transformer.
        """
    )
    
    # Estadísticas
    stats = rag.get_statistics()
    print(f"\nEstadísticas de la base: {stats}")
    
    # Búsqueda
    print("\n--- Búsqueda de documentos relevantes ---")
    query = "¿Cómo funcionan las redes neuronales?"
    results = rag.retrieve(query, top_k=2)
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['id']} (Similaridad: {doc['similarity']:.3f})")
        print(f"   {doc['preview']}")
    
    # Generación con contexto
    print("\n--- Generación con contexto RAG ---")
    result = rag.generate_with_context(
        query="¿Cuál es la diferencia entre backpropagation y forward pass?",
        top_k=2,
        max_new_tokens=256
    )
    
    print(f"\nRespuesta:\n{result['response']}")
    print(f"Fuentes: {result['sources']}")


def example_5_neural_activations():
    """Ejemplo 5: Demostraciones de funciones de activación"""
    print("\n" + "="*60)
    print("EJEMPLO 5: Funciones de Activación Neural")
    print("="*60)
    
    # Crear tensor de prueba
    x = torch.linspace(-5, 5, 100)
    
    activations = {
        "Sigmoid": NeuralActivations.sigmoid,
        "ReLU": NeuralActivations.relu,
        "Tanh": NeuralActivations.tanh,
        "GELU": NeuralActivations.gelu,
        "Swish": NeuralActivations.swish
    }
    
    print("\nCaracterísticas de funciones de activación:")
    print("-" * 60)
    
    for name, func in activations.items():
        y = func(x)
        
        print(f"\n{name}:")
        print(f"  Mín: {y.min():.3f}, Máx: {y.max():.3f}")
        print(f"  Media: {y.mean():.3f}, Desv. Est: {y.std():.3f}")
        
        # Propiedades específicas
        if name == "Sigmoid":
            print(f"  Rango: (0, 1) - Ideal para probabilidades binarias")
        elif name == "ReLU":
            print(f"  Rango: [0, ∞) - Resuelve vanishing gradient")
        elif name == "Tanh":
            print(f"  Rango: (-1, 1) - Centrado en cero")
        elif name == "GELU":
            print(f"  Rango: (-∞, ∞) - Usado en Transformers modernos")


def example_6_advanced_generation():
    """Ejemplo 6: Generación avanzada con parámetros optimizados"""
    print("\n" + "="*60)
    print("EJEMPLO 6: Generación Avanzada")
    print("="*60)
    
    config = ModelConfig(quantize=True)
    engine = AIEngine(config)
    
    prompts = [
        ("Creativo", "Escribe una poesía corta sobre IA", 0.9, 0.95, 50),
        ("Preciso", "¿Cuál es la capital de Francia?", 0.3, 0.9, 30),
        ("Equilibrado", "Explica la gravedad en términos simples", 0.7, 0.9, 50)
    ]
    
    for mode, prompt, temp, top_p, top_k in prompts:
        print(f"\n--- Modo: {mode} ---")
        print(f"Prompt: {prompt}")
        
        result = engine.generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=temp,
            top_p=top_p,
            top_k=top_k
        )
        
        print(f"Respuesta: {result['response'][:200]}...")
        print(f"Velocidad: {result['tokens_per_second']:.1f} tok/s")


def example_7_batch_processing():
    """Ejemplo 7: Procesamiento por lotes (simulado)"""
    print("\n" + "="*60)
    print("EJEMPLO 7: Procesamiento por Lotes")
    print("="*60)
    
    config = ModelConfig(quantize=True)
    engine = AIEngine(config)
    
    prompts = [
        "¿Qué es machine learning?",
        "¿Qué es deep learning?",
        "¿Qué es IA?",
        "¿Cuál es la diferencia entre ML y DL?"
    ]
    
    print("\nProcesando múltiples prompts...\n")
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Procesando: {prompt}")
        
        result = engine.generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7
        )
        
        results.append({
            "prompt": prompt,
            "response": result["response"],
            "time": result["time_elapsed"]
        })
    
    print("\n" + "-"*60)
    print("RESUMEN DE RESULTADOS:")
    print("-"*60)
    
    total_time = sum(r["time"] for r in results)
    total_tokens = sum(len(r["response"].split()) for r in results)
    
    print(f"Total de solicitudes: {len(results)}")
    print(f"Tiempo total: {total_time:.2f}s")
    print(f"Tokens generados: {total_tokens}")
    print(f"Promedio tiempo por solicitud: {total_time/len(results):.2f}s")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EJEMPLOS DE USO - AI ENGINE")
    print("="*60)
    
    # Seleccionar ejemplos para ejecutar
    examples = {
        1: ("Generación Básica", example_1_basic_generation),
        2: ("Control de Temperatura", example_2_temperature_control),
        3: ("Embeddings", example_3_embeddings),
        4: ("Sistema RAG", example_4_rag_system),
        5: ("Funciones de Activación", example_5_neural_activations),
        6: ("Generación Avanzada", example_6_advanced_generation),
        7: ("Procesamiento por Lotes", example_7_batch_processing),
    }
    
    print("\nEjemplos disponibles:")
    for num, (name, _) in examples.items():
        print(f"  {num}. {name}")
    
    # Ejecutar todos los ejemplos
    print("\nEjecutando todos los ejemplos...\n")
    
    for num, (name, func) in sorted(examples.items()):
        try:
            func()
        except Exception as e:
            logger.error(f"Error en ejemplo {num}: {e}")
    
    print("\n" + "="*60)
    print("✓ Demostración completada")
    print("="*60)
