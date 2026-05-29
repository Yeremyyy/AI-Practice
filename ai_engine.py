"""
AI Engine - Production-Ready LLM Wrapper
Arquitectura optimizada para deployment con neuronas artificiales
Funciones de activación: Sigmoid, ReLU, GELU, Tanh
Pesos inteligentes: Xavier/Glorot & He initialization
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import json
from pathlib import Path

# Configuración de logging con niveles detallados
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActivationFunction(Enum):
    """Funciones de activación neural disponibles"""
    SIGMOID = "sigmoid"
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SOFTMAX = "softmax"


@dataclass
class ModelConfig:
    """Configuración del modelo - Buenas prácticas para production"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "float16"  # float16 para eficiencia, float32 para precisión máxima
    max_tokens: int = 2048
    temperature: float = 0.7
    quantize: bool = True  # 4-bit quantization para reducir memoria 75%
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    load_in_8bit: bool = False


class NeuralActivations:
    """Implementación de funciones de activación neural con estudios de neuronas"""
    
    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        """
        Sigmoid: σ(x) = 1 / (1 + e^-x)
        Rango: (0, 1) - Usado para probabilidades binarias
        Derivada: σ(x) * (1 - σ(x))
        """
        return torch.sigmoid(x)
    
    @staticmethod
    def sigmoid_derivative(x: torch.Tensor) -> torch.Tensor:
        """Derivada de sigmoid para backpropagation"""
        sig = torch.sigmoid(x)
        return sig * (1 - sig)
    
    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        """
        ReLU: max(0, x)
        Rango: [0, ∞)
        Ventaja: Resuelve problema de vanishing gradients
        Derivada: 1 si x > 0, 0 si x <= 0
        """
        return torch.nn.functional.relu(x)
    
    @staticmethod
    def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
        """
        Leaky ReLU: max(negative_slope * x, x)
        Mejora sobre ReLU: permite gradientes negativos
        """
        return torch.nn.functional.leaky_relu(x, negative_slope)
    
    @staticmethod
    def gelu(x: torch.Tensor) -> torch.Tensor:
        """
        GELU: Gaussian Error Linear Unit
        Fórmula: x * Φ(x) donde Φ es la función de distribución normal
        Usado en: BERT, GPT, Transformers modernos
        Rango: (-∞, ∞)
        """
        return torch.nn.functional.gelu(x)
    
    @staticmethod
    def tanh(x: torch.Tensor) -> torch.Tensor:
        """
        Tanh: (e^x - e^-x) / (e^x + e^-x)
        Rango: (-1, 1)
        Ventaja: Centrado en cero, converge más rápido que sigmoid
        Derivada: 1 - tanh(x)^2
        """
        return torch.tanh(x)
    
    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Softmax: e^x_i / Σ(e^x_j)
        Normaliza outputs a distribución de probabilidad
        Suma total = 1
        Usado en: Capa de salida para clasificación
        """
        return torch.nn.functional.softmax(x, dim=dim)
    
    @staticmethod
    def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        ELU: Exponential Linear Unit
        Mejor que ReLU para backpropagation
        """
        return torch.nn.functional.elu(x, alpha)
    
    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        """
        Swish: x * sigmoid(x)
        Usado en: EfficientNet, modern architectures
        """
        return x * torch.sigmoid(x)


class WeightInitializer:
    """Inicialización inteligente de pesos - Buenas prácticas neural"""
    
    @staticmethod
    def xavier_uniform(tensor: torch.Tensor) -> torch.Tensor:
        """
        Inicialización Xavier/Glorot Uniforme
        Ideal para: Redes profundas con sigmoid/tanh
        Fórmula: U(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
        Mantiene varianza consistente en capas
        """
        torch.nn.init.xavier_uniform_(tensor)
        return tensor
    
    @staticmethod
    def xavier_normal(tensor: torch.Tensor) -> torch.Tensor:
        """Inicialización Xavier/Glorot Normal"""
        torch.nn.init.xavier_normal_(tensor)
        return tensor
    
    @staticmethod
    def he_normal(tensor: torch.Tensor) -> torch.Tensor:
        """
        Inicialización He Normal
        Ideal para: ReLU y variantes
        Fórmula: N(0, √(2/fan_in))
        Previene degradación de gradientes en capas profundas
        """
        torch.nn.init.kaiming_normal_(tensor, mode='fan_out', nonlinearity='relu')
        return tensor
    
    @staticmethod
    def he_uniform(tensor: torch.Tensor) -> torch.Tensor:
        """Inicialización He Uniforme"""
        torch.nn.init.kaiming_uniform_(tensor, mode='fan_out', nonlinearity='relu')
        return tensor
    
    @staticmethod
    def lecun_normal(tensor: torch.Tensor) -> torch.Tensor:
        """
        Inicialización LeCun Normal
        Ideal para: SELU activation
        Fórmula: N(0, √(1/fan_in))
        """
        torch.nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='linear')
        return tensor
    
    @staticmethod
    def orthogonal(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """
        Inicialización Ortogonal
        Ideal para: RNN, LSTM
        Preserva norma de gradientes
        """
        torch.nn.init.orthogonal_(tensor, gain=gain)
        return tensor


class NeuralLayer:
    """Capa neural con pesos optimizados"""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationFunction = ActivationFunction.RELU,
        weight_init_method: str = "he_normal"
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Inicializar pesos
        self.weights = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)
        
        # Aplicar inicialización inteligente
        if weight_init_method == "xavier_uniform":
            WeightInitializer.xavier_uniform(self.weights)
        elif weight_init_method == "he_normal":
            WeightInitializer.he_normal(self.weights)
        elif weight_init_method == "lecun_normal":
            WeightInitializer.lecun_normal(self.weights)
        elif weight_init_method == "orthogonal":
            WeightInitializer.orthogonal(self.weights)
        
        logger.info(f"Layer creada: {input_size} -> {output_size} con {weight_init_method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con función de activación"""
        z = torch.matmul(x, self.weights) + self.bias
        
        if self.activation == ActivationFunction.SIGMOID:
            return NeuralActivations.sigmoid(z)
        elif self.activation == ActivationFunction.RELU:
            return NeuralActivations.relu(z)
        elif self.activation == ActivationFunction.GELU:
            return NeuralActivations.gelu(z)
        elif self.activation == ActivationFunction.TANH:
            return NeuralActivations.tanh(z)
        elif self.activation == ActivationFunction.SOFTMAX:
            return NeuralActivations.softmax(z)
        
        return z


class AIEngine:
    """
    Motor IA principal - Production Ready
    Implementa arquitectura neural con todas las buenas prácticas
    """
    
    def __init__(self, config: ModelConfig = None):
        """Inicializa el motor IA con validación completa"""
        self.config = config or ModelConfig()
        self.device = torch.device(self.config.device)
        self.model = None
        self.tokenizer = None
        
        logger.info("="*60)
        logger.info("Inicializando AIEngine")
        logger.info(f"Device: {self.device}")
        logger.info(f"Modelo: {self.config.model_name}")
        logger.info(f"Precision: {self.config.precision}")
        logger.info(f"Quantization: {self.config.quantize}")
        logger.info("="*60)
        
        self._load_model()
        self._validate_setup()
        logger.info("✓ AIEngine inicializado correctamente")
    
    def _load_model(self) -> None:
        """Carga el modelo LLM con manejo robusto de errores"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import warnings
            warnings.filterwarnings("ignore")
            
            logger.info("Descargando tokenizador...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Configurar tokens especiales
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Descargando modelo...")
            
            # Configuración de quantización 4-bit
            if self.config.quantize:
                try:
                    from bitsandbytes.config import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    logger.info("Usando quantización 4-bit (reduce memoria 75%)")
                except ImportError:
                    logger.warning("bitsandbytes no disponible, usando float16")
                    quantization_config = None
            else:
                quantization_config = None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.config.precision == "float16" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Modo evaluación para inferencia
            self.model.eval()
            
            logger.info("✓ Modelo cargado exitosamente")
            self._log_model_stats()
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise RuntimeError(f"Fallo en carga de modelo: {e}")
    
    def _log_model_stats(self) -> None:
        """Registra estadísticas del modelo"""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Parámetros totales: {total_params:,}")
            logger.info(f"Parámetros entrenables: {trainable_params:,}")
            
            if self.device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"Memoria GPU asignada: {memory_allocated:.2f} GB")
                logger.info(f"Memoria GPU reservada: {memory_reserved:.2f} GB")
        except Exception as e:
            logger.warning(f"No se pudieron obtener estadísticas: {e}")
    
    def _validate_setup(self) -> None:
        """Valida que el setup esté correcto"""
        try:
            assert self.model is not None, "Modelo no fue cargado"
            assert self.tokenizer is not None, "Tokenizador no fue cargado"
            
            # Test de generación rápido
            logger.info("Realizando test de generación...")
            test_prompt = "test"
            inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=1)
            
            logger.info("✓ Test de generación exitoso")
        except Exception as e:
            logger.error(f"❌ Error en validación: {e}")
            raise
    
    @torch.no_grad()  # Sin gradientes para inferencia
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        repetition_penalty: float = None
    ) -> Dict[str, any]:
        """
        Genera texto basado en prompt con control granular
        
        Args:
            prompt: Texto de entrada
            max_new_tokens: Máximo de tokens a generar (1-2048)
            temperature: Control de creatividad (0-2)
            top_p: Nucleus sampling (0-1)
            top_k: Top-k sampling
            repetition_penalty: Penalidad por repetición
            
        Returns:
            Dict con response, tokens_used, tiempo de ejecución
        """
        import time
        
        try:
            # Validación exhaustiva
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt debe ser string no vacío")
            
            if len(prompt.strip()) == 0:
                raise ValueError("Prompt no puede estar vacío")
            
            # Usar valores por defecto de config si no se especifican
            temperature = temperature or self.config.temperature
            top_p = top_p or self.config.top_p
            top_k = top_k or self.config.top_k
            repetition_penalty = repetition_penalty or self.config.repetition_penalty
            
            # Validación de parámetros
            if not 0 <= temperature <= 2:
                raise ValueError("Temperature debe estar entre 0 y 2")
            if not 1 <= max_new_tokens <= 2048:
                raise ValueError("max_new_tokens debe estar entre 1 y 2048")
            if not 0 <= top_p <= 1:
                raise ValueError("top_p debe estar entre 0 y 1")
            
            logger.info(f"Generando desde prompt: '{prompt[:50]}...'")
            
            start_time = time.time()
            
            # Tokenización con manejo de errores
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            input_length = inputs["input_ids"].shape[1]
            
            # Generación con mixed precision para velocidad
            with torch.cuda.amp.autocast() if self.device.type == "cuda" else torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_new_tokens, 2048),
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decodificación segura
            response = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
            
            elapsed_time = time.time() - start_time
            tokens_generated = len(self.tokenizer.encode(response))
            
            logger.info(f"✓ Generación completada en {elapsed_time:.2f}s")
            logger.info(f"  Tokens generados: {tokens_generated}")
            logger.info(f"  Velocidad: {tokens_generated/elapsed_time:.1f} tok/s")
            
            return {
                "response": response,
                "tokens_generated": tokens_generated,
                "time_elapsed": elapsed_time,
                "tokens_per_second": tokens_generated / elapsed_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error en generación: {e}")
            raise RuntimeError(f"Fallo en generación: {e}")
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Obtiene embeddings del texto para búsqueda semántica
        Usado en RAG (Retrieval-Augmented Generation)
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Usar última capa oculta como embedding
            embeddings = outputs.hidden_states[-1].mean(dim=1)
            
            logger.info(f"✓ Embeddings obtenidos: {embeddings.shape}")
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo embeddings: {e}")
            raise
    
    def clear_cache(self) -> None:
        """Limpia la memoria caché GPU"""
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.info("✓ Caché GPU limpiado")
        except Exception as e:
            logger.warning(f"No se pudo limpiar caché: {e}")


# Prueba y ejemplo de uso
if __name__ == "__main__":
    logger.info("Iniciando demo del AIEngine")
    
    config = ModelConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        quantize=True,
        temperature=0.7
    )
    
    engine = AIEngine(config)
    
    # Generación
    result = engine.generate(
        "¿Cómo funciona el aprendizaje profundo en redes neuronales?",
        max_new_tokens=256
    )
    
    print("\n" + "="*60)
    print("RESPUESTA DEL IA:")
    print("="*60)
    print(result["response"])
    print(f"\nTokens: {result['tokens_generated']} | Tiempo: {result['time_elapsed']:.2f}s")
    print("="*60)
