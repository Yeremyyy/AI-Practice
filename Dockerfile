# Multi-stage build para optimización
FROM pytorch/pytorch:2.1.0-cuda12.1-runtime-ubuntu22.04 as base

# Metadata
LABEL maintainer="AI Practice"
LABEL description="Motor IA Production-Ready con arquitectura neural"

# Set working directory
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage final
FROM base as production

WORKDIR /app

# Copiar código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/logs /app/data /app/models

# Permisos
RUN chmod +x api.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose puerto
EXPOSE 8000

# Comando de inicio
CMD ["python", "api.py"]
