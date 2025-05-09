FROM nvidia/cuda:12.2.0-base-ubuntu22.04

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copiar todo el proyecto
COPY . .

# Instalar dependencias Python
# Instalar PyTorch sin soporte CUDA
RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Luego instala el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn


# Variables de entorno para configuraciones
ENV PYTHONPATH=/app
ENV HF_MODEL_PATH=/app/experiments/trained_models
ENV CONFIG_PATH=/app/options/test/HMA_SRx2.yaml
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS="ignore::UserWarning"
ENV CUDA_VISIBLE_DEVICES=-1
ENV HF_HOME=/cache/huggingface

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]