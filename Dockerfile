FROM python:3.10-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código-fonte
COPY src/ src/
COPY data/ data/

# Porta para serviço de inferência (se aplicável)
EXPOSE 8000

# Comando padrão para execução
CMD ["python", "-m", "src.inference"]