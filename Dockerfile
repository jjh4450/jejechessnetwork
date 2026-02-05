# NVIDIA PyTorch 공식 이미지 사용 (CUDA 12.1 + cuDNN 8 + PyTorch 2.1)
FROM nvcr.io/nvidia/pytorch:25.12-py3

# Stockfish 설치
RUN apt-get update && \
    apt-get install -y stockfish && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt python-chess tqdm
