FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /hydra
WORKDIR /hydra

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

CMD ["uvicorn", "hydra-api.main:app", "--host", "0.0.0.0", "--port", "8000"]
