FROM python:3.9-slim

# Метаданные
LABEL maintainer="Саликова Мария"
LABEL description="Ethereum Fraud Detection — ML Project"

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем зависимости первыми (кэш-слой)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код проекта
COPY src/ ./src/
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY models/ ./models/
COPY report/ ./report/
COPY tests/ ./tests/

# Переменные окружения для воспроизводимости
ENV PYTHONHASHSEED=42
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Порт для Jupyter
EXPOSE 8888

# По умолчанию запускаем Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# Устанавливаем порт для FastAPI
EXPOSE 8000

# Запускаем FastAPI через uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]