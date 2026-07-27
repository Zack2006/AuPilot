FROM python:3.12.13-slim
WORKDIR /app
ENV PYTHONPATH=/app

# Install presentation-layer dependencies only.
COPY requirements-frontend.txt .
RUN pip install --no-cache-dir -r requirements-frontend.txt

# All data access goes through the backend API.
COPY frontend frontend
COPY .streamlit .streamlit

CMD ["streamlit", "run", "frontend/Home.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
