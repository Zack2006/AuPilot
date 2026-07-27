FROM python:3.12.13-slim

ARG MODEL_RUNTIME_DIR=model_runtime

WORKDIR /opt/aupilot-models

COPY ${MODEL_RUNTIME_DIR}/requirements.lock /tmp/requirements.model.lock
RUN python -m pip install --no-cache-dir -r /tmp/requirements.model.lock

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ${MODEL_RUNTIME_DIR}/ /opt/aupilot-models/

ENV PYTHONPATH=/opt/aupilot-models/runtime_source
ENV PYTHONDONTWRITEBYTECODE=1

RUN python verify_runtime_package.py .

COPY model_sidecar/app.py /opt/receiver/app.py

CMD ["uvicorn", "app:app", "--app-dir", "/opt/receiver", "--host", "0.0.0.0", "--port", "8100"]
