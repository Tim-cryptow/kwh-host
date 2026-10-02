# A stand-in for vllm/vllm-openai in sandbox tests: same command line, same API, no GPU.
# CI builds it from python:3.12-slim; offline, pass BASE and put the wheels in ./wheels.
ARG BASE=python:3.12-slim
FROM ${BASE}
COPY wheels/ /wheels/
RUN python3 -m venv /opt/venv \
 && (/opt/venv/bin/pip install --no-cache-dir --no-index --find-links /wheels fastapi uvicorn \
     || /opt/venv/bin/pip install --no-cache-dir fastapi uvicorn) \
 && rm -rf /wheels
COPY server.py /app/server.py
ENTRYPOINT ["/opt/venv/bin/python", "/app/server.py"]
