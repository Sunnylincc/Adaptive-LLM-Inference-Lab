FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
CMD ["python", "scripts/run_policy_eval.py", "--config", "configs/default.yaml", "--output-prefix", "results/docker_eval"]
