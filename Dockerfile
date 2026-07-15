FROM python:3.12-slim
WORKDIR /app
COPY render_requirements.txt /app
RUN pip install --no-cache-dir -r render_requirements.txt
COPY . /app
CMD uvicorn main:app --host 0.0.0.0 --port $PORT