FROM python:3.13-slim

ENV PYTHONUNBUFFERED=TRUE

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9696

ENTRYPOINT ["uvicorn","predict:app","--host","0.0.0.0","--port","9696"]