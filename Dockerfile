FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501 8000
CMD ["streamlit", "run", "apps/streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
