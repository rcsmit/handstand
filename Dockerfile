FROM gcr.io/handstandanalyzer/handstand-base

WORKDIR /app
COPY . .

EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
