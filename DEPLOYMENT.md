# Deployment Guide

This project includes a Streamlit app located at `aeroreach/ui/app.py`.

## Prepare the repository
- Ensure `.venv` is not committed. The provided `.gitignore` already excludes `.venv` and common caches.

## Local development (recommended)
1. Create a virtual environment and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. Run the Streamlit app:

```powershell
.venv\Scripts\streamlit.exe run .\aeroreach\ui\app.py
```

## Docker deployment (recommended for production)
1. Build the Docker image:

```powershell
docker build -t aeroreach:latest .
```

2. Run the container:

```powershell
docker run -p 8501:8501 aeroreach:latest
```

Or use docker-compose for convenience:

```powershell
docker-compose up --build
```

Your app will be available at `http://localhost:8501`.

## Notes and best practices
- Do not commit large data files (e.g., `AeroReach Insights.csv`) into the repository; instead use an environment-specific data volume or external data store for production.
- For Streamlit Cloud, you can connect the repository and use the `requirements.txt` provided.
- For more advanced deployments consider using a multi-stage Dockerfile and pinning dependency versions in `requirements.txt`.
