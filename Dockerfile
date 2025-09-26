# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install system dependencies for common Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application
COPY . /app

# Install the local package into the image so `import aeroreach` resolves
RUN pip install --no-deps -e .

# Expose Streamlit port
EXPOSE 8501

# Streamlit config to run headless
ENV STREAMLIT_SERVER_HEADLESS=true

# Run Streamlit app
CMD ["streamlit", "run", "aeroreach/ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
