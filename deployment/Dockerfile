# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy only essential application files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .  
COPY utils ./utils  

# Expose the Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
