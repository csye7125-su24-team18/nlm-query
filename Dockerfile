# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Create a script to run both FastAPI and Streamlit
RUN echo '#!/bin/bash\n\
uvicorn backend:app --host 0.0.0.0 --port 8000 &\n\
streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0\n'\
> /app/start.sh

RUN chmod +x /app/start.sh

# Run the script when the container launches
CMD ["/app/start.sh"]
