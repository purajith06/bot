# Use a lightweight Python image
FROM python:3.11.8

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Install FAISS, a dependency
RUN apt-get update && apt-get install -y libopenblas-dev && \
    pip install faiss-cpu

# Expose the necessary ports (if using Gradio for the interface)
EXPOSE 7860

# Set environment variables for Elasticsearch
ENV ELASTICSEARCH_URL=http://localhost:9200

# Run the application
CMD ["python", "Gradio_app.py", "--port", "7860"]

