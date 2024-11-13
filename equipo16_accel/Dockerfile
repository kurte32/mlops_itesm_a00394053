# Use the official Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install prerequisites
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Create a logs directory
RUN mkdir -p /app/logs

# Copy the requirements.txt into the container
COPY requirements.txt /app/

RUN pip3 install --upgrade pip && \
    pip3 install --pre --no-cache-dir -r requirements.txt


# Copy the rest of the application code into the container
COPY . /app/

# Ensure the logs directory has the appropriate permissions
RUN chmod -R 755 /app/logs

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Define environment variables (if any)
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "refactor.app:app", "--host", "0.0.0.0", "--port", "8000"]
