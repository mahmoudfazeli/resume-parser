# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /resume-parser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*  # Correct chaining of commands

# Copy the current directory contents into the container at /resume-parser
COPY . /resume-parser

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
