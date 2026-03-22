FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for PyMuPDF and psycopg2
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the API code
COPY . .

# Temporarily keep the container alive since we haven't built main.py yet!
CMD ["tail", "-f", "/dev/null"]
