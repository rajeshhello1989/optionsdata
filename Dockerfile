# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port (Optional, but good practice)
EXPOSE 5000

# Command to run the application using Gunicorn, binding to the host's required port ($PORT)
CMD gunicorn --bind 0.0.0.0:$PORT app:app
