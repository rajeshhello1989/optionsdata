# gunicorn_config.py

# Worker processes (usually 2n + 1, where n is number of CPU cores)
# Use 2 or 3 for Render's starter plan
workers = 3

# Timeout for workers (Matplotlib chart generation can sometimes take a moment)
timeout = 120

# Bind to the environment variable PORT provided by Render
bind = "0.0.0.0:8000" # Render sets the port, but this is a common setting

# Log file (optional, but good for debugging)
errorlog = '-' 
accesslog = '-'
