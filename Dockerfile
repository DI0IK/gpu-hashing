FROM python:3.11-slim

RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

# Install pip dependencies (use requirements.txt if present)
# We copy the file alone first to leverage Docker layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app

# Ensure logs are not buffered
ENV PYTHONUNBUFFERED=1

# Expose Flask port
EXPOSE 5000

# Run the proxy
USER app
CMD ["python", "proxy.py"]
