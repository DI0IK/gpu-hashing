FROM python:3.11-slim

RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

# Install pip dependencies (use requirements.txt if present)
# We copy the file alone first to leverage Docker layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install a production WSGI server so we don't run the Flask dev server inside
# the container (which prints the "WARNING: This is a development server" message).
# We install Gunicorn separately to avoid forcing it into requirements.txt.
RUN pip install --no-cache-dir gunicorn

# Copy the rest of the project
COPY . /app

# Ensure logs are not buffered
ENV PYTHONUNBUFFERED=1

# Expose Flask port
EXPOSE 5000

# Run the proxy under Gunicorn (production WSGI server)
USER app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "proxy:app"]
