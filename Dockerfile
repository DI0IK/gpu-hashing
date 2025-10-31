# Use an official lightweight Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This is done in a separate layer to leverage Docker's build cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY proxy.py .

# Create the /data directory and set permissions
# This ensures the app user can write to the volume
RUN mkdir /data && chown -R appuser:appuser /app /data

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Declare the volume for persistent data
VOLUME /data

# The command to run the application
# We run the script directly, as it manages its own threads and Flask server
CMD ["python", "proxy.py"]
