# Use the latest Python image
FROM python:3.10.11

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirement files first to leverage Docker caching
COPY requirements.txt requirements-test.txt requirements-dev.txt ./

# Install only production dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt  

# Include test dependencies
RUN pip install -r requirements-test.txt  

# Copy the rest of the application code
COPY . .

# Expose FastAPI's default port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]
