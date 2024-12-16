# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Set environment variables for PostgreSQL connection (can also be passed through Docker Compose)
ENV DB_HOST=postgres
ENV DB_PORT=5432
ENV DB_NAME=your_database_name
ENV DB_USER=your_username
ENV DB_PASSWORD=your_password

# Expose the port the app will run on (if needed for an API or service)
EXPOSE 5000

# Run the Python script when the container starts
CMD ["python", "app.py"]
