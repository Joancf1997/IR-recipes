# Use the official PostgreSQL image from Docker Hub
FROM postgres:13

# Set environment variables for the PostgreSQL database
ENV POSTGRES_DB=your_database_name
ENV POSTGRES_USER=your_username
ENV POSTGRES_PASSWORD=your_password

# Initialize pgvector extension
RUN apt-get update && \
    apt-get install -y postgresql-server-dev-13 && \
    git clone https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install

# Expose PostgreSQL port
EXPOSE 5432
