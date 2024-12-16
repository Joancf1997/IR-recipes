# Database Configuration
DB_CONFIG = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"]
}

# PostgreSQL Integration
def store_embeddings_to_postgresql(titles, embeddings):
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Ensure pgvector extension is enabled
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create the recipes table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recipes (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                ingredients TEXT[],
                directions TEXT[],
                link varchar(255) ,
                source varchar(255),
                ner text[],
                site varchar(255),
                embedding VECTOR(384) -- Match embedding size
            );
        """)

        # Prepare data for bulk insertion
        records = [(titles[i], embeddings[i].tolist()) for i in range(len(titles))]
        insert_query = """
            INSERT INTO recipes (title, embedding) 
            VALUES %s;
        """
        execute_values(cursor, insert_query, records)

        # Commit the transaction
        conn.commit()
        print("Embeddings stored successfully in PostgreSQL.")

    except Exception as e:
        print("Error storing embeddings in PostgreSQL:", e)

    finally:
        if conn:
            cursor.close()
            conn.close()
