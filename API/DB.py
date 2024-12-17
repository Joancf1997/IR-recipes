import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import ast


load_dotenv()


# Database Configuration
db_config = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"]
}


def test_db_connection(): 
    global db_config

    print("testing DB connection....")
    try:
        conn = psycopg2.connect(**db_config)
        print("Database connection established successfully!")
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


def insert_recipes_to_db(dataframe):
    global db_config

    print("Inserting data into db...")
    # Establish connection
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    # SQL insert query
    insert_query = """
    INSERT INTO recipes (
        id, name, description, ingredients, ingredients_raw_str, 
        serving_size, servings, steps, tags, search_terms, 
        processed_description, embedded_description, cluster_id
    ) VALUES %s
    """
    
    # Prepare data for bulk insert
    records = []
    for _, row in dataframe.iterrows():
        records.append((
            row['id'],
            row['name'],
            row['description'],
            ast.literal_eval(row['ingredients']),
            ast.literal_eval(row['ingredients_raw_str']),
            row['serving_size'],
            row['servings'],
            ast.literal_eval(row['steps']),
            ast.literal_eval(row['tags']),
            row['search_terms'],
            row['processed_description'],
            row['embedded_description'].tolist(),
            row['Cluster']
        ))
    
    # Bulk insert using execute_values for performance
    cursor.execute("DELETE FROM recipes;")
    execute_values(cursor, insert_query, records)

    # Commit transaction and close connection
    conn.commit()
    cursor.close()
    conn.close()
    print("Data successfully inserted into the database!")
