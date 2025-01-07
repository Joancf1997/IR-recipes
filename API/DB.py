import os
import ast
import psycopg2
import numpy as np 
from dotenv import load_dotenv
from psycopg2.extras import execute_values


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
        return True
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        return False
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





"""
    Extract all the embeddings from the current recipes to train the classifier
"""
def get_recipes_embeddings():
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    # Query to retrieve embeddings and cluster_id
    query = "SELECT embedded_description, cluster_id FROM recipes"
    cursor.execute(query)
    rows = cursor.fetchall()

    # Close the connection
    cursor.close()
    conn.close()
    
    # Convert query result to DataFrame
    embeddings = [row[0] for row in rows]  # List of embeddings
    cluster_ids = [row[1] for row in rows]  # List of cluster IDs
    
    # Convert embeddings to a numpy array
    embeddings_array = np.array(embeddings)
    
    # Convert cluster IDs to a numpy array
    cluster_ids_array = np.array(cluster_ids)
    
    return embeddings_array, cluster_ids_array
