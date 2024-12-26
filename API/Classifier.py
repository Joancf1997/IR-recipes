import os 
from dotenv import load_dotenv
import psycopg2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

db_config = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"]
}
classification_model = None


# Database connection
def get_data_from_db():
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


def train_classifier():
    global classification_model
    print("Training the classifier")
    # Fetch data from DB
    X, y = get_data_from_db()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize a classifier (e.g., RandomForestClassifier)
    classification_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    classification_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classification_model.predict(X_test)

    # Evaluate the model
    print(classification_report(y_test, y_pred))


def classify_recipe(embeded_recipe):
    global classification_model
    print("Classifying data...")
    # Do the classification
    return classification_model.predict(embeded_recipe)

    

