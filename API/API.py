from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pre_processing import preprocess_dataset
from DB import test_db_connection
from Clusters import clusterize_embeddings
from Classifier import train_classifier, classify_recipe
import os


test_db_connection()
n_rows = 1000
n_clusters = 3
embedding_model = None
data_path = '../Data/recipes_w_data.csv'


# Create the initial embeding of the "catalog recipes"
def initial_recipes_embeding(): 
    global embedding_model

    # Load the dataset
    print("Reading datasource, csv....")
    recipes = pd.read_csv(data_path, nrows=n_rows)

    # (Clean and prepare text) - Preprocess the recipes 
    preprocessed_recipes = preprocess_dataset(recipes)

    # (Generate embeddings) - Load a pre-trained SentenceTransformer model 
    print("Embedding descriptions....")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  
    preprocessed_recipes["embedded_description"] = preprocessed_recipes["processed_description"].apply(
        lambda x: embedding_model.encode(x) if x else None
    )

    # Clusterize the documents using the embedings
    clusterize_embeddings(preprocessed_recipes, n_clusters)

    # Train the classifier based on the obtained clusters
    train_classifier()



# Generate the embedding of the documents 
initial_recipes_embeding()



n_test = 3
def test_classifier():
    global embedding_model
    print("testing the classifier...")

    header = pd.read_csv(data_path, nrows=0).columns  # Read only the header
    test_data = pd.read_csv(data_path, skiprows=range(1, n_rows + 10), nrows=n_test)
    test_data.columns = header 
    print(test_data['description'])

    # (Clean and prepare text) - Preprocess the recipes 
    test_preprocessed_recipes = preprocess_dataset(test_data)

    # Embed the recipes 
    test_preprocessed_recipes["embedded_description"] = test_preprocessed_recipes["processed_description"].apply(
        lambda x: embedding_model.encode(x) if x else None
    )

    # Classify the recipes 
    for index, row in test_preprocessed_recipes.iterrows():
        embedding = np.array(row['embedded_description']).reshape(1, -1)  # Convert to 2D array
        prediction = classify_recipe(embedding)     
        print(f"Row {index}: Classified as Cluster {prediction[0]}")
    

test_classifier()