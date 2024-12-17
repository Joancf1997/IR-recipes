from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pre_processing import preprocess_dataset
from DB import test_db_connection
from Clusters import clusterize_embeddings
import os


test_db_connection()
n_rows = 1000
n_clusters = 10


# Create the initial embeding of the "catalog recipes"
def initial_recipes_embeding(): 
    # Load the dataset
    print("Reading datasource, csv....")
    recipes = pd.read_csv('../Data/recipes_w_data.csv', nrows=n_rows)
    recipes = recipes[:n_rows]

    # (Clean and prepare text) - Preprocess the recipes 
    preprocessed_recipes = preprocess_dataset(recipes)

    # (Generate embeddings) - Load a pre-trained SentenceTransformer model 
    print("Embedding descriptions....")
    model = SentenceTransformer("all-MiniLM-L6-v2")  
    preprocessed_recipes["embedded_description"] = preprocessed_recipes["processed_description"].apply(
        lambda x: model.encode(x) if x else None
    )

    # Clusterize the documents using the embedings
    clusterize_embeddings(preprocessed_recipes, n_clusters)



# Generate the embedding of the documents 
initial_recipes_embeding()



