from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from representation import preprocess_dataset
# from DB import store_embeddings_to_postgresql
from Clusters import visualize_verctor_docs
import os

# Sample dataset
recipes = [
    "Spicy Indian Chicken: Mix spices and chicken, then grill.",
    "Spicy Indian Chicken massala: Mix spices and chicken, then grilled.",
    "Vegan Salad: Combine lettuce, tomatoes, and olive oil.",
    "Green Salad: Combine lettuce, tomatoes, and olive oil and salt.",
    "Seafood Pasta: Cook pasta and add shrimp with garlic sauce.",
    "shrimp Pasta: Cook pasta and add shrimp with tomatoe sauce."
]

# Load the dataset
df = pd.read_csv('../Data/recipes_w_data.csv', usecols=['steps'])
descriptions = df['steps'].dropna().to_numpy()
recipes = descriptions[:200]


# Create the initial embeding of the "catalog recipes"
def initial_recipes_embeding(recipes): 
    # (Clean and tokenize) - Preprocess the recipes 
    preprocessed_recipes = preprocess_dataset(recipes)
    print(preprocessed_recipes[0])

    # (Generate embeddings) - Load a pre-trained SentenceTransformer model 
    model = SentenceTransformer("all-MiniLM-L6-v2")  
    recipe_embeddings = model.encode(preprocessed_recipes)

    # (Visualize the documents) in the embeded space
    visualize_verctor_docs(recipes, recipe_embeddings)

    # (Store embeddings) - PostgreSQL
    # store_embeddings_to_postgresql(preprocessed_recipes, recipe_embeddings)


# Load test data
initial_recipes_embeding(recipes)