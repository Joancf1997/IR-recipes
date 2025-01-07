import os
import pickle                                                   # Save the models 
import numpy as np                                              # Work with the data
import pandas as pd                                             # Work with the data
import matplotlib.pyplot as plt                                 # Visualize the clusters
from sklearn.cluster import KMeans                              # CLustering algorithm 
from sklearn.decomposition import PCA                           # Reduce the dim to visualize 
from sklearn.preprocessing import StandardScaler                # Scale the embeddings of the documents
from sklearn.metrics import classification_report               # Classification report of the model
from sklearn.ensemble import RandomForestClassifier             # Classification model 
from sklearn.model_selection import train_test_split            # Classification model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from DB import insert_recipes_to_db, get_recipes_embeddings, get_embedded_descriptions




"""
    - Generates the clusterization given the N recipes to work with / Saves the model 
    - Trains the classification algorithm / saves the model 
"""

def genesis(data_path, n_rows, num_clusters, visualize_clusters):
    recipes_emb = recipes_embeding(data_path, n_rows)               
    clusterization_model(recipes_emb, num_clusters)       
    classification_model()                                          
    if(visualize_clusters):
        Cluster_visualization(recipes_emb, num_clusters)
    return True
    


# ===================== Create the initial embeding of the "catalog recipes" =====================
def recipes_embeding(data_path, n_rows): 
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
    embedding_model.save("embedding_model")
    return preprocessed_recipes



#  ===================== Clusterization Model =====================
def clusterization_model(dataframe, num_clusters):
    print("Clusterizing documents...")
    # Standardize the embeddings to improve clustering performance
    embeddings = np.array(dataframe["embedded_description"].tolist())
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Apply K-Means clustering with n clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_embeddings)
    dataframe['Cluster'] = labels

    # Save the clustering model 
    print("Saving models...")
    with open("kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    # (Store dataset and embeddings) - PostgreSQL
    insert_recipes_to_db(dataframe)



#  ===================== Classifier Model =====================
def classification_model():
    print("Training the classifier...")
    # Fetch data from DB
    X, y = get_recipes_embeddings()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize a classifier (e.g., RandomForestClassifier)
    classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
    classification_model.fit(X_train, y_train)
    y_pred = classification_model.predict(X_test)
    with open("classification_model.pkl", "wb") as f:
        pickle.dump(classification_model, f)

    # Evaluate the model
    print(classification_report(y_test, y_pred))




#  ===================== Visualization part =====================
def Cluster_visualization(recipes_emb, num_clusters):
    print("Visualization of the clusters...")
    # Perform PCA for dimensionality reduction to 3D
    embeddings = np.vstack(recipes_emb["embedded_description"].values)
    labels = np.vstack(recipes_emb["Cluster"].values)

    pca = PCA(n_components=3)
    pca_embeddings = pca.fit_transform(embeddings)

    # Create a DataFrame for easy visualization
    df = pd.DataFrame(pca_embeddings, columns=['PCA1', 'PCA2', 'PCA3'])
    df['Cluster'] = labels

    # Define distinct colors for each cluster
    cluster_colors = [
        "red", "blue", "purple", "green", "orange", 
        "yellow", "pink", "brown", "gray", "cyan", 
        "magenta", "lime", "indigo", "teal", "navy"
    ]
    df['Color'] = df['Cluster'].map(lambda x: cluster_colors[x])

    # Plot the results in 3D space
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with specific colors for clusters
    for cluster_id in range(num_clusters):
        cluster_data = df[df['Cluster'] == cluster_id]
        ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'], cluster_data['PCA3'], color=cluster_colors[cluster_id], label=f'Cluster {cluster_id}')

    # Add labels and title
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('Cluster Analysis of Recipes in 3D Space')

    # Add a legend
    ax.legend()
    # Show the plot
    plt.show()



def classify_document(recipe):
    # Load the embedding model 
    print("Classifying recipe...")
    embedding_model = SentenceTransformer("embedding_model")

    # Load the classification model 
    with open("classification_model.pkl", "rb") as f:
        classify_recipe = pickle.load(f)

    # (Clean and prepare text) - Preprocess the recipes 
    test_preprocessed_recipe = clean_text(recipe)

    # Embed the recipes 
    recipe_embedding = embedding_model.encode(test_preprocessed_recipe)
    recipe_embedding = recipe_embedding.reshape(1, -1)   # Reshape the embedding to be a 2D array

    # Classify the recipes 
    prediction = classify_recipe.predict(recipe_embedding)
    print(prediction[0])
    return prediction[0]
    




#  ================================== NLP query extration ==================================
"""
    Retrieve the top N most relevant documents for a given natural language query.
"""
def get_top_documents(user_query, top_n=3):
    print("Extraxting NLP query...")
    # Load the pre-trained embedding model
    embedding_model = SentenceTransformer("embedding_model")
    
    # Encode the user's query to obtain its embedding
    query_embedding = embedding_model.encode(user_query).reshape(1, -1)
    
    # Retrieve document IDs and their embeddings from the database
    documents = get_embedded_descriptions()
    
    # Extract embeddings and document metadata
    doc_ids = []
    doc_names = []
    doc_embeddings = []
    
    for doc in documents:
        doc_ids.append(doc[0])
        doc_names.append(doc[1])
        doc_embeddings.append(np.array(doc[2]))  # Convert stored embedding string to NumPy array
    
    # Calculate cosine similarity between the query embedding and all document embeddings
    similarities = cosine_similarity(query_embedding, np.array(doc_embeddings)).flatten()
    
    # Get the top N most relevant document indices
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Prepare the results
    top_documents = []
    for idx in top_indices:
        top_documents.append({
            "id": doc_ids[idx],
            "name": doc_names[idx],
            "similarity": similarities[idx]
        })
    
    return top_documents





# =================== Documents Pre - Processing ===================
stop_words = {"and", "the", "is", "in", "of", "with", "then"}   # Predefined stop words

# Text cleaning and tokenization
def clean_text(text):
    import string
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = text.split()  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return " ".join(tokens)  # Recombine tokens for embedding

# Preprocess dataset
def preprocess_dataset(dataset):
    print("Pre Processing Data...")
    dataset_clean = dataset.copy()
    dataset_clean = dataset_clean.dropna()
    dataset_clean['processed_description'] = dataset_clean['description'].apply(clean_text)

    return dataset_clean
