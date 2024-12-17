import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from DB import insert_recipes_to_db



# Function to visualize the documents in the embedded space
def clusterize_embeddings(dataframe, num_clusters):
    print("Clusterizing documents...")

#  ===================== Clusterization part =====================

    # Standardize the embeddings to improve clustering performance
    embeddings = np.array(dataframe["embedded_description"].tolist())
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Apply K-Means clustering with n clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_embeddings)
    dataframe['Cluster'] = labels

    # (Store dataset and embeddings) - PostgreSQL
    insert_recipes_to_db(dataframe)


#  ===================== Visualization part =====================
    # Perform PCA for dimensionality reduction to 3D
    pca = PCA(n_components=3)
    pca_embeddings = pca.fit_transform(scaled_embeddings)

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
        ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'], cluster_data['PCA3'],
                   color=cluster_colors[cluster_id], label=f'Cluster {cluster_id}')

    # Add labels and title
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('Cluster Analysis of Recipes in 3D Space')

    # Add a legend
    ax.legend()
    # Show the plot
    plt.show()


    return dataframe
    