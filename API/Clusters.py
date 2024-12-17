import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Function to visualize the documents in the embedded space
def visualize_verctor_docs(recipes, embeddings):
    # Standardize the embeddings to improve clustering performance
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Apply K-Means clustering with 3 clusters
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_embeddings)

    # Perform PCA for dimensionality reduction to 3D
    pca = PCA(n_components=3)
    pca_embeddings = pca.fit_transform(scaled_embeddings)

    # Create a DataFrame for easy visualization
    df = pd.DataFrame(pca_embeddings, columns=['PCA1', 'PCA2', 'PCA3'])
    df['Cluster'] = labels
    df['Recipe'] = recipes

    # Define distinct colors for each cluster
    cluster_colors = ['red', 'blue', 'green']
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


    # Print the first two 'Recipe' for each cluster
    # for cluster_id, group in df.groupby('Cluster'):
    #     print(f"Cluster {cluster_id}:")
    #     print(group['Recipe'].head(5).to_list())  # Convert to a list for cleaner output
    #     print()

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
