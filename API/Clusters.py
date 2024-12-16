import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Funciton to visualize the the documents in the embeded space
def visualize_verctor_docs(recipes, embeddings):
    # Standardize the embeddings to improve clustering performance
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    # Apply K-Means clustering (you can adjust the number of clusters based on your data)
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_embeddings)

    # Perform PCA for dimensionality reduction to 3D
    pca = PCA(n_components=3)
    pca_embeddings = pca.fit_transform(scaled_embeddings)

    # Create a DataFrame for easy visualization
    df = pd.DataFrame(pca_embeddings, columns=['PCA1', 'PCA2', 'PCA3'])
    df['Cluster'] = labels
    df['Recipe'] = recipes

    # Plot the results in 3D space
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with colors based on clusters
    scatter = ax.scatter(df['PCA1'], df['PCA2'], df['PCA3'], c=df['Cluster'], cmap='viridis', s=50)

    # Add labels and title
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('Cluster Analysis of Recipes in 3D Space')

    # Add a color bar
    cbar = fig.colorbar(scatter)
    cbar.set_label('Cluster ID')

    # Annotate each point with the recipe name
    # for i in range(len(df)):
    #     ax.text(df['PCA1'][i], df['PCA2'][i], df['PCA3'][i], df['Recipe'][i], size=8)

    # Show the plot
    plt.show()