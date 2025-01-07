from IR import genesis, classify_document
from DB import test_db_connection



n_rows = 1100
n_clusters = 10
visualize_clusters = True
data_path = '../Data/recipes_w_data.csv'



# Generate the initial setup
# if(test_db_connection()):
#     genesis(data_path, n_rows, n_clusters, visualize_clusters)


# Classify a recipe
recipe = "Sanwditch with cheese"
classify_document(recipe)
