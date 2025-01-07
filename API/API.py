from IR import genesis, classify_document, get_top_documents
from DB import test_db_connection



n_rows = 1100
n_clusters = 10
visualize_clusters = True
data_path = '../Data/recipes_w_data.csv'



# Generate the initial setup
# if(test_db_connection()):
#     genesis(data_path, n_rows, n_clusters, visualize_clusters)


# Classify a recipe
# recipe = "Sanwditch with cheese"
# classify_document(recipe)




#  NLP Query extraction 
user_query = "Sandwitch with cheese, meat on the oven"
top_documents = get_top_documents(user_query)
print("Top Relevant Documents:")
for doc in top_documents:
    print(f"ID: {doc['id']}, Name: {doc['name']}, Similarity: {doc['similarity']:.4f}")