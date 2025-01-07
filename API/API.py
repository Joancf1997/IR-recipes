from DB import test_db_connection
from flask import Flask, request, jsonify
from IR import genesis, classify_document, get_top_documents



app = Flask(__name__)


# List the top 40 recipes of the database 



# Generate the setup for all existing recipes
@app.route("/setup", methods=["POST"])
def setup():
    data = request.get_json()
    data_path = '../Data/recipes_w_data.csv'
    n_rows = data.get("n_rows", 1000)
    n_clusters = data.get("n_clusters", 10)
    visualize = data.get("visualize_clusters", False)

    if test_db_connection():
        genesis(data_path, n_rows, n_clusters, visualize)
        return jsonify({"msg": "SetUp done correctly"}), 200
    else:
        return jsonify({"error": "Database connection failed"}), 500
    

#  Classify the provided recipe 
@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    recipe = data.get("recipe", "")

    if not recipe:
        return jsonify({"error": "No recipe provided"}), 400

    response = classify_document(recipe)
    return jsonify(response), 200


# Return the top N recipes more related to the query
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    response = get_top_documents(user_query, 5)
    return jsonify(response), 200

if __name__ == "__main__":
    app.run(debug=True)