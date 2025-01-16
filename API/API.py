from flask_cors import CORS
from DB import test_db_connection
from flask import Flask, request, jsonify
from IR import genesis, classify_document, get_top_documents, listRecipes
import time
from functools import wraps

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

# Utility function to measure execution time
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        response = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Endpoint {func.__name__} executed in {execution_time:.4f} seconds")
        return response
    return wrapper

# List the top 40 recipes of the database 
@app.route('/recipesList', methods=['GET'])
@measure_time
def recipes_list():
    recipes_list = listRecipes()
    return jsonify({"recipes": recipes_list}), 200

# Generate the setup for all existing recipes
@app.route("/setup", methods=["POST"])
@measure_time
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

# Classify the provided recipe 
@app.route("/classify", methods=["POST"])
@measure_time
def classify():
    data = request.get_json()
    recipe = data.get("recipe", "")

    if not recipe:
        return jsonify({"error": "No recipe provided"}), 400

    response, recipes = classify_document(recipe)
    return jsonify({"class": str(response), "recipes": recipes}), 200

# Return the top N recipes more related to the query
@app.route("/query", methods=["POST"])
@measure_time
def query():
    data = request.get_json()
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    response = get_top_documents(user_query, 5)
    return jsonify(response), 200

if __name__ == "__main__":
    app.run(debug=True)
