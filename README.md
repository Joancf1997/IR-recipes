# IR-Recipes: Information Retrieval for Recipe Transcriptions

The **IR-Recipes** project, a **information retrieval** techniques and systems. This repository aims to showcase methodologies for extracting and retrieving relevant information from recipe transcriptions, enabling intuitive and efficient search capabilities.

## 🎯 Project Objectives

- Develop an **information retrieval system** to process and analyze recipe transcriptions.
- Support **natural language queries** for enhanced user experience.
- Leverage techniques like **text preprocessing**, **vectorization**, and **ranking algorithms**.
- Implement evaluation metrics to measure the system's performance.

## 🚀 Features

- **Preprocessing Pipeline**:
  - Tokenization, lemmatization, and stopword removal.
- **Search Engine**:
  - Query matching using vector space models.
  - Ranking results based on relevance.
- **Evaluation Metrics**:
  - Precision, Recall, and F1-Score.
  - Mean Average Precision (MAP).
- **Interactive Interface**:
  - A simple interface for inputting queries and displaying results.

## 🛠️ Technologies

- **Programming Language**: Python
- **Libraries**:
  - `NLTK` for text preprocessing
  - `scikit-learn` for vectorization and metrics
  - `Flask` for a web-based interface (optional)

## 📂 Project Structure

```
IR-recipes/
├── data/                   # Dataset containing recipe transcriptions
├── preprocessing/          # Scripts for cleaning and preparing text data
├── retrieval/              # Core IR system and query processing
├── evaluation/             # Scripts to evaluate the retrieval performance
├── app/                    # Flask app for the user interface
├── notebooks/              # Jupyter notebooks for experiments and analysis
├── requirements.txt        # Dependencies for the project
└── README.md               # Project documentation
```

## 📊 Dataset

The dataset consists of **food of recipes**, including ingredients, instructions, and metadata. Ensure data is formatted as plain text or CSV for compatibility with preprocessing scripts.

### Example Entry:

```
Title: Chocolate Chip Cookies
Ingredients: Flour, Sugar, Butter, Eggs, Chocolate Chips
Instructions: Preheat oven to 350°F. Mix ingredients. Bake for 12-15 minutes.
```

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Joancf1997/IR-recipes.git
   cd IR-recipes
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the preprocessing script:

   ```bash
   python preprocessing/preprocess_data.py
   ```

4. Launch the api system:

   ```bash
   python API.py
   ```

5. (Optional) Start the web interface:

   ```bash
   cd demo
   npm run dev app/app.py
   ```

## 🧪 How to Use

1. Prepare the dataset and place it in the `data/` directory.
2. Run the preprocessing scripts to clean and prepare the data.
3. Start the retrieval system and input natural language queries.
4. (Optional) Use the Flask app for a user-friendly interface.


Metrics calculated include:
- **Precision**
- **Recall**
- **F1-Score**
- **Mean Average Precision (MAP)**

## 📚 References

This project leverages concepts and techniques from:
- "Introduction to Information Retrieval" by Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze
- Relevant research papers and online resources on modern IR systems.


