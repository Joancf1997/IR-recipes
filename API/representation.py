

# Predefined stop words
stop_words = {"and", "the", "is", "in", "of", "with", "then"}

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
    return [clean_text(recipe) for recipe in dataset]