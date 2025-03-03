import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the dataset
file_path = "5d_nic.csv"
df = pd.read_csv(file_path)

# Extract NIC numbers and descriptions
nics = df.iloc[:, 0].astype(str).tolist()
descriptions = df.iloc[:, 1].astype(str).tolist()

# Load pre-trained Sentence-BERT model
print("Loading Sentence-BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for descriptions
print("Generating embeddings...")
embeddings = model.encode(descriptions, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Define API endpoint for search
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, 5)
    
    results = [
        {"rank": i+1, "NIC": nics[idx], "Description": descriptions[idx], "Score": float(distances[0][i])}
        for i, idx in enumerate(indices[0])
    ]

    return jsonify({"query": query, "results": results})

port = int(os.environ.get('PORT', 10000))  # Use Render-assigned port
app.run(host='0.0.0.0', port=port)
