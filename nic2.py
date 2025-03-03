import os
import pandas as pd
import faiss
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# File path
file_path = "5d_nic.csv"

# Load dataset in chunks (to prevent memory overload)
print("Loading dataset...")
chunksize = 1000  # Load only 1000 rows at a time
reader = pd.read_csv(file_path, usecols=[0, 1], chunksize=chunksize)

nics = []
descriptions = []
for chunk in reader:
    nics.extend(chunk.iloc[:, 0].astype(str).tolist())
    descriptions.extend(chunk.iloc[:, 1].astype(str).tolist())

# Load a lightweight model
print("Loading optimized Sentence-BERT model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v1")

# Generate embeddings (in smaller batches)
print("Generating embeddings...")
embeddings = model.encode(descriptions, convert_to_numpy=True, batch_size=64)

# Optimize FAISS index (Reduces memory usage)
d = embeddings.shape[1]
quantizer = faiss.IndexFlatL2(d)  # Standard index
index = faiss.IndexIVFFlat(quantizer, d, 100)  # Use IVF quantization
index.train(embeddings)
index.add(embeddings)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, 5)

    results = [
        {"rank": i+1, "NIC": nics[idx], "Description": descriptions[idx], "Score": round(float(distances[0][i]), 4)}
        for i, idx in enumerate(indices[0])
    ]

    return jsonify({"query": query, "results": results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render assigns a port
    app.run(host='0.0.0.0', port=port)
