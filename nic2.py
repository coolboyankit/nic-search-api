import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the dataset
file_path = "5d_nic.csv"  # Ensure this file is in the same directory

df = pd.read_csv(file_path)
# Extract NIC numbers from the first column and descriptions from the second
nics = df.iloc[:, 0].astype(str).tolist()
descriptions = df.iloc[:, 1].astype(str).tolist()

# Load a pre-trained Sentence-BERT model
print("Loading Sentence-BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for descriptions
print("Generating embeddings...")
embeddings = model.encode(descriptions, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Search Function
def search(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    print("\nTop Results:")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. NIC: {nics[idx]}, Description: {descriptions[idx]} (Score: {distances[0][i]})")

# Run Search Loop
while True:
    user_query = input("\nEnter search query (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break
    search(user_query)
