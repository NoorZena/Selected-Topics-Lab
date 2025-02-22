import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load document embeddings (Ensure embeddings.npy exists)
embeddings = np.load("embeddings.npy")

# Load document contents
document_store = {}

with open("documents.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(": ", 1)  # Split ID and text
        if len(parts) == 2:
            doc_id, content = parts
            document_store[doc_id] = content  # Store document content

# Function to retrieve top K most similar documents
def find_similar_documents(query_vector, embeddings, top_k=10):
    scores = cosine_similarity(query_vector.reshape(1, -1), embeddings)[0]
    ranked_indices = scores.argsort()[-top_k:][::-1]  # Get top-K indices
    document_ids = list(document_store.keys())  # Get document IDs
    return [(document_ids[i], scores[i]) for i in ranked_indices]

# Function to transform user query into a vector (Replace with a real NLP model)
def convert_query_to_vector(query):
    return np.random.rand(embeddings.shape[1])  # Placeholder random vector

# Streamlit App UI
st.title("📄 Intelligent Document Search")
st.subheader("Find relevant Reuters news articles using Word Embeddings")

# Input field for user query
query = st.text_input("Enter your search term:")

if st.button("Search"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid query!")
    else:
        query_vector = convert_query_to_vector(query)
        st.write(f"🔍 Searching across {len(document_store)} documents...")
        st.write(f"Embedding size: {embeddings.shape}")

        # Retrieve and display results
        results = find_similar_documents(query_vector, embeddings)
        
        st.write("### 🔥 Most Relevant Documents:")
        for doc_id, score in results:
            content = document_store.get(doc_id, "⚠️ No content available")
            st.write(f"📄 **{doc_id}** (Score: {score:.4f})")
            st.write(f"{content[:500]}...")  # Show first 500 characters
            st.write("---")
