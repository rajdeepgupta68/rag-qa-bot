import chromadb
from sentence_transformers import SentenceTransformer

#Config
COLLECTION_NAME = "rag_docs"
TOP_K = 5  

#Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#Connect to ChromaDB 
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(COLLECTION_NAME)

#Retrieve top-K relevant chunks 
def retrieve(query, top_k=TOP_K):
    # Embed the question
    query_embedding = embedder.encode(query).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    return chunks, sources
