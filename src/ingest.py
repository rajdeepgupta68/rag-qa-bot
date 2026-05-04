import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

#Config 
DATA_FOLDER = "data"
COLLECTION_NAME = "rag_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

#Load embedding model 
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#Connect to ChromaDB 
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(COLLECTION_NAME)

#Extract text 
def extract_text_from_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

#Recursive chunking 
def recursive_chunk(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    #Separators in order of preference
    separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
    
    def split(text, separators):
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        
        #Try each separator
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current = ""
                
                for part in parts:
                    candidate = current + sep + part if current else part
                    
                    if len(candidate) <= chunk_size:
                        current = candidate
                    else:
                        if current.strip():
                            chunks.append(current.strip())
                        # If single part is too large, recurse with next separator
                        if len(part) > chunk_size:
                            remaining_seps = separators[separators.index(sep)+1:]
                            if remaining_seps:
                                chunks.extend(split(part, remaining_seps))
                            else:
                                # Hard split as last resort
                                for i in range(0, len(part), chunk_size - overlap):
                                    chunks.append(part[i:i+chunk_size].strip())
                        else:
                            current = part
                
                if current.strip():
                    chunks.append(current.strip())
                
                return [c for c in chunks if c]
        
        #Fallback: hard split
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i+chunk_size].strip())
        return chunks
    
    raw_chunks = split(text, separators)
    
    #Add overlap between chunks
    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0:
            
            prev_end = raw_chunks[i-1][-overlap:]
            chunk = prev_end + " " + chunk
        overlapped.append(chunk[:chunk_size])
    
    return overlapped


#Main ingestion 
def ingest_documents():
    files = os.listdir(DATA_FOLDER)
    if not files:
        print("No files found in data/ folder!")
        return

    for filename in files:
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f"Processing: {filename}")

        # Skip already ingested
        existing_ids = collection.get()["ids"]
        if f"{filename}_chunk_0" in existing_ids:
            print(f"  -> Already ingested, skipping {filename}")
            continue

        # Extract text
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        elif filename.endswith(".txt"):
            text = extract_text_from_txt(filepath)
        else:
            print(f"  -> Skipping unsupported file: {filename}")
            continue

        # Chunk it
        chunks = recursive_chunk(text)
        print(f"  -> {len(chunks)} chunks created")

        # Embed and store
        for i, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{filename}_chunk_{i}"],
                metadatas=[{"source": filename, "chunk_index": i}]
            )

        print(f"  -> Stored in ChromaDB")

    print(f"\nIngestion complete! Total chunks: {collection.count()}")


if __name__ == "__main__":
    ingest_documents()