vector_db - Complete folder of vector database

Number of documents: 12,124 text blocks 
Vector Quantity: 12,124 embedding vectors
Embedded Model: all-mpnet-base-v2 (768 dimensions) 
Metadata fields: source(the title of the book), chapter
Database Type: ChromaDB (Open-source Vector Database)
Environment requirements: pip install chromadb sentence-transformers pandas tqdm

In the "vector_db" folder: 
chroma.sqlite3 - Stores document content, metadata and IDs 
.bin file - Stores vector index data 
These documents collectively represent 12,124 documents + 12,124 vectors

Complete "vector_db" folder containing:
- `chroma.sqlite3` - Main database file (documents, metadata, IDs)
- `data_level0.bin` - Actual vector data (embedding vectors)
- `header.bin` - Index metadata and configuration  
- `index_metadata.pickle` - Additional index metadata
- `length.bin` - Vector length information (for normalization)
- `link_lists.bin` - Graph index connection information

Connection code example:
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection(name="docs")