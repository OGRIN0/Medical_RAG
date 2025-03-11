import os
import uuid
import chromadb
from chromadb.errors import InvalidCollectionException
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name="documents", persist_directory="./chroma_db"):
        """Initialize the vector store with ChromaDB."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2" 
        )
        

        logger.info(f"Creating collection: {collection_name}")
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                get_or_create=True  
            )
            logger.info(f"Collection created or fetched: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name, 
                    embedding_function=self.embedding_function
                )
                logger.info(f"Retrieved existing collection: {collection_name}")
            except Exception as e2:
                logger.error(f"Fatal error: Could not create or get collection: {str(e2)}")
                raise

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Add a document to the vector store.
        Returns the document ID.
        """
        doc_id = str(uuid.uuid4())
        
        if not metadata:
            metadata = {}
        
        chunks = self._chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            chunk_metadata = {**metadata, "doc_id": doc_id, "chunk_index": i}
            
            self.collection.add(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[chunk_metadata]
            )
        
        return doc_id
    
    def _chunk_text(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """
        Split text into smaller chunks for better embedding and retrieval.
        Simple chunking by sentences/paragraphs.
        """
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
        
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += " " + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def query(self, query_text: str, n_results: int = 5) -> Dict:
        """
        Query the vector store and return the most relevant documents.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        return results
    
    def get_all_documents(self) -> List[Dict]:
        """
        Get all documents in the store with their metadata.
        Returns a list of document summaries.
        """
        all_data = self.collection.get()
        
        doc_map = {}
        for i, doc_id in enumerate(all_data["ids"]):
            base_id = all_data["metadatas"][i]["doc_id"]
            if base_id not in doc_map:
                doc_map[base_id] = {
                    "id": base_id,
                    "metadata": {k: v for k, v in all_data["metadatas"][i].items() 
                               if k not in ["doc_id", "chunk_index"]},
                    "chunk_count": 0,
                    "sample_text": all_data["documents"][i][:100] + "..."
                }
            doc_map[base_id]["chunk_count"] += 1
            
        return list(doc_map.values())
    
    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document and all its chunks from the store.
        """
        all_data = self.collection.get()
        
        to_delete = []
        for i, metadata in enumerate(all_data["metadatas"]):
            if metadata.get("doc_id") == doc_id:
                to_delete.append(all_data["ids"][i])
        
        if to_delete:
            self.collection.delete(ids=to_delete)
