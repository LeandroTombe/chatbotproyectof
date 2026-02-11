"""
ChromaDB implementation of vector store.
Provides persistent storage for embeddings using ChromaDB.
"""
import logging
from turtle import distance
from typing import List, Optional, Dict, Any
import json

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None  # type: ignore

from vectorstore.base import BaseVectorStore, VectorStoreException
from domain.models import Chunk, SearchResult, RetrievalConfig
from core.config import CHROMA_DIR, CHROMA_PERSIST, CHROMA_COLLECTION_NAME

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    Vector store implementation using ChromaDB.
    Provides persistent storage and efficient similarity search.
    """

    def __init__(
        self,
        dimension: int,
        config: Optional[RetrievalConfig] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            dimension: Dimension of vectors to store
            config: Retrieval configuration
            collection_name: Name of the collection (uses default if None)
            persist_directory: Directory for persistence (uses default if None)

        Raises:
            VectorStoreException: If ChromaDB is not available
        """
        if not CHROMADB_AVAILABLE:
            raise VectorStoreException(
                "ChromaDB is not installed. Install with: pip install chromadb"
            )

        super().__init__(dimension, config)

        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or str(CHROMA_DIR)

        try:
            # Initialize ChromaDB client
            self.client = chromadb.Client(  # type: ignore[misc]
                Settings(  # type: ignore[misc]
                    persist_directory=self.persist_directory,
                    is_persistent=CHROMA_PERSIST,
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"dimension": dimension}
            )

            logger.info(
                f"ChromaVectorStore initialized: collection='{self.collection_name}', "
                f"dimension={dimension}, persist_dir='{self.persist_directory}'"
            )

        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def add_chunk(self, chunk: Chunk) -> None:
        """
        Add a chunk with its embedding to ChromaDB.

        Args:
            chunk: Chunk to add (must have embedding)

        Raises:
            VectorStoreException: If adding fails
            ValueError: If chunk doesn't have embedding
        """
        if not chunk.has_embedding:
            raise ValueError(f"Chunk {chunk.id} does not have embedding")

        if chunk.embedding is None:
            raise ValueError(f"Chunk {chunk.id} embedding is None")

        self.validate_embedding(chunk.embedding)

        try:
            # Prepare metadata
            metadata = {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata  # Include custom metadata
            }

            # Add to ChromaDB
            self.collection.add(
                ids=[chunk.id],
                embeddings=[chunk.embedding],
                documents=[chunk.content],
                metadatas=[metadata]
            )

            logger.debug(f"Added chunk {chunk.id} to ChromaDB")

        except Exception as e:
            error_msg = f"Error adding chunk to ChromaDB: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add multiple chunks to ChromaDB in batch.

        Args:
            chunks: List of chunks to add

        Raises:
            VectorStoreException: If adding fails
        """
        if not chunks:
            logger.warning("Empty chunks list provided")
            return

        try:
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for chunk in chunks:
                if not chunk.has_embedding or chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.id} does not have embedding")

                self.validate_embedding(chunk.embedding)

                ids.append(chunk.id)
                embeddings.append(chunk.embedding)
                documents.append(chunk.content)
                metadatas.append({
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata
                })

            # Batch add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Added {len(chunks)} chunks to ChromaDB")

        except Exception as e:
            error_msg = f"Error adding chunks to ChromaDB: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks in ChromaDB.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            where: Optional metadata filter (ChromaDB format)

        Returns:
            List of SearchResult objects

        Raises:
            VectorStoreException: If search fails
        """
        self.validate_embedding(query_embedding)

        k = top_k if top_k is not None else self.config.top_k
        min_s = min_score if min_score is not None else self.config.min_score

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where
            )

            search_results: List[SearchResult] = []

            if results.get("ids") and results["ids"][0]:  # type: ignore[index]
                for i in range(len(results["ids"][0])):  # type: ignore[index]

                    chunk_id = results["ids"][0][i]  # type: ignore[index]

                    content = (
                        results["documents"][0][i]  # type: ignore[index]
                        if results.get("documents") and results["documents"][0]  # type: ignore[index]
                        else ""
                    )

                    metadata = (
                        results["metadatas"][0][i]  # type: ignore[index]
                        if results.get("metadatas") and results["metadatas"][0]  # type: ignore[index]
                        else {}
                    )

                    distance = (
                        results["distances"][0][i]  # type: ignore[index]
                        if results.get("distances") and results["distances"][0]  # type: ignore[index]
                        else 0.0
                    )

                    # Convert distance → similarity score
                    # Assuming cosine metric
                    score = 1.0 - (distance / 2.0)

                    if score < min_s:
                        continue

                    chunk_index_val = metadata.get("chunk_index", 0)
                    chunk_index = (
                        int(chunk_index_val)
                        if isinstance(chunk_index_val, (int, float, str))
                        else 0
                    )

                    chunk = Chunk(
                        id=chunk_id,
                        document_id=str(metadata.get("document_id", "unknown")),
                        content=str(content),
                        chunk_index=chunk_index,
                        embedding=None,
                        metadata={
                            k: v for k, v in metadata.items()
                            if k not in ["document_id", "chunk_index"]
                        }
                    )

                    search_results.append(
                        SearchResult(
                            chunk=chunk,
                            score=score,
                            document_name=str(metadata.get("file_name", "unknown"))
                        )
                    )

            logger.debug(
                f"ChromaDB search: {len(search_results)} results "
                f"(top_k={k}, min_score={min_s}, where={where})"
            )

            return search_results

        except Exception as e:
            error_msg = f"Error searching ChromaDB: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e


    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk by ID from ChromaDB.

        Args:
            chunk_id: ID of chunk to delete

        Returns:
            True if deleted, False if not found

        Raises:
            VectorStoreException: If deletion fails
        """
        try:
            # Check if exists
            result = self.collection.get(ids=[chunk_id])
            if not result['ids']:
                logger.debug(f"Chunk {chunk_id} not found for deletion")
                return False

            # Delete
            self.collection.delete(ids=[chunk_id])
            logger.debug(f"Deleted chunk {chunk_id} from ChromaDB")
            return True

        except Exception as e:
            error_msg = f"Error deleting chunk from ChromaDB: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def delete_chunks_by_document(self, document_id: str) -> int:
        """
        Delete all chunks from a document.

        Args:
            document_id: Document ID

        Returns:
            Number of chunks deleted

        Raises:
            VectorStoreException: If deletion fails
        """
        try:
            # Get all chunks from document
            results = self.collection.get(
                where={"document_id": document_id}
            )

            if not results['ids']:
                logger.debug(f"No chunks found for document {document_id}")
                return 0

            # Delete all found chunks
            self.collection.delete(ids=results['ids'])

            count = len(results['ids'])
            logger.info(f"Deleted {count} chunks from document {document_id}")
            return count

        except Exception as e:
            error_msg = f"Error deleting chunks by document from ChromaDB: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """
        Get a chunk by ID from ChromaDB.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk if found, None otherwise

        Raises:
            VectorStoreException: If retrieval fails
        """
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"]
            )

            if not result['ids']:
                return None

            # Reconstruct Chunk
            metadata = result['metadatas'][0] if result['metadatas'] else {}  # type: ignore[index]
            chunk_index_val = metadata.get("chunk_index", 0)
            chunk_index = int(chunk_index_val) if isinstance(chunk_index_val, (int, float, str)) else 0  # type: ignore[arg-type]
            
            # Get embedding safely
            embedding = None
            if result.get('embeddings') and result['embeddings']:  # type: ignore[index]
                emb_data = result['embeddings'][0]  # type: ignore[index]
                if emb_data:
                    embedding = list(emb_data)
            
            chunk = Chunk(
                id=chunk_id,
                document_id=str(metadata.get("document_id", "unknown")),
                content=str(result['documents'][0]) if result['documents'] else "",  # type: ignore[index]
                chunk_index=chunk_index,
                embedding=embedding,
                metadata={k: v for k, v in metadata.items() 
                        if k not in ["document_id", "chunk_index"]}
            )

            return chunk

        except Exception as e:
            error_msg = f"Error getting chunk from ChromaDB: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def count(self) -> int:
        """
        Get total number of chunks in ChromaDB.

        Returns:
            Number of chunks
        """
        try:
            result = self.collection.count()
            return result
        except Exception as e:
            logger.error(f"Error counting chunks in ChromaDB: {str(e)}")
            return 0

    def clear(self) -> None:
        """
        Delete all chunks from the collection.

        Raises:
            VectorStoreException: If clearing fails
        """
        try:
            # Delete the collection and recreate it
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"dimension": self.dimension}
            )
            logger.info(f"Cleared ChromaDB collection '{self.collection_name}'")

        except Exception as e:
            error_msg = f"Error clearing ChromaDB: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e

    def persist(self) -> None:
        """
        Explicitly persist data to disk.
        Note: ChromaDB with newer versions auto-persists, this is a no-op.
        """
        # ChromaDB 0.4+ auto-persists, no manual persist needed
        logger.debug("ChromaDB auto-persists data (no manual action needed)")
