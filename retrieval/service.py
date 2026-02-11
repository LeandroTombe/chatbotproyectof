"""
Retrieval service for finding relevant chunks for user queries.
Orchestrates embedding generation and vector search.
"""
from typing import List, Optional, Dict, Any
import logging
import uuid

from domain.models import Query, SearchResult, RetrievalConfig
from embeddings.base import BaseEmbedding
from vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class RetrievalException(Exception):
    """Exception raised for retrieval errors"""
    pass


class RetrievalService:
    """
    Servicio de retrieval que coordina la búsqueda de chunks relevantes.
    Combina generación de embeddings y búsqueda en vector store.
    """
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_provider: BaseEmbedding,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Inicializa el servicio de retrieval.
        
        Args:
            vector_store: Vector store para búsqueda
            embedding_provider: Proveedor de embeddings
            config: Configuración de retrieval. Si es None, usa valores por defecto.
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if vector_store is None:
            raise ValueError("vector_store cannot be None")
        if embedding_provider is None:
            raise ValueError("embedding_provider cannot be None")
        
        # Validar que las dimensiones coincidan
        if vector_store.dimension != embedding_provider.get_dimension():
            raise ValueError(
                f"Dimension mismatch: vector_store={vector_store.dimension}, "
                f"embedding_provider={embedding_provider.get_dimension()}"
            )
        
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.config = config or RetrievalConfig()
        self.config.validate()
        
        logger.info(
            f"RetrievalService initialized with dimension={vector_store.dimension}, "
            f"top_k={self.config.top_k}, min_score={self.config.min_score}"
        )
    
    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        document_id: Optional[str] = None
    ) -> Query:
        """
        Recupera chunks relevantes para una consulta.
        
        Args:
            query_text: Texto de la consulta del usuario
            top_k: Número de resultados (usa config por defecto si None)
            min_score: Score mínimo (usa config por defecto si None)
            document_id: Filtrar por documento específico (opcional)
            
        Returns:
            Objeto Query con resultados
            
        Raises:
            RetrievalException: Si hay error en el retrieval
            ValueError: Si query_text está vacío
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")
        
        query_text = query_text.strip()
        query_id = str(uuid.uuid4())
        
        logger.info(f"Processing query {query_id}: '{query_text[:50]}...'")
        
        try:
            # Generar embedding de la consulta
            query_embedding = self._generate_query_embedding(query_text)
            
            # Buscar en el vector store
            results = self._search_vector_store(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
                document_id=document_id
            )
            
            # Crear objeto Query del dominio
            query = Query(
                id=query_id,
                text=query_text,
                embedding=query_embedding,
                results=results
            )
            
            logger.info(
                f"Query {query_id} completed: {len(results)} results found"
            )
            
            return query
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            raise RetrievalException(error_msg) from e
    
    def retrieve_simple(self, query_text: str) -> List[SearchResult]:
        """
        Versión simplificada de retrieve que retorna solo los resultados.
        Útil para casos donde no se necesita el objeto Query completo.
        
        Args:
            query_text: Texto de la consulta
            
        Returns:
            Lista de SearchResult
            
        Raises:
            RetrievalException: Si hay error en el retrieval
        """
        query = self.retrieve(query_text)
        return query.results
    
    def retrieve_context(
        self,
        query_text: str,
        max_context_length: Optional[int] = None,
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        Recupera chunks relevantes y los concatena en un contexto único.
        Útil para construir el contexto de un prompt de chat.
        
        Args:
            query_text: Texto de la consulta
            max_context_length: Longitud máxima del contexto en caracteres (opcional)
            separator: Separador entre chunks
            
        Returns:
            Contexto concatenado como string
            
        Raises:
            RetrievalException: Si hay error en el retrieval
        """
        results = self.retrieve_simple(query_text)
        
        if not results:
            logger.warning(f"No results found for query: '{query_text[:50]}...'")
            return ""
        
        # Concatenar chunks
        context_parts = []
        total_length = 0
        
        for result in results:
            chunk_text = result.chunk.content
            
            # Verificar límite de longitud si está definido
            if max_context_length is not None:
                if total_length + len(chunk_text) + len(separator) > max_context_length:
                    logger.debug(
                        f"Reached max context length ({max_context_length}), "
                        f"stopping at {len(context_parts)} chunks"
                    )
                    break
            
            context_parts.append(chunk_text)
            total_length += len(chunk_text) + len(separator)
        
        context = separator.join(context_parts)
        
        logger.info(
            f"Built context: {len(context_parts)} chunks, "
            f"{len(context)} characters"
        )
        
        return context
    
    def retrieve_with_metadata(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Recupera chunks con metadata enriquecida.
        
        Args:
            query_text: Texto de la consulta
            top_k: Número de resultados
            min_score: Score mínimo
            
        Returns:
            Lista de diccionarios con chunk content, score, y metadata
            
        Raises:
            RetrievalException: Si hay error en el retrieval
        """
        query = self.retrieve(query_text, top_k=top_k, min_score=min_score)
        
        results_with_metadata = []
        for result in query.results:
            result_dict = {
                "content": result.chunk.content,
                "score": result.score,
                "document_name": result.document_name,
                "document_id": result.chunk.document_id,
                "chunk_id": result.chunk.id,
                "chunk_index": result.chunk.chunk_index,
                "metadata": result.chunk.metadata
            }
            results_with_metadata.append(result_dict)
        
        return results_with_metadata
    
    def _generate_query_embedding(self, query_text: str) -> List[float]:
        """
        Genera el embedding para la consulta.
        
        Args:
            query_text: Texto de la consulta
            
        Returns:
            Vector de embedding
            
        Raises:
            RetrievalException: Si hay error generando el embedding
        """
        try:
            embedding = self.embedding_provider.embed_text(query_text)
            logger.debug(f"Generated query embedding: dimension={len(embedding)}")
            return embedding
        except Exception as e:
            error_msg = f"Error generating query embedding: {str(e)}"
            logger.error(error_msg)
            raise RetrievalException(error_msg) from e
    
    def _search_vector_store(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        document_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Busca en el vector store.
        
        Args:
            query_embedding: Vector de embedding de la consulta
            top_k: Número de resultados
            min_score: Score mínimo
            document_id: Filtrar por documento
            
        Returns:
            Lista de SearchResult
            
        Raises:
            RetrievalException: Si hay error en la búsqueda
        """
        try:
            k = top_k if top_k is not None else self.config.top_k
            min_s = min_score if min_score is not None else self.config.min_score
            
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k,
                min_score=min_s,
                document_id=document_id
            )
            
            logger.debug(
                f"Vector store search: {len(results)} results "
                f"(top_k={k}, min_score={min_s})"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Error searching vector store: {str(e)}"
            logger.error(error_msg)
            raise RetrievalException(error_msg) from e
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del servicio de retrieval.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "dimension": self.vector_store.dimension,
            "total_chunks": self.vector_store.count(),
            "embedding_model": self.embedding_provider.get_model_name(),
            "config": {
                "top_k": self.config.top_k,
                "min_score": self.config.min_score
            }
        }
    
    def check_health(self) -> bool:
        """
        Verifica la salud del servicio.
        
        Returns:
            True si el servicio está operacional
        """
        try:
            # Verificar vector store
            count = self.vector_store.count()
            
            # Intentar generar un embedding de prueba
            test_embedding = self.embedding_provider.embed_text("health check")
            
            # Validar dimensión
            if len(test_embedding) != self.vector_store.dimension:
                logger.error("Dimension mismatch in health check")
                return False
            
            logger.info(
                f"Health check passed: {count} chunks, "
                f"dimension={len(test_embedding)}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


def create_retrieval_service(
    vector_store: BaseVectorStore,
    embedding_provider: BaseEmbedding,
    top_k: int = 5,
    min_score: float = 0.7
) -> RetrievalService:
    """
    Función de conveniencia para crear un servicio de retrieval.
    
    Args:
        vector_store: Vector store a usar
        embedding_provider: Proveedor de embeddings
        top_k: Número de resultados por defecto
        min_score: Score mínimo por defecto
        
    Returns:
        RetrievalService configurado
    """
    config = RetrievalConfig(top_k=top_k, min_score=min_score)
    return RetrievalService(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        config=config
    )
