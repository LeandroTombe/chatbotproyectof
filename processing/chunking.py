"""
Chunking module for splitting documents into smaller pieces.
Handles text segmentation with overlap for better context preservation.
"""
from typing import List, Optional
import logging
import uuid

from domain.models import Chunk, ChunkingConfig

logger = logging.getLogger(__name__)


class ChunkingException(Exception):
    """Exception raised for chunking errors"""
    pass


class TextChunker:
    """
    Divide texto largo en chunks más pequeños con overlap.
    Preserva el contexto mediante solapamiento entre chunks.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Inicializa el chunker con configuración.
        
        Args:
            config: Configuración de chunking. Si es None, usa valores por defecto.
        """
        self.config = config or ChunkingConfig()
        self.config.validate()
        logger.info(
            f"TextChunker initialized with chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap}"
        )
    
    def chunk_text(
        self, 
        text: str, 
        document_id: str,
        metadata: Optional[dict] = None
    ) -> List[Chunk]:
        """
        Divide un texto en chunks con overlap.
        
        Args:
            text: Texto a dividir en chunks
            document_id: ID del documento origen
            metadata: Metadatos adicionales para cada chunk
            
        Returns:
            Lista de objetos Chunk
            
        Raises:
            ChunkingException: Si hay error en el procesamiento
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for document {document_id}")
            return []
        
        try:
            # Intentar dividir por el separador primero
            chunks = self._split_by_separator(text)
            
            if not chunks:
                logger.info("Separator-based splitting failed, using character-based")
                chunks = self._split_by_characters(text)
            
            # Crear objetos Chunk del dominio
            chunk_objects = []
            for idx, chunk_content in enumerate(chunks):
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    "chunk_method": "separator" if self.config.separator in text else "character",
                    "original_length": len(text)
                })
                
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    content=chunk_content.strip(),
                    chunk_index=idx,
                    metadata=chunk_metadata
                )
                chunk_objects.append(chunk)
            
            logger.info(
                f"Created {len(chunk_objects)} chunks for document {document_id}"
            )
            return chunk_objects
            
        except Exception as e:
            error_msg = f"Error chunking text for document {document_id}: {str(e)}"
            logger.error(error_msg)
            raise ChunkingException(error_msg) from e
    
    def _split_by_separator(self, text: str) -> List[str]:
        """
        Divide el texto usando el separador configurado.
        
        Args:
            text: Texto a dividir
            
        Returns:
            Lista de chunks como strings
        """
        # Dividir por el separador
        segments = text.split(self.config.separator)
        
        chunks = []
        current_chunk = ""
        
        for segment in segments:
            # Si agregar el segmento no excede el tamaño, agregarlo
            if len(current_chunk) + len(segment) + len(self.config.separator) <= self.config.chunk_size:
                if current_chunk:
                    current_chunk += self.config.separator + segment
                else:
                    current_chunk = segment
            else:
                # Guardar el chunk actual si no está vacío
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Si el segmento mismo es más grande que chunk_size, dividirlo
                if len(segment) > self.config.chunk_size:
                    sub_chunks = self._split_large_segment(segment)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = segment
        
        # Agregar el último chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Aplicar overlap
        return self._apply_overlap(chunks)
    
    def _split_by_characters(self, text: str) -> List[str]:
        """
        Divide el texto por caracteres cuando no hay separador efectivo.
        
        Args:
            text: Texto a dividir
            
        Returns:
            Lista de chunks como strings
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.config.chunk_size - self.config.chunk_overlap
        
        return chunks
    
    def _split_large_segment(self, segment: str) -> List[str]:
        """
        Divide un segmento que es más grande que chunk_size.
        
        Args:
            segment: Segmento a dividir
            
        Returns:
            Lista de sub-chunks
        """
        chunks = []
        start = 0
        
        while start < len(segment):
            end = start + self.config.chunk_size
            chunks.append(segment[start:end])
            start += self.config.chunk_size - self.config.chunk_overlap
        
        return chunks
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Aplica overlap entre chunks consecutivos.
        
        Args:
            chunks: Lista de chunks sin overlap
            
        Returns:
            Lista de chunks con overlap aplicado
        """
        if len(chunks) <= 1 or self.config.chunk_overlap <= 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            # Tomar el final del chunk anterior
            previous_chunk = chunks[i - 1]
            overlap_text = previous_chunk[-self.config.chunk_overlap:] if len(previous_chunk) > self.config.chunk_overlap else previous_chunk
            
            # Combinar con el inicio del chunk actual
            current_chunk = overlap_text + " " + chunks[i]
            overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def get_chunk_count_estimate(self, text: str) -> int:
        """
        Estima el número de chunks que se generarán.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Número estimado de chunks
        """
        if not text:
            return 0
        
        effective_chunk_size = self.config.chunk_size - self.config.chunk_overlap
        estimated_chunks = max(1, (len(text) + effective_chunk_size - 1) // effective_chunk_size)
        
        return estimated_chunks


def chunk_text_simple(
    text: str,
    document_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Chunk]:
    """
    Función de conveniencia para chunking rápido con configuración simple.
    
    Args:
        text: Texto a dividir
        document_id: ID del documento origen
        chunk_size: Tamaño de cada chunk en caracteres
        chunk_overlap: Solapamiento entre chunks
        
    Returns:
        Lista de objetos Chunk
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunker = TextChunker(config)
    return chunker.chunk_text(text, document_id)
