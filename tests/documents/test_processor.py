"""
Unit tests for DocumentProcessor.
"""
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import pytest

from documents.processor import DocumentProcessor, ProcessorException
from domain.models import Document, ProcessingStatus, Chunk
from documents.loaders.base_loader import LoaderException


class TestDocumentProcessorInitialization:
    """Tests para inicialización de DocumentProcessor"""
    
    def test_initialization_with_all_dependencies(self):
        """Prueba inicialización con todas las dependencias"""
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        loader = Mock()
        
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            loader=loader
        )
        
        assert processor.chunker == chunker
        assert processor.embedder == embedder
        assert processor.vector_store == vector_store
        assert processor.loader == loader
    
    def test_initialization_without_loader(self):
        """Prueba inicialización sin loader (usa factory)"""
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store
        )
        
        assert processor.chunker == chunker
        assert processor.embedder == embedder
        assert processor.vector_store == vector_store
        assert processor.loader is None


class TestDocumentProcessorProcessDocument:
    """Tests para process_document"""
    
    def test_process_document_success_with_loader(self):
        """Prueba procesamiento exitoso con loader provisto"""
        # Setup mocks
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        loader = Mock()
        
        # Mock document
        mock_doc = Document(
            id="doc1",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING,
            metadata={"content": "Test content for chunking"}
        )
        loader.load.return_value = mock_doc
        
        # Mock chunks
        mock_chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="Test content",
                chunk_index=0,
                metadata={}
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="for chunking",
                chunk_index=1,
                metadata={}
            )
        ]
        chunker.chunk_text.return_value = mock_chunks
        
        # Mock embeddings
        embedder.embed_text.side_effect = [
            [0.1, 0.2, 0.3],  # embedding for chunk1
            [0.4, 0.5, 0.6]   # embedding for chunk2
        ]
        
        # Create processor and process
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            loader=loader
        )
        
        result = processor.process_document("test.pdf")
        
        # Verify document was loaded
        loader.load.assert_called_once_with("test.pdf")
        
        # Verify chunking
        chunker.chunk_text.assert_called_once_with(
            text="Test content for chunking",
            document_id="doc1",
            metadata={"content": "Test content for chunking"}
        )
        
        # Verify embeddings generated
        assert embedder.embed_text.call_count == 2
        embedder.embed_text.assert_any_call("Test content")
        embedder.embed_text.assert_any_call("for chunking")
        
        # Verify chunks stored
        vector_store.add_chunks.assert_called_once_with(mock_chunks)
        
        # Verify document status
        assert result.status == ProcessingStatus.COMPLETED
        assert result.total_chunks == 2
        assert result.processed_at is not None
    
    def test_process_document_success_with_factory(self):
        """Prueba procesamiento exitoso usando factory"""
        # Setup mocks
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        
        # Mock document
        mock_doc = Document(
            id="doc2",
            file_path=Path("test2.pdf"),
            file_name="test2.pdf",
            status=ProcessingStatus.PENDING,
            metadata={"content": "Factory loaded content"}
        )
        
        # Mock chunk
        mock_chunk = Chunk(
            id="chunk1",
            document_id="doc2",
            content="Factory loaded content",
            chunk_index=0,
            metadata={}
        )
        chunker.chunk_text.return_value = [mock_chunk]
        
        # Mock embedding
        embedder.embed_text.return_value = [0.7, 0.8, 0.9]
        
        # Create processor without loader
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store
        )
        
        # Mock the factory get_loader
        with patch('ingestion.processor.get_loader') as mock_get_loader:
            mock_loader = Mock()
            mock_loader.load.return_value = mock_doc
            mock_get_loader.return_value = mock_loader
            
            result = processor.process_document("test2.pdf")
            
            # Verify factory was used
            mock_get_loader.assert_called_once_with("test2.pdf")
            mock_loader.load.assert_called_once_with("test2.pdf")
            
            # Verify processing completed
            assert result.status == ProcessingStatus.COMPLETED
            assert result.total_chunks == 1
    
    def test_process_document_handles_path_object(self):
        """Prueba que acepta Path objects"""
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        loader = Mock()
        
        mock_doc = Document(
            id="doc3",
            file_path=Path("test3.pdf"),
            file_name="test3.pdf",
            status=ProcessingStatus.PENDING,
            metadata={"content": "Content"}
        )
        loader.load.return_value = mock_doc
        
        mock_chunk = Chunk(
            id="chunk1",
            document_id="doc3",
            content="Content",
            chunk_index=0,
            metadata={}
        )
        chunker.chunk_text.return_value = [mock_chunk]
        embedder.embed_text.return_value = [0.1, 0.2]
        
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            loader=loader
        )
        
        # Pass Path object
        result = processor.process_document(Path("test3.pdf"))
        
        # Should convert to string when calling loader
        loader.load.assert_called_once_with("test3.pdf")
        assert result.status == ProcessingStatus.COMPLETED


class TestDocumentProcessorErrors:
    """Tests para manejo de errores"""
    
    def test_process_document_loader_fails(self):
        """Prueba error al cargar documento"""
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        loader = Mock()
        
        loader.load.side_effect = LoaderException("Failed to load")
        
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            loader=loader
        )
        
        with pytest.raises(ProcessorException, match="Failed to load document"):
            processor.process_document("test.pdf")
    
    def test_process_document_no_content_in_metadata(self):
        """Prueba error cuando documento no tiene content en metadata"""
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        loader = Mock()
        
        mock_doc = Document(
            id="doc4",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING,
            metadata={}  # No content
        )
        loader.load.return_value = mock_doc
        
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            loader=loader
        )
        
        with pytest.raises(ProcessorException, match="has no 'content' in metadata"):
            processor.process_document("test.pdf")
        
        # Document should be marked as failed
        assert mock_doc.status == ProcessingStatus.FAILED
        assert mock_doc.error_message is not None
    
    def test_process_document_empty_content(self):
        """Prueba error cuando content está vacío"""
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        loader = Mock()
        
        mock_doc = Document(
            id="doc5",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING,
            metadata={"content": "   "}  # Empty/whitespace
        )
        loader.load.return_value = mock_doc
        
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            loader=loader
        )
        
        with pytest.raises(ProcessorException, match="has empty content"):
            processor.process_document("test.pdf")
        
        assert mock_doc.status == ProcessingStatus.FAILED
    
    def test_process_document_no_chunks_created(self):
        """Prueba error cuando no se crean chunks"""
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        loader = Mock()
        
        mock_doc = Document(
            id="doc6",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING,
            metadata={"content": "Some content"}
        )
        loader.load.return_value = mock_doc
        
        # Chunker returns empty list
        chunker.chunk_text.return_value = []
        
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            loader=loader
        )
        
        with pytest.raises(ProcessorException, match="No chunks created"):
            processor.process_document("test.pdf")
        
        assert mock_doc.status == ProcessingStatus.FAILED
    
    def test_process_document_chunking_fails(self):
        """Prueba error durante chunking"""
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        loader = Mock()
        
        mock_doc = Document(
            id="doc7",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING,
            metadata={"content": "Content"}
        )
        loader.load.return_value = mock_doc
        
        from processing.chunking import ChunkingException
        chunker.chunk_text.side_effect = ChunkingException("Chunking failed")
        
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            loader=loader
        )
        
        with pytest.raises(ProcessorException, match="Error processing"):
            processor.process_document("test.pdf")
        
        assert mock_doc.status == ProcessingStatus.FAILED
    
    def test_process_document_embedding_fails(self):
        """Prueba error durante generación de embeddings"""
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        loader = Mock()
        
        mock_doc = Document(
            id="doc8",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING,
            metadata={"content": "Content"}
        )
        loader.load.return_value = mock_doc
        
        mock_chunk = Chunk(
            id="chunk1",
            document_id="doc8",
            content="Content",
            chunk_index=0,
            metadata={}
        )
        chunker.chunk_text.return_value = [mock_chunk]
        
        from embeddings.base import EmbeddingException
        embedder.embed_text.side_effect = EmbeddingException("Embedding failed")
        
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            loader=loader
        )
        
        with pytest.raises(ProcessorException, match="Error processing"):
            processor.process_document("test.pdf")
        
        assert mock_doc.status == ProcessingStatus.FAILED
    
    def test_process_document_vector_store_fails(self):
        """Prueba error al almacenar en vector store"""
        chunker = Mock()
        embedder = Mock()
        vector_store = Mock()
        loader = Mock()
        
        mock_doc = Document(
            id="doc9",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING,
            metadata={"content": "Content"}
        )
        loader.load.return_value = mock_doc
        
        mock_chunk = Chunk(
            id="chunk1",
            document_id="doc9",
            content="Content",
            chunk_index=0,
            metadata={}
        )
        chunker.chunk_text.return_value = [mock_chunk]
        embedder.embed_text.return_value = [0.1, 0.2]
        
        from vectorstore.base import VectorStoreException
        vector_store.add_chunks.side_effect = VectorStoreException("Storage failed")
        
        processor = DocumentProcessor(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            loader=loader
        )
        
        with pytest.raises(ProcessorException, match="Error processing"):
            processor.process_document("test.pdf")
        
        assert mock_doc.status == ProcessingStatus.FAILED


class TestDocumentProcessorHelperMethods:
    """Tests para métodos helper privados"""
    
    def test_extract_text_success(self):
        """Prueba extracción exitosa de texto"""
        processor = DocumentProcessor(
            chunker=Mock(),
            embedder=Mock(),
            vector_store=Mock()
        )
        
        doc = Document(
            id="doc10",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING,
            metadata={"content": "Test text content"}
        )
        
        text = processor._extract_text(doc)
        assert text == "Test text content"
    
    def test_extract_text_no_content_key(self):
        """Prueba error cuando no hay key 'content'"""
        processor = DocumentProcessor(
            chunker=Mock(),
            embedder=Mock(),
            vector_store=Mock()
        )
        
        doc = Document(
            id="doc11",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING,
            metadata={"other_key": "value"}
        )
        
        with pytest.raises(ProcessorException, match="has no 'content' in metadata"):
            processor._extract_text(doc)
    
    def test_extract_text_empty_content(self):
        """Prueba error cuando content es vacío"""
        processor = DocumentProcessor(
            chunker=Mock(),
            embedder=Mock(),
            vector_store=Mock()
        )
        
        doc = Document(
            id="doc12",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING,
            metadata={"content": ""}
        )
        
        with pytest.raises(ProcessorException, match="has empty content"):
            processor._extract_text(doc)
