"""
Main demo script for ChatBot RAG System.
Demonstrates the complete document processing and retrieval pipeline.
"""
from pathlib import Path
from typing import Optional
import logging

from config.settings import settings
from domain.models import ChunkingConfig
from embeddings.base import EmbeddingConfig
from embeddings.factory import create_embedder
from vectorstore.base import InMemoryVectorStore
from processing.chunking import TextChunker
from documents.processor import DocumentProcessor
from retrieval.retriever import DocumentRetriever
from ingestion.pipeline import IngestionPipeline
from chat.llm_clients.ollama_client import OllamaClient
from chat.llm_clients.base import LLMConfig, LLMConnectionError
from chat.rag_service import RAGService, RAGConfig
from chat.security import SecurityConfig
from chat.models import Message, MessageRole

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_separator(title: str = ""):
    """Print a visual separator"""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'-'*80}\n")


def setup_components():
    """
    Initialize all components of the RAG system.
    
    Returns:
        Tuple of (processor, retriever, pipeline)
    """
    print_separator("SETTING UP RAG SYSTEM COMPONENTS")
    
    # 1. Embedding configuration
    embedding_config = EmbeddingConfig(
        model_name=settings.EMBEDDING_MODEL,
        dimension=settings.EMBEDDING_DIMENSION,
        batch_size=settings.EMBEDDING_BATCH_SIZE
    )
    
    # Create embedder using factory pattern (no if/else needed)
    embedder = create_embedder(
        provider=settings.EMBEDDING_PROVIDER,
        config=embedding_config
    )
    print(f"✓ Embedder: {embedder} (provider: {settings.EMBEDDING_PROVIDER})")
    
    # 2. Vector Store
    vector_store = InMemoryVectorStore(dimension=settings.EMBEDDING_DIMENSION)
    print(f"✓ Vector Store: {vector_store}")
    
    # 3. Chunking configuration
    chunking_config = ChunkingConfig(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separator=settings.CHUNK_SEPARATOR
    )
    chunker = TextChunker(config=chunking_config)
    print(f"✓ Chunker: chunk_size={chunking_config.chunk_size}, overlap={chunking_config.chunk_overlap}")
    
    # 4. Document Processor
    processor = DocumentProcessor(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store
    )
    print(f"✓ Document Processor: Ready")
    
    # 5. Document Retriever
    retriever = DocumentRetriever(
        embedder=embedder,
        vector_store=vector_store
    )
    print(f"✓ Document Retriever: Ready")
    
    # 6. Ingestion Pipeline
    pipeline = IngestionPipeline(
        processor=processor,
        supported_extensions=['.pdf']
    )
    print(f"✓ Ingestion Pipeline: Ready")
    
    print("\n✅ All components initialized successfully!")
    
    return processor, retriever, pipeline, vector_store


def demo_process_single_file(processor: DocumentProcessor, file_path: str):
    """
    Demonstrate processing a single PDF file.
    
    Args:
        processor: DocumentProcessor instance
        file_path: Path to PDF file
    """
    print_separator("DEMO: PROCESSING SINGLE FILE")
    
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        print(f"   Please place a PDF file at this location or provide a different path.")
        return None
    
    print(f"📄 Processing file: {path.name}")
    print(f"   Path: {path}")
    
    try:
        document = processor.process_document(file_path)
        
        print(f"\n✅ Processing completed!")
        print(f"   Document ID: {document.id}")
        print(f"   Status: {document.status.value}")
        print(f"   Total Chunks: {document.total_chunks}")
        print(f"   File Name: {document.file_name}")
        
        if document.processed_at:
            print(f"   Processed At: {document.processed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return document
        
    except Exception as e:
        print(f"\n❌ Error processing file: {str(e)}")
        logger.exception("Error in demo_process_single_file")
        return None


def demo_batch_processing(pipeline: IngestionPipeline, directory_path: str):
    """
    Demonstrate batch processing of multiple files.
    
    Args:
        pipeline: IngestionPipeline instance
        directory_path: Path to directory containing PDFs
    """
    print_separator("DEMO: BATCH PROCESSING DIRECTORY")
    
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        print(f"❌ Directory not found: {directory_path}")
        print(f"   Creating directory: {directory_path}")
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   Please add PDF files to {directory_path} and run again.")
        return None
    
    print(f"📁 Processing directory: {dir_path}")
    print(f"   Recursive: True")
    
    try:
        result = pipeline.process_directory(
            directory_path=directory_path,
            recursive=True,
            continue_on_error=True
        )
        
        print(f"\n✅ Batch processing completed!")
        print(f"   Total Files: {result.total_files}")
        print(f"   Successful: {result.successful}")
        print(f"   Failed: {result.failed}")
        print(f"   Success Rate: {result.success_rate:.1f}%")
        print(f"   Total Chunks: {result.total_chunks}")
        
        if result.successful > 0:
            print(f"\n📊 Successful files:")
            for file_path in result.get_successful_files():
                print(f"   ✓ {Path(file_path).name}")
        
        if result.failed > 0:
            print(f"\n⚠️  Failed files:")
            for file_path in result.get_failed_files():
                print(f"   ✗ {Path(file_path).name}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error in batch processing: {str(e)}")
        logger.exception("Error in demo_batch_processing")
        return None


def demo_search(retriever: DocumentRetriever, query: str, top_k: int = 5):
    """
    Demonstrate semantic search.
    
    Args:
        retriever: DocumentRetriever instance
        query: Search query
        top_k: Number of results to return
    """
    print_separator("DEMO: SEMANTIC SEARCH")
    
    print(f"🔍 Query: '{query}'")
    print(f"   Top K: {top_k}")
    
    try:
        results = retriever.search(query=query, top_k=top_k)
        
        if not results:
            print(f"\n❌ No results found.")
            print(f"   Make sure documents have been processed first.")
            return
        
        print(f"\n✅ Found {len(results)} results:\n")
        
        for idx, result in enumerate(results, 1):
            print(f"📄 Result #{idx}")
            print(f"   Score: {result.score:.4f}")
            print(f"   Document: {result.document_name}")
            print(f"   Chunk Index: {result.chunk.chunk_index}")
            print(f"   Content Preview: {result.chunk.content[:200]}...")
            print()
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during search: {str(e)}")
        logger.exception("Error in demo_search")
        return None


def demo_get_context(retriever: DocumentRetriever, query: str, top_k: int = 3):
    """
    Demonstrate context extraction for RAG.
    
    Args:
        retriever: DocumentRetriever instance
        query: Search query
        top_k: Number of chunks to include
    """
    print_separator("DEMO: RAG CONTEXT EXTRACTION")
    
    print(f"🔍 Query: '{query}'")
    print(f"   Top K: {top_k}")
    
    try:
        context = retriever.get_context(query=query, top_k=top_k)
        
        if not context:
            print(f"\n❌ No context found.")
            return
        
        print(f"\n✅ Context extracted ({len(context)} characters):\n")
        print("="*80)
        print(context)
        print("="*80)
        
        return context
        
    except Exception as e:
        print(f"\n❌ Error extracting context: {str(e)}")
        logger.exception("Error in demo_get_context")
        return None


def show_vector_store_stats(vector_store: InMemoryVectorStore):
    """
    Display vector store statistics.
    
    Args:
        vector_store: InMemoryVectorStore instance
    """
    print_separator("VECTOR STORE STATISTICS")
    
    total_chunks = vector_store.count()
    
    print(f"📊 Vector Store Stats:")
    print(f"   Total Chunks: {total_chunks}")
    print(f"   Dimension: {vector_store.dimension}")
    
    if total_chunks > 0:
        all_chunks = vector_store.get_all_chunks()
        
        # Get unique documents
        unique_docs = set(chunk.document_id for chunk in all_chunks)
        print(f"   Unique Documents: {len(unique_docs)}")
        
        # Show document breakdown
        if len(unique_docs) <= 10:  # Only show if reasonable number
            print(f"\n   Document Breakdown:")
            for doc_id in sorted(unique_docs):
                doc_chunks = [c for c in all_chunks if c.document_id == doc_id]
                print(f"   - {doc_id}: {len(doc_chunks)} chunks")


def setup_chat_service(
    retriever: DocumentRetriever
) -> Optional[RAGService]:
    """
    Initialize the chat service with Ollama.
    
    Args:
        retriever: DocumentRetriever instance
        
    Returns:
        RAGService instance or None if Ollama is not available
    """
    print_separator("SETTING UP CHAT SERVICE")
    
    print(f"🤖 Initializing Ollama client...")
    print(f"   Model: {settings.OLLAMA_MODEL}")
    print(f"   Base URL: {settings.OLLAMA_BASE_URL}")
    
    # 1. LLM Configuration
    llm_config = LLMConfig(
        model_name=settings.OLLAMA_MODEL,
        temperature=settings.OLLAMA_TEMPERATURE,
        max_tokens=settings.OLLAMA_MAX_TOKENS,
        timeout=settings.OLLAMA_TIMEOUT
    )
    
    # 2. Create Ollama client
    ollama_client = OllamaClient(llm_config, base_url=settings.OLLAMA_BASE_URL)
    
    # 3. Check if Ollama is available
    if not ollama_client.is_available():
        print(f"\n⚠️  Ollama is not running at {settings.OLLAMA_BASE_URL}")
        print(f"   To use the chat service:")
        print(f"   1. Install Ollama from https://ollama.ai")
        print(f"   2. Run: ollama serve")
        print(f"   3. Pull a model: ollama pull {settings.OLLAMA_MODEL}")
        print(f"\n   Skipping chat service setup...\n")
        return None
    
    print(f"✓ Ollama is running!")
    
    # 4. Get model info
    try:
        model_info = ollama_client.get_model_info()
        
        if not model_info['available']:
            print(f"\n⚠️  Model '{settings.OLLAMA_MODEL}' is not available")
            print(f"   Available models: {', '.join(model_info['all_models'])}")
            print(f"\n   To pull the model:")
            print(f"   ollama pull {settings.OLLAMA_MODEL}")
            print(f"\n   Skipping chat service setup...\n")
            return None
        
        print(f"✓ Model '{settings.OLLAMA_MODEL}' is available")
        
    except Exception as e:
        print(f"⚠️  Could not get model info: {e}")
    
    # 5. RAG Configuration
    rag_config = RAGConfig(
        top_k=settings.RAG_TOP_K,
        min_relevance=settings.RAG_MIN_RELEVANCE,
        max_context_length=settings.RAG_MAX_CONTEXT_LENGTH,
        include_sources=settings.RAG_INCLUDE_SOURCES,
        strict_mode=settings.RAG_STRICT_MODE,
        system_prompt=settings.RAG_SYSTEM_PROMPT
    )
    
    # 6. Security configuration
    security_config = SecurityConfig(
        enabled=settings.RAG_ENABLE_SECURITY
    )
    
    # 7. Create RAG service
    rag_service = RAGService(
        retriever=retriever,
        llm_client=ollama_client,
        config=rag_config,
        security_config=security_config
    )
    
    print(f"✓ RAG Service initialized")
    print(f"\n✅ Chat service ready!")
    
    # 7. Pre-warm the model with a simple query
    print(f"\n🔥 Pre-warming model (primera consulta puede tardar)...")
    try:
        warmup_messages = [
            Message(MessageRole.USER, "Hi")
        ]
        ollama_client.generate(warmup_messages, max_tokens=10)
        print(f"✓ Model is ready!\n")
    except Exception as e:
        print(f"⚠️  Model warmup warning: {e}")
        print(f"   Continuing anyway...\n")
    
    return rag_service


def demo_chat(rag_service: RAGService, query: str):
    """
    Demonstrate a single chat interaction.
    
    Args:
        rag_service: RAGService instance
        query: User query
    """
    print_separator("DEMO: RAG CHAT")
    
    print(f"👤 User: {query}")
    print(f"\n🤔 Processing...")
    
    try:
        response = rag_service.chat(query)
        
        print(f"\n🤖 Assistant: {response.content}")
        
        if response.sources:
            print(f"\n📚 Sources ({len(response.sources)}):")
            for idx, source in enumerate(response.sources, 1):
                print(f"   [{idx}] Document: {source.document_id}")
                print(f"       Relevance: {source.relevance_score:.2f}")
                print(f"       Content: {source.content[:100]}...")
                print()
        
        print(f"ℹ️  Metadata:")
        print(f"   Model: {response.metadata.get('model')}")
        print(f"   Sources used: {response.metadata.get('num_sources')}")
        print(f"   Has context: {response.metadata.get('has_context')}")
        
        return response
        
    except LLMConnectionError as e:
        print(f"\n❌ Connection Error: {str(e)}")
        print(f"   Make sure Ollama is running.")
        return None
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.exception("Error in demo_chat")
        return None


def interactive_chat(rag_service: RAGService):
    """
    Interactive chat session with the RAG service.
    
    Args:
        rag_service: RAGService instance
    """
    print_separator("INTERACTIVE CHAT")
    
    print("💬 Chat with your documents!")
    print("   Commands:")
    print("   - Type your question to chat")
    print("   - 'clear' to clear conversation history")
    print("   - 'history' to see conversation history")
    print("   - 'quit' or 'exit' to exit")
    print()
    
    while True:
        try:
            query = input("👤 You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            if query.lower() == 'clear':
                rag_service.clear_conversation()
                print("🗑️  Conversation history cleared.\n")
                continue
            
            if query.lower() == 'history':
                history = rag_service.get_conversation_history()
                messages = history.get_messages(include_system=False)
                
                if not messages:
                    print("📭 No conversation history yet.\n")
                else:
                    print(f"\n📜 Conversation History ({len(messages)} messages):\n")
                    for msg in messages:
                        role_icon = "👤" if msg.role.value == "user" else "🤖"
                        print(f"{role_icon} {msg.role.value.capitalize()}: {msg.content[:100]}...")
                    print()
                continue
            
            # Chat
            print("\n🤔 Thinking...")
            response = rag_service.chat(query)
            
            print(f"\n🤖 Assistant: {response.content}\n")
            
            if response.sources:
                print(f"📚 [{len(response.sources)} sources used]\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except LLMConnectionError as e:
            print(f"\n❌ Connection Error: {str(e)}")
            print(f"   Ollama may have stopped. Please restart it.\n")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


def main():
    """
    Main demo function - orchestrates all demos.
    """
    print("\n")
    print("="*80)
    print(" "*20 + "CHATBOT RAG SYSTEM - DEMO")
    print("="*80)
    print("\nThis demo showcases the complete RAG pipeline:")
    print("  1. Document processing (PDF → Chunks → Embeddings → Storage)")
    print("  2. Batch ingestion (multiple files)")
    print("  3. Semantic search (query → relevant chunks)")
    print("  4. Context extraction (for RAG)")
    print("  5. RAG Chat with Ollama (conversational AI)")
    print()
    
    # Setup components
    processor, retriever, pipeline, vector_store = setup_components()
    
    # Configuration
    DATA_DIR = Path(settings.DATA_DIR)
    SAMPLE_PDF = DATA_DIR / "sample.pdf"
    
    # Demo 1: Process single file
    print("\n")
    input("Press ENTER to start Demo 1: Process Single File...")
    
    document = demo_process_single_file(processor, str(SAMPLE_PDF))
    
    # Demo 2: Batch processing
    print("\n")
    input("Press ENTER to start Demo 2: Batch Processing...")
    
    batch_result = demo_batch_processing(pipeline, str(DATA_DIR))
    
    # Show vector store stats
    print("\n")
    input("Press ENTER to view Vector Store Statistics...")
    
    show_vector_store_stats(vector_store)
    
    # Demo 3: Semantic search
    print("\n")
    input("Press ENTER to start Demo 3: Semantic Search...")
    
    demo_search(
        retriever,
        query="What is the main topic of the document?",
        top_k=5
    )
    
    # Demo 4: Context extraction
    print("\n")
    input("Press ENTER to start Demo 4: RAG Context Extraction...")
    
    demo_get_context(
        retriever,
        query="Explain the key concepts",
        top_k=3
    )
    
    # Demo 5: Setup chat service
    print("\n")
    response = input("Press ENTER to setup Chat Service (or 'skip' to skip chat demos)...")
    
    rag_service = None
    if response.lower() != 'skip':
        rag_service = setup_chat_service(
            retriever=retriever
        )
    
    if rag_service:
        # Demo 6: Single chat interaction
        print("\n")
        input("Press ENTER to start Demo 5: RAG Chat...")
        
        demo_chat(
            rag_service,
            query="¿Cuál es el tema principal del documento?"
        )
        
        # Demo 7: Interactive chat
        print("\n")
        response = input("Press ENTER to start Interactive Chat (or 'skip' to skip)...")
        
        if response.lower() != 'skip':
            interactive_chat(rag_service)
    else:
        print("\n⚠️  Chat service not available. Skipping chat demos.\n")
    
    # Interactive search (non-chat)
    if not rag_service or response.lower() == 'skip':
        print_separator("INTERACTIVE SEARCH")
        print("You can now query the documents interactively.")
        print("Type your questions (or 'quit' to exit):\n")
        
        while True:
            try:
                query = input("🔍 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print()
                results = retriever.search(query=query, top_k=3)
                
                if results:
                    print(f"Found {len(results)} results:\n")
                    for idx, result in enumerate(results, 1):
                        print(f"{idx}. [Score: {result.score:.3f}] {result.chunk.content[:150]}...")
                    print()
                else:
                    print("No results found.\n")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
    
    print_separator("DEMO COMPLETED")
    print("Thank you for trying the ChatBot RAG System!")
    print()



if __name__ == "__main__":
    main()
