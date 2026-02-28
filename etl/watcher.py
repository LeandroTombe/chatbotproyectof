"""
ETL Watcher — Ingesta automática de documentos.

Monitorea una carpeta en tiempo real con watchdog y, cada vez que
aparece un archivo nuevo soportado (.pdf), lo procesa de punta a punta:

  Archivo  →  Load  →  Chunk  →  Embed  →  ChromaDB

Uso como script autónomo:
    python etl/watcher.py                        # vigila data/pdfs
    python etl/watcher.py --watch data/mis_docs  # carpeta custom
    python etl/watcher.py --once  data/pdfs      # procesa sin vigilar

Uso programático:
    from etl.watcher import ETLWatcher, ingest_file, ingest_directory
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import threading
from pathlib import Path
from typing import Set

from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileMovedEvent,
)

from config.settings import settings
from domain.models import ChunkingConfig
from embeddings.base import EmbeddingConfig
from embeddings.factory import create_embedder
from vectorstore import create_vector_store
from ingestion.chunking import TextChunker
from ingestion.processor import DocumentProcessor, ProcessorException
from ingestion.pipeline import IngestionPipeline, BatchResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("etl.watcher")

SUPPORTED_EXTENSIONS: Set[str] = {".pdf"}

# Pausa (segundos) antes de procesar un archivo recién detectado,
# para darle tiempo a que termine de copiarse / descargarse.
DEBOUNCE_SECONDS: float = 2.0


# ---------------------------------------------------------------------------
# Construcción de componentes del pipeline
# ---------------------------------------------------------------------------

def _build_processor() -> DocumentProcessor:
    """
    Crea todos los componentes del pipeline RAG usando la configuración
    del archivo .env y los devuelve como DocumentProcessor listo para usar.
    """
    # Embedder
    emb_cfg = EmbeddingConfig(
        model_name=settings.EMBEDDING_MODEL,
        dimension=settings.EMBEDDING_DIMENSION,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
    embedder = create_embedder(provider=settings.EMBEDDING_PROVIDER, config=emb_cfg)
    logger.info(
        "Embedder: %s  (dim=%s)", settings.EMBEDDING_PROVIDER, settings.EMBEDDING_DIMENSION
    )

    # Vector store (Chroma u otro)
    try:
        vector_store = create_vector_store(
            provider=settings.VECTOR_STORE_TYPE,
            dimension=settings.EMBEDDING_DIMENSION,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )
        logger.info(
            "Vector store: %s  (colección=%s)",
            settings.VECTOR_STORE_TYPE,
            settings.CHROMA_COLLECTION_NAME,
        )
    except ValueError as exc:
        logger.warning("%s — usando almacenamiento en memoria", exc)
        vector_store = create_vector_store(
            provider="memory", dimension=settings.EMBEDDING_DIMENSION
        )

    # Chunker
    chunk_cfg = ChunkingConfig(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separator=settings.CHUNK_SEPARATOR,
    )
    chunker = TextChunker(config=chunk_cfg)
    logger.info(
        "Chunker: size=%s  overlap=%s", chunk_cfg.chunk_size, chunk_cfg.chunk_overlap
    )

    return DocumentProcessor(chunker=chunker, embedder=embedder, vector_store=vector_store)


# ---------------------------------------------------------------------------
# Funciones de ingesta pública
# ---------------------------------------------------------------------------

def ingest_file(file_path: str | Path, processor: DocumentProcessor | None = None) -> bool:
    """
    Procesa un único archivo a través del pipeline ETL completo.

    Args:
        file_path: Ruta al archivo a procesar.
        processor:  DocumentProcessor ya construido. Si es None se crea uno nuevo.

    Returns:
        True si el procesamiento fue exitoso, False en caso contrario.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("Archivo no encontrado: %s", path)
        return False

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning("Extensión no soportada: %s", path.suffix)
        return False

    _proc = processor or _build_processor()

    logger.info("── ETL ── Iniciando procesamiento: %s", path.name)
    start = time.perf_counter()
    try:
        doc = _proc.process_document(str(path))
        elapsed = time.perf_counter() - start
        logger.info(
            "── ETL ── ✓ %s  →  %d chunks  (%.2fs)",
            path.name,
            doc.total_chunks,
            elapsed,
        )
        return True
    except ProcessorException as exc:
        elapsed = time.perf_counter() - start
        logger.error("── ETL ── ✗ %s  →  %s  (%.2fs)", path.name, exc, elapsed)
        return False


def ingest_directory(
    directory: str | Path,
    recursive: bool = False,
    processor: DocumentProcessor | None = None,
) -> BatchResult:
    """
    Procesa todos los archivos soportados de una carpeta.

    Args:
        directory:  Carpeta a procesar.
        recursive:  Si True busca también en subcarpetas.
        processor:  DocumentProcessor ya construido (opcional).

    Returns:
        BatchResult con el resumen del procesamiento por lotes.
    """
    _proc = processor or _build_processor()
    pipeline = IngestionPipeline(processor=_proc, supported_extensions=list(SUPPORTED_EXTENSIONS))

    logger.info("── ETL ── Procesando carpeta: %s  (recursive=%s)", directory, recursive)
    result = pipeline.process_directory(directory, recursive=recursive)

    logger.info(
        "── ETL ── Resumen: %d/%d exitosos · %d chunks · tasa=%.1f%%",
        result.successful,
        result.total_files,
        result.total_chunks,
        result.success_rate,
    )

    if result.get_failed_files():
        for f in result.get_failed_files():
            logger.warning("── ETL ── ✗ falló: %s", f)

    return result


# ---------------------------------------------------------------------------
# Manejador de eventos del sistema de archivos
# ---------------------------------------------------------------------------

class _PDFEventHandler(FileSystemEventHandler):
    """
    Responde a eventos de creación / movimiento de archivos en la carpeta
    vigilada y lanza el pipeline ETL sobre los PDFs detectados.
    """

    def __init__(self, processor: DocumentProcessor) -> None:
        super().__init__()
        self._processor = processor
        self._pending: Set[str] = set()   # rutas pendientes de procesar
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if not event.is_directory:
            self._schedule(str(event.src_path))

    def on_moved(self, event: FileMovedEvent) -> None:  # type: ignore[override]
        if not event.is_directory:
            self._schedule(str(event.dest_path))

    # ------------------------------------------------------------------
    def _schedule(self, path: str) -> None:
        """Agenda el archivo para procesarlo luego del debounce."""
        if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        with self._lock:
            if path in self._pending:
                return
            self._pending.add(path)

        logger.info("── ETL ── Archivo detectado: %s", Path(path).name)
        thread = threading.Thread(
            target=self._process_after_debounce,
            args=(path,),
            daemon=True,
        )
        thread.start()

    def _process_after_debounce(self, path: str) -> None:
        """Espera DEBOUNCE_SECONDS y luego procesa el archivo."""
        time.sleep(DEBOUNCE_SECONDS)
        with self._lock:
            self._pending.discard(path)

        ingest_file(path, processor=self._processor)


# ---------------------------------------------------------------------------
# ETLWatcher — clase principal
# ---------------------------------------------------------------------------

class ETLWatcher:
    """
    Vigila una carpeta y ejecuta el pipeline ETL automáticamente
    cuando aparece un archivo nuevo soportado.

    >>> watcher = ETLWatcher("data/pdfs")
    >>> watcher.start()          # no bloquea
    >>> watcher.stop()
    """

    def __init__(self, watch_dir: str | Path = "data/pdfs") -> None:
        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self._processor = _build_processor()
        self._observer: Observer | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Inicia el observador en un hilo de fondo (no bloqueante)."""
        if self._observer and self._observer.is_alive():
            logger.warning("ETLWatcher ya está en ejecución.")
            return

        handler = _PDFEventHandler(self._processor)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.watch_dir), recursive=False)
        self._observer.start()
        logger.info("ETLWatcher activo — vigilando: %s", self.watch_dir.resolve())

    def stop(self) -> None:
        """Detiene el observador."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("ETLWatcher detenido.")

    def run_forever(self) -> None:
        """Bloquea el proceso actual mientras vigila la carpeta. Ctrl+C para salir."""
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # Soporte para uso como context manager
    def __enter__(self) -> "ETLWatcher":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Punto de entrada CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETL Watcher — ingesta automática de documentos en ChromaDB"
    )
    parser.add_argument(
        "--watch",
        metavar="DIR",
        default=settings.DATA_DIR,
        help="Carpeta a vigilar (default: %(default)s)",
    )
    parser.add_argument(
        "--once",
        metavar="DIR",
        default=None,
        help="Procesa todos los PDFs de DIR una sola vez y termina",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Buscar PDFs en subcarpetas también (solo con --once)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.once:
        # Modo de procesamiento por lotes (sin watcher)
        result = ingest_directory(args.once, recursive=args.recursive)
        sys.exit(0 if result.failed == 0 else 1)
    else:
        # Modo watcher (correr indefinidamente)
        print(f"\n  ETL Watcher iniciando…")
        print(f"  Carpeta vigilada : {Path(args.watch).resolve()}")
        print(f"  Vector store     : {settings.VECTOR_STORE_TYPE}")
        print(f"  Embedder         : {settings.EMBEDDING_PROVIDER}")
        print(f"  Colección Chroma : {settings.CHROMA_COLLECTION_NAME}")
        print(f"\n  Copiá tus PDFs a la carpeta y serán ingestados automáticamente.")
        print(f"  Presioná Ctrl+C para salir.\n")

        watcher = ETLWatcher(watch_dir=args.watch)
        watcher.run_forever()
