"""
ETL package — ingesta automática de documentos.

Exporta el watcher y la función de ingesta manual
para que otros módulos puedan reutilizarlos.
"""
from etl.watcher import ETLWatcher, ingest_file, ingest_directory

__all__ = ["ETLWatcher", "ingest_file", "ingest_directory"]
