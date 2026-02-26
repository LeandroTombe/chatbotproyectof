"""
ChatBot RAG — Sistema de preguntas y respuestas sobre documentos.

Flujo principal:
  1. Inicializar componentes (embedder, vector store, chunker)
  2. Ingresar documentos PDF al sistema
  3. Buscar información relevante en los documentos
  4. Chatear con la IA restringida a los documentos cargados
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import tkinter as tk
    from tkinter import filedialog as _fd
    _TKINTER = True
except ImportError:
    _TKINTER = False

from config.settings import settings
from domain.models import ChunkingConfig
from embeddings.base import EmbeddingConfig
from embeddings.factory import create_embedder
from vectorstore import create_vector_store, BaseVectorStore
from ingestion.chunking import TextChunker
from ingestion.processor import DocumentProcessor
from ingestion.pipeline import IngestionPipeline
from retrieval.retriever import DocumentRetriever, RetrieverException
from chat.llm_clients.ollama_client import OllamaClient
from chat.llm_clients.base import LLMConfig, LLMConnectionError
from chat.rag_service import RAGService, RAGConfig
from chat.security import SecurityConfig
from chat.models import Message, MessageRole

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Utilidades de interfaz
# ─────────────────────────────────────────────────────────────────────────────

W = 60  # ancho de la caja


def _titulo(texto: str) -> None:
    """Imprime un encabezado de sección."""
    barra = "─" * W
    relleno = max(0, W - len(texto) - 2)
    print(f"\n┌{barra}┐")
    print(f"│  {texto}{' ' * relleno}│")
    print(f"└{barra}┘\n")


def _linea() -> None:
    print("  " + "·" * (W - 2))


def _ok(msg: str)   -> None: print(f"  ✓  {msg}")
def _aviso(msg: str) -> None: print(f"  ⚠  {msg}")
def _error(msg: str) -> None: print(f"  ✗  {msg}")


def _leer(prompt: str) -> str:
    """Lee una línea del usuario. Ctrl+C devuelve cadena vacía."""
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        return ""


def _abrir_explorador_archivo() -> str:
    """Abre el explorador de Windows para seleccionar un PDF.
    Devuelve la ruta elegida, o cadena vacía si se canceló."""
    if not _TKINTER:
        return ""
    root = tk.Tk()
    root.withdraw()          # oculta la ventana principal de tkinter
    root.attributes("-topmost", True)   # el diálogo aparece al frente
    ruta = _fd.askopenfilename(
        title="Seleccioná un archivo PDF",
        filetypes=[("Archivos PDF", "*.pdf"), ("Todos los archivos", "*.*")],
    )
    root.destroy()
    return ruta or ""


def _abrir_explorador_carpeta() -> str:
    """Abre el explorador de Windows para seleccionar una carpeta.
    Devuelve la ruta elegida, o cadena vacía si se canceló."""
    if not _TKINTER:
        return ""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ruta = _fd.askdirectory(
        title="Seleccioná la carpeta con los PDFs",
        mustexist=True,
    )
    root.destroy()
    return ruta or ""


# ─────────────────────────────────────────────────────────────────────────────
#  Inicialización del sistema
# ─────────────────────────────────────────────────────────────────────────────

def inicializar() -> tuple[DocumentProcessor, DocumentRetriever, IngestionPipeline, BaseVectorStore]:
    """Crea y conecta todos los componentes del pipeline RAG."""
    _titulo("INICIALIZANDO SISTEMA")

    # Embedder
    emb_cfg = EmbeddingConfig(
        model_name=settings.EMBEDDING_MODEL,
        dimension=settings.EMBEDDING_DIMENSION,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
    embedder = create_embedder(provider=settings.EMBEDDING_PROVIDER, config=emb_cfg)
    _ok(f"Embedder      : {settings.EMBEDDING_PROVIDER}  (dim={settings.EMBEDDING_DIMENSION})")

    # Vector store
    try:
        vector_store = create_vector_store(
            provider=settings.VECTOR_STORE_TYPE,
            dimension=settings.EMBEDDING_DIMENSION,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )
        detalle = f"  colección={settings.CHROMA_COLLECTION_NAME}" if settings.VECTOR_STORE_TYPE == "chroma" else ""
        _ok(f"Vector store  : {settings.VECTOR_STORE_TYPE.capitalize()}{detalle}")
    except ValueError as exc:
        _aviso(f"{exc} — usando almacenamiento en memoria")
        vector_store = create_vector_store(provider="memory", dimension=settings.EMBEDDING_DIMENSION)
        _ok("Vector store  : Memoria (no persistente)")

    # Chunker
    chunk_cfg = ChunkingConfig(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separator=settings.CHUNK_SEPARATOR,
    )
    chunker = TextChunker(config=chunk_cfg)
    _ok(f"Chunker       : tamaño={chunk_cfg.chunk_size}  solapamiento={chunk_cfg.chunk_overlap}")

    # Procesador, recuperador y pipeline
    processor = DocumentProcessor(chunker=chunker, embedder=embedder, vector_store=vector_store)
    retriever = DocumentRetriever(embedder=embedder, vector_store=vector_store)
    pipeline  = IngestionPipeline(processor=processor, supported_extensions=[".pdf"])
    _ok("Procesador    : listo")
    _ok("Recuperador   : listo  (estándar · MMR · expandida)")
    _ok("Pipeline      : listo  (.pdf)")

    print("\n  Sistema listo.\n")
    return processor, retriever, pipeline, vector_store


# ─────────────────────────────────────────────────────────────────────────────
#  Ingesta de documentos
# ─────────────────────────────────────────────────────────────────────────────

def menu_ingesta(processor: DocumentProcessor, pipeline: IngestionPipeline) -> None:
    """Submenú para cargar documentos al sistema."""
    while True:
        _titulo("CARGAR DOCUMENTOS")
        print("  1  Cargar un archivo PDF")
        print("  2  Cargar todos los PDFs de una carpeta")
        print("  0  Volver al menú principal\n")

        opcion = _leer("  Opción: ")

        if opcion == "1":
            ruta = _pedir_ruta_archivo()
            if ruta:
                _cargar_archivo(processor, ruta)
        elif opcion == "2":
            carpeta = _pedir_ruta_carpeta()
            if carpeta:
                _cargar_carpeta(pipeline, carpeta)
        elif opcion in {"0", ""}:
            break
        else:
            _aviso("Opción no válida.\n")


def _pedir_ruta_archivo() -> str:
    """Pide al usuario que elija un PDF — con explorador o escribiendo la ruta."""
    print()
    if _TKINTER:
        print("  ¿Cómo querés elegir el archivo?")
        print("    1  Abrir explorador de archivos  ← recomendado")
        print("    2  Escribir la ruta manualmente")
        eleccion = _leer("\n  Opción: ")
        if eleccion != "2":
            print("\n  Abriendo explorador…")
            ruta = _abrir_explorador_archivo()
            if ruta:
                _ok(f"Archivo seleccionado: {Path(ruta).name}\n")
                return ruta
            _aviso("No se seleccionó ningún archivo.\n")
            return ""
    ruta = _leer("\n  Ruta del archivo PDF: ")
    return ruta


def _pedir_ruta_carpeta() -> str:
    """Pide al usuario que elija una carpeta — con explorador o escribiendo la ruta."""
    print()
    if _TKINTER:
        print("  ¿Cómo querés elegir la carpeta?")
        print("    1  Abrir explorador de carpetas  ← recomendado")
        print("    2  Escribir la ruta manualmente")
        eleccion = _leer("\n  Opción: ")
        if eleccion != "2":
            print("\n  Abriendo explorador…")
            ruta = _abrir_explorador_carpeta()
            if ruta:
                _ok(f"Carpeta seleccionada: {ruta}\n")
                return ruta
            _aviso("No se seleccionó ninguna carpeta.\n")
            return ""
    ruta = _leer("\n  Ruta de la carpeta: ")
    return ruta or settings.DATA_DIR


def _cargar_archivo(processor: DocumentProcessor, ruta: str) -> None:
    path = Path(ruta)
    if not path.exists():
        _error(f"Archivo no encontrado: {ruta}\n")
        return

    print(f"\n  Procesando: {path.name} …")
    try:
        doc = processor.process_document(ruta)
        print()
        _ok(f"Archivo  : {path.name}")
        _ok(f"Estado   : {doc.status.value}")
        _ok(f"Fragmentos generados: {doc.total_chunks}")
        if doc.processed_at:
            _ok(f"Finalizado: {doc.processed_at.strftime('%d/%m/%Y %H:%M:%S')}")
    except Exception as exc:
        _error(f"Error al procesar: {exc}")
        logger.exception("_cargar_archivo")
    print()


def _cargar_carpeta(pipeline: IngestionPipeline, carpeta: str) -> None:
    dir_path = Path(carpeta)
    if not dir_path.exists():
        _aviso(f"Carpeta no encontrada. Creando: {carpeta}")
        dir_path.mkdir(parents=True, exist_ok=True)
        print("  Agregá archivos PDF y volvé a intentarlo.\n")
        return

    print(f"\n  Procesando carpeta: {dir_path} …\n")
    try:
        resultado = pipeline.process_directory(
            directory_path=carpeta, recursive=True, continue_on_error=True
        )
        _ok(f"Archivos encontrados : {resultado.total_files}")
        _ok(f"Procesados con éxito : {resultado.successful}  ({resultado.success_rate:.0f} %)")
        if resultado.failed:
            _aviso(f"Con errores          : {resultado.failed}")
        _ok(f"Total de fragmentos  : {resultado.total_chunks}")

        if resultado.successful:
            print("\n  Archivos procesados:")
            for fp in resultado.get_successful_files():
                print(f"    ✓  {Path(fp).name}")
        if resultado.failed:
            print("\n  Archivos con error:")
            for fp in resultado.get_failed_files():
                print(f"    ✗  {Path(fp).name}")
    except Exception as exc:
        _error(f"Error en la carga masiva: {exc}")
        logger.exception("_cargar_carpeta")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Búsqueda de documentos
# ─────────────────────────────────────────────────────────────────────────────

_MODOS = {
    "1": ("estándar",  "Semántica estándar — rápida y precisa"),
    "2": ("mmr",       "MMR — resultados variados sin duplicados"),
    "3": ("expandida", "Expandida — mejor cobertura temática"),
}


def menu_busqueda(retriever: DocumentRetriever) -> None:
    """Bucle interactivo de búsqueda con selección de modo."""
    modo_actual = "expandida"

    while True:
        etiqueta_modo = next(v[1] for v in _MODOS.values() if v[0] == modo_actual)
        _titulo("BÚSQUEDA EN DOCUMENTOS")
        print(f"  Modo actual:  {etiqueta_modo}\n")
        print("  Comandos disponibles:")
        print("    modo      →  cambiar modo de búsqueda")
        print("    salir     →  volver al menú principal\n")

        consulta = _leer("  Consulta: ")

        if not consulta:
            continue
        if consulta.lower() == "salir":
            break
        if consulta.lower() == "modo":
            modo_actual = _seleccionar_modo(modo_actual)
            continue

        _ejecutar_busqueda(retriever, consulta, modo_actual)


def _seleccionar_modo(modo_actual: str) -> str:
    """Muestra un submenú para elegir el modo de búsqueda."""
    print()
    print("  Modos de búsqueda:")
    for key, (modo, descripcion) in _MODOS.items():
        marca = " ◀" if modo == modo_actual else ""
        print(f"    {key}  {descripcion}{marca}")
    print()
    eleccion = _leer("  Elegí un modo (1/2/3): ")
    if eleccion in _MODOS:
        nuevo = _MODOS[eleccion][0]
        _ok(f"Modo cambiado a: {_MODOS[eleccion][1]}\n")
        return nuevo
    _aviso("Opción no válida, se mantiene el modo actual.\n")
    return modo_actual


def _ejecutar_busqueda(retriever: DocumentRetriever, consulta: str, modo: str) -> None:
    """Realiza la búsqueda y muestra los resultados."""
    try:
        if modo == "mmr":
            resultados = retriever.search_mmr(query=consulta, top_k=5)
        elif modo == "expandida":
            resultados = retriever.search_expanded(query=consulta, top_k=5)
        else:
            resultados = retriever.search(query=consulta, top_k=5)
    except RetrieverException as exc:
        _error(str(exc))
        return

    print()
    if not resultados:
        _aviso("No se encontraron resultados. Verificá que haya documentos cargados.\n")
        return

    print(f"  Se encontraron {len(resultados)} resultado(s):\n")

    for i, r in enumerate(resultados, 1):
        doc = getattr(r, "document_name", None) or r.chunk.document_id
        relevancia = int(r.score * 100)
        print(f"  ┌─ Resultado {i}  {'─' * 30}")
        print(f"  │  Documento  : {doc}")
        print(f"  │  Relevancia : {relevancia}%  │  Fragmento #{r.chunk.chunk_index}")
        print(f"  │")
        # Texto a 56 caracteres por línea dentro del cuadro
        texto = r.chunk.content.strip()[:400]
        for linea in _partir_texto(texto, 54):
            print(f"  │  {linea}")
        print(f"  └{'─' * 47}")
        print()


def _partir_texto(texto: str, ancho: int) -> list[str]:
    """Divide un texto largo en líneas de ancho máximo `ancho`."""
    palabras = texto.split()
    lineas, actual = [], ""
    for palabra in palabras:
        if len(actual) + len(palabra) + 1 <= ancho:
            actual = f"{actual} {palabra}".lstrip()
        else:
            if actual:
                lineas.append(actual)
            actual = palabra
    if actual:
        lineas.append(actual)
    return lineas or [""]


# ─────────────────────────────────────────────────────────────────────────────
#  Chat con IA
# ─────────────────────────────────────────────────────────────────────────────

def menu_chat(retriever: DocumentRetriever) -> None:
    """Conecta con Ollama y abre el chat interactivo."""
    _titulo("CHAT CON IA")

    rag_service = _conectar_ollama(retriever)
    if rag_service is None:
        _leer("\n  Presioná ENTER para volver al menú… ")
        return

    _chat_interactivo(rag_service)


def _seleccionar_o_descargar_modelo(
    cliente: "OllamaClient", info: dict
) -> Optional[str]:
    """Permite elegir un modelo disponible o descargar uno nuevo.

    Retorna el nombre del modelo elegido, o None si el usuario cancela.
    """
    modelos: list = info.get("all_models") or []

    while True:
        print()
        _linea()
        print("  ¿Qué querés hacer?\n")

        for i, m in enumerate(modelos, start=1):
            print(f"    {i}.  Usar  '{m}'")

        n = len(modelos)
        print(f"    {n + 1}.  Descargar un modelo nuevo")
        print(f"    0.  Cancelar\n")

        opcion = _leer("  Opción: ").strip()

        if opcion == "0":
            return None

        if opcion == str(n + 1):
            nombre = _leer(
                "  Nombre del modelo (ej: llama3.2, mistral, phi3): "
            ).strip()
            if not nombre:
                _aviso("Nombre vacío — operación cancelada.")
                return None
            print(f"\n  Descargando '{nombre}'… esto puede tardar varios minutos.")
            print("  No cerrés la ventana.\n")
            try:
                cliente.pull_model(nombre)
                _ok(f"Modelo '{nombre}' descargado correctamente.")
                return nombre
            except Exception as exc:
                _error(f"No se pudo descargar '{nombre}': {exc}")
                return None

        try:
            idx = int(opcion) - 1
            if 0 <= idx < n:
                return modelos[idx]
        except ValueError:
            pass

        _aviso("Opción inválida, intentá de nuevo.")


def _conectar_ollama(retriever: DocumentRetriever) -> Optional[RAGService]:
    """Intenta conectarse a Ollama y construye el servicio RAG."""
    print(f"  Conectando con Ollama…")
    print(f"  URL    : {settings.OLLAMA_BASE_URL}\n")

    modelo_activo = settings.OLLAMA_MODEL
    cliente: Optional[OllamaClient] = None

    # ── loop: permite cambiar de modelo sin salir del menú ───────────────────
    while True:
        llm_cfg = LLMConfig(
            model_name=modelo_activo,
            temperature=settings.OLLAMA_TEMPERATURE,
            max_tokens=settings.OLLAMA_MAX_TOKENS,
            timeout=settings.OLLAMA_TIMEOUT,
        )
        cliente = OllamaClient(llm_cfg, base_url=settings.OLLAMA_BASE_URL)

        if not cliente.is_available():
            _error(f"No se pudo conectar con Ollama en {settings.OLLAMA_BASE_URL}")
            print(
                "\n  Para iniciarlo:\n"
                "    1. Instalá Ollama  →  https://ollama.ai\n"
                "    2. Ejecutá: ollama serve\n"
            )
            return None

        _ok("Ollama disponible")
        print(f"  Modelo configurado : {modelo_activo}\n")

        try:
            info = cliente.get_model_info()
        except Exception as exc:
            _aviso(f"No se pudo verificar el modelo: {exc}")
            info = {"available": True, "all_models": []}

        if not info.get("available"):
            disp = info.get("all_models") or []
            _error(f"El modelo '{modelo_activo}' no está descargado.")
            if disp:
                print(f"  Modelos disponibles: {', '.join(disp)}\n")
            else:
                print()

            modelo_nuevo = _seleccionar_o_descargar_modelo(cliente, info)
            if modelo_nuevo is None:
                return None
            modelo_activo = modelo_nuevo
            print()
            continue  # volver a verificar con el nuevo modelo

        _ok(f"Modelo '{modelo_activo}' listo")
        break  # todo OK, seguir

    rag_cfg = RAGConfig(
        top_k=settings.RAG_TOP_K,
        min_relevance=settings.RAG_MIN_RELEVANCE,
        max_context_length=settings.RAG_MAX_CONTEXT_LENGTH,
        include_sources=settings.RAG_INCLUDE_SOURCES,
        strict_mode=settings.RAG_STRICT_MODE,
        system_prompt=settings.RAG_SYSTEM_PROMPT,
    )
    servicio = RAGService(
        retriever=retriever,
        llm_client=cliente,
        config=rag_cfg,
        security_config=SecurityConfig(enabled=settings.RAG_ENABLE_SECURITY),
    )

    print("\n  Preparando modelo…")
    try:
        cliente.generate([Message(MessageRole.USER, "Hola")], max_tokens=5)
        _ok("Modelo calentado y listo\n")
    except Exception as exc:
        _aviso(f"No se pudo pre-calentar: {exc}\n")

    return servicio


def _chat_interactivo(rag_service: RAGService) -> None:
    """Bucle de conversación con la IA."""
    _titulo("CHAT ACTIVO")
    print("  Hacé tus preguntas sobre los documentos cargados.")
    print("  Comandos:")
    print("    borrar     →  limpiar el historial de conversación")
    print("    historial  →  ver mensajes anteriores")
    print("    salir      →  volver al menú principal\n")

    while True:
        pregunta = _leer("  Vos  : ")

        if not pregunta:
            continue

        if pregunta.lower() == "salir":
            print()
            break

        if pregunta.lower() == "borrar":
            rag_service.clear_conversation()
            _ok("Historial borrado.\n")
            continue

        if pregunta.lower() == "historial":
            mensajes = rag_service.get_conversation_history().get_messages(include_system=False)
            if not mensajes:
                print("  (el historial está vacío)\n")
            else:
                print()
                for m in mensajes:
                    quien = "Vos" if m.role.value == "user" else " IA"
                    print(f"  {quien}  :  {m.content[:120]}")
                print()
            continue

        try:
            print("  IA   : …\n")
            respuesta = rag_service.chat(pregunta)
            # Mostrar la respuesta en líneas cortas
            for linea in _partir_texto(respuesta.content, W - 10):
                print(f"  IA   :  {linea}")
            print()
            if respuesta.sources:
                print(f"  [ {len(respuesta.sources)} fuente(s) utilizada(s) ]")
                for i, src in enumerate(respuesta.sources, 1):
                    print(f"    {i}. {src.document_id}  —  relevancia {src.relevance_score:.0%}")
            print()
        except LLMConnectionError as exc:
            _error(f"Conexión perdida: {exc}\n")
        except Exception as exc:
            _error(f"{exc}\n")
            logger.exception("_chat_interactivo")


# ─────────────────────────────────────────────────────────────────────────────
#  Estadísticas
# ─────────────────────────────────────────────────────────────────────────────

def mostrar_estadisticas(vector_store: BaseVectorStore) -> None:
    """Muestra información sobre el contenido del vector store."""
    _titulo("ESTADÍSTICAS DEL SISTEMA")

    total = vector_store.count()
    print(f"  Fragmentos almacenados : {total}")
    print(f"  Dimensión de vectores  : {vector_store.dimension}")

    if total > 0 and hasattr(vector_store, "get_all_chunks"):
        try:
            fragmentos = vector_store.get_all_chunks()  # type: ignore[attr-defined]
            docs_unicos = {c.document_id for c in fragmentos}
            print(f"  Documentos distintos   : {len(docs_unicos)}\n")
            if docs_unicos:
                print("  Detalle por documento:")
                for did in sorted(docs_unicos):
                    n = sum(1 for c in fragmentos if c.document_id == did)
                    print(f"    • {did}  →  {n} fragmento(s)")
        except Exception:
            pass

    print()
    _leer("  Presioná ENTER para continuar… ")


# ─────────────────────────────────────────────────────────────────────────────
#  Menú principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Cabecera
    print()
    print("  ╔" + "═" * (W - 2) + "╗")
    print("  ║" + "CHATBOT RAG".center(W - 2) + "║")
    print("  ║" + "Sistema de preguntas sobre documentos".center(W - 2) + "║")
    print("  ╚" + "═" * (W - 2) + "╝")
    print()

    # Inicialización (siempre al arrancar)
    processor, retriever, pipeline, vector_store = inicializar()

    # Bucle del menú principal
    while True:
        docs_cargados = vector_store.count()
        estado = f"{docs_cargados} fragmento(s) en memoria" if docs_cargados else "sin documentos cargados"

        print(f"  ┌{'─' * (W - 2)}┐")
        print(f"  │{'MENÚ PRINCIPAL'.center(W - 2)}│")
        print(f"  │  Estado: {estado:<{W - 12}}│")
        print(f"  ├{'─' * (W - 2)}┤")
        print(f"  │  1  Cargar documentos                              │")
        print(f"  │  2  Buscar en documentos                           │")
        print(f"  │  3  Chat con IA  (requiere Ollama)                 │")
        print(f"  │  4  Ver estadísticas                               │")
        print(f"  │  0  Salir                                          │")
        print(f"  └{'─' * (W - 2)}┘")

        opcion = _leer("\n  Opción: ")

        if opcion == "1":
            menu_ingesta(processor, pipeline)
        elif opcion == "2":
            if docs_cargados == 0:
                print()
                _aviso("No hay documentos cargados. Cargá algunos primero (opción 1).\n")
            else:
                menu_busqueda(retriever)
        elif opcion == "3":
            if docs_cargados == 0:
                print()
                _aviso("No hay documentos cargados. Cargá algunos primero (opción 1).\n")
            else:
                menu_chat(retriever)
        elif opcion == "4":
            mostrar_estadisticas(vector_store)
        elif opcion == "0":
            print("\n  ¡Hasta luego!\n")
            break
        else:
            print()
            _aviso("Opción no válida. Ingresá 0, 1, 2, 3 o 4.\n")


if __name__ == "__main__":
    main()


