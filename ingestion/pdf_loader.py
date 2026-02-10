"""
PDF Loader module for ingesting PDF documents.
Handles PDF reading and text extraction.
Only responsible for reading PDFs, not managing state.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import logging

# Importaciones condicionales
PYPDF2_AVAILABLE = False
PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    if TYPE_CHECKING:
        import PyPDF2

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    if TYPE_CHECKING:
        import pdfplumber

logger = logging.getLogger(__name__)


class PDFLoaderException(Exception):
    """Exception raised for PDF loading errors"""
    pass


class PDFContent:
    """
    Representa el contenido extraído de un PDF.
    No conoce estados ni dominio.
    """
    def __init__(
        self,
        text: str,
        file_path: Path,
        page_count: int,
        metadata: Dict[str, Any]
    ):
        self.text = text
        self.file_path = file_path
        self.page_count = page_count
        self.metadata = metadata


class PDFLoader:
    """
    Carga y extrae texto de archivos PDF.
    Responsabilidad única: leer PDFs y extraer texto.
    """
    
    def __init__(self, backend: str = "pypdf2"):
        """
        Inicializa el PDF loader.
        
        Args:
            backend: Backend a usar ('pypdf2' o 'pdfplumber')
        """
        self.backend = backend.lower()
        self._validate_backend()
    
    def _validate_backend(self):
        """Valida que el backend esté disponible"""
        if self.backend == "pypdf2" and not PYPDF2_AVAILABLE:
            raise PDFLoaderException(
                "PyPDF2 no está instalado. Ejecuta: pip install PyPDF2"
            )
        elif self.backend == "pdfplumber" and not PDFPLUMBER_AVAILABLE:
            raise PDFLoaderException(
                "pdfplumber no está instalado. Ejecuta: pip install pdfplumber"
            )
        elif self.backend not in ["pypdf2", "pdfplumber"]:
            raise PDFLoaderException(
                f"Backend '{self.backend}' no soportado. Usa 'pypdf2' o 'pdfplumber'"
            )
    
    def load(self, file_path: Path) -> PDFContent:
        """
        Carga un PDF y extrae su contenido.
        
        Args:
            file_path: Ruta al archivo PDF
            
        Returns:
            PDFContent con el texto y metadata
            
        Raises:
            PDFLoaderException: Si hay error al leer el PDF
        """
        self._validate_file(file_path)
        
        try:
            if self.backend == "pypdf2":
                text, page_count = self._extract_with_pypdf2(file_path)
            else:  # pdfplumber
                text, page_count = self._extract_with_pdfplumber(file_path)
            
            metadata = {
                "backend": self.backend,
                "file_size": file_path.stat().st_size,
                "file_name": file_path.name
            }
            
            logger.info(f"PDF cargado: {file_path.name} ({len(text)} chars, {page_count} pages)")
            
            return PDFContent(
                text=text,
                file_path=file_path,
                page_count=page_count,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"Error al cargar PDF {file_path.name}: {str(e)}"
            logger.error(error_msg)
            raise PDFLoaderException(error_msg) from e
    
    def _validate_file(self, file_path: Path):
        """Valida que el archivo exista y sea PDF"""
        if not file_path.exists():
            raise PDFLoaderException(f"Archivo no encontrado: {file_path}")
        
        if file_path.suffix.lower() != ".pdf":
            raise PDFLoaderException(f"El archivo no es un PDF: {file_path}")
    
    def _extract_with_pypdf2(self, file_path: Path) -> tuple[str, int]:
        """Extrae texto usando PyPDF2"""
        if not PYPDF2_AVAILABLE:
            raise PDFLoaderException("PyPDF2 no está disponible")
        
        import PyPDF2  # Import local para asegurar disponibilidad
        
        text_parts: List[str] = []
        
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_count = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error en página {page_num + 1}: {e}")
                    continue
        
        return "\n\n".join(text_parts), page_count
    
    def _extract_with_pdfplumber(self, file_path: Path) -> tuple[str, int]:
        """Extrae texto usando pdfplumber"""
        if not PDFPLUMBER_AVAILABLE:
            raise PDFLoaderException("pdfplumber no está disponible")
        
        import pdfplumber  # Import local para asegurar disponibilidad
        
        text_parts: List[str] = []
        
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error en página {page_num + 1}: {e}")
                    continue
        
        return "\n\n".join(text_parts), page_count