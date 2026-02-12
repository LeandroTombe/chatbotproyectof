#!/bin/bash
# ============================================================================
# Docker Setup Script for ChatBot RAG Project
# ============================================================================

set -e  # Exit on error

echo "🚀 ChatBot RAG - Docker Setup Script"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker no está instalado${NC}"
    echo "Por favor, instala Docker desde: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose no está instalado${NC}"
    echo "Por favor, instala Docker Compose desde: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✓ Docker instalado: $(docker --version)${NC}"
echo -e "${GREEN}✓ Docker Compose instalado: $(docker-compose --version)${NC}"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠ Archivo .env no encontrado${NC}"
    if [ -f .env.docker ]; then
        echo "Copiando .env.docker a .env..."
        cp .env.docker .env
        echo -e "${GREEN}✓ Archivo .env creado${NC}"
    else
        echo -e "${RED}❌ No se encuentra .env.docker${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Archivo .env existe${NC}"
fi
echo ""

# Create necessary directories
echo "📁 Creando directorios necesarios..."
mkdir -p data logs documents models vectorstore_data
echo -e "${GREEN}✓ Directorios creados${NC}"
echo ""

# Build Docker images
echo "🔨 Construyendo imágenes Docker..."
docker-compose build
echo -e "${GREEN}✓ Imágenes construidas${NC}"
echo ""

# Start services
echo "🚀 Iniciando servicios..."
docker-compose up -d
echo -e "${GREEN}✓ Servicios iniciados${NC}"
echo ""

# Wait for Ollama to be healthy
echo "⏳ Esperando que Ollama esté listo..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker-compose exec -T ollama ollama list &> /dev/null; then
        echo -e "${GREEN}✓ Ollama está listo${NC}"
        break
    fi
    attempt=$((attempt + 1))
    echo "Intento $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}❌ Timeout esperando a Ollama${NC}"
    echo "Ver logs: docker-compose logs ollama"
    exit 1
fi
echo ""

# Download default Ollama model
echo "📥 Descargando modelo de Ollama (llama3.2)..."
echo "Esto puede tardar varios minutos..."
if docker-compose exec -T ollama ollama pull llama3.2; then
    echo -e "${GREEN}✓ Modelo llama3.2 descargado${NC}"
else
    echo -e "${YELLOW}⚠ No se pudo descargar llama3.2${NC}"
    echo "Puedes descargarlo manualmente después con:"
    echo "  docker-compose exec ollama ollama pull llama3.2"
fi
echo ""

# Show status
echo "📊 Estado de los servicios:"
docker-compose ps
echo ""

# Show next steps
echo -e "${GREEN}✅ Setup completado!${NC}"
echo ""
echo "Próximos pasos:"
echo "  1. Ver logs:          docker-compose logs -f"
echo "  2. Ejecutar chatbot:  docker-compose exec chatbot python main.py"
echo "  3. Ejecutar tests:    docker-compose run --rm chatbot python -m pytest"
echo "  4. Acceder shell:     docker-compose exec chatbot bash"
echo ""
echo "Para más información, ver README.docker.md"
echo ""
