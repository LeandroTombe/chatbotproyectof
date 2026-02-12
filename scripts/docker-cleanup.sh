#!/bin/bash
# ============================================================================
# Docker Cleanup Script for ChatBot RAG Project
# ============================================================================

set -e

echo "🧹 ChatBot RAG - Docker Cleanup Script"
echo "======================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Ask for confirmation
echo -e "${YELLOW}⚠️  ADVERTENCIA: Este script eliminará:${NC}"
echo "  - Contenedores del proyecto"
echo "  - Imágenes del proyecto"
echo "  - Volúmenes del proyecto (modelos y datos)"
echo "  - Red del proyecto"
echo ""
read -p "¿Estás seguro? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Operación cancelada"
    exit 0
fi

# Stop and remove containers
echo "🛑 Deteniendo contenedores..."
docker-compose down
echo -e "${GREEN}✓ Contenedores detenidos${NC}"
echo ""

# Ask about volumes
read -p "¿Eliminar volúmenes (modelos y datos)? (yes/no): " -r
echo ""
if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "🗑️  Eliminando volúmenes..."
    docker-compose down -v
    docker volume rm chatbot-ollama-models chatbot-hf-models chatbot-vectorstore-data 2>/dev/null || true
    echo -e "${GREEN}✓ Volúmenes eliminados${NC}"
else
    echo "⏭️  Volúmenes conservados"
fi
echo ""

# Remove images
echo "🗑️  Eliminando imágenes del proyecto..."
docker rmi chatbotproyecto-chatbot 2>/dev/null || true
echo -e "${GREEN}✓ Imágenes eliminadas${NC}"
echo ""

# Remove network
echo "🗑️  Eliminando red..."
docker network rm chatbot-network 2>/dev/null || true
echo -e "${GREEN}✓ Red eliminada${NC}"
echo ""

# Ask about system cleanup
read -p "¿Ejecutar limpieza general de Docker? (yes/no): " -r
echo ""
if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "🧹 Ejecutando limpieza general..."
    docker system prune -f
    echo -e "${GREEN}✓ Limpieza general completada${NC}"
fi
echo ""

echo -e "${GREEN}✅ Cleanup completado!${NC}"
echo ""
echo "Para volver a empezar:"
echo "  ./scripts/docker-setup.sh"
echo ""
