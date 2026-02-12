# ============================================================================
# Docker Cleanup Script for Windows PowerShell
# ChatBot RAG Project
# ============================================================================

# Colors
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }
function Write-Info { Write-Host $args -ForegroundColor Cyan }

Write-Info "🧹 ChatBot RAG - Docker Cleanup Script (Windows)"
Write-Info "==============================================="
Write-Host ""

# Confirmation
Write-Warning "⚠️  ADVERTENCIA: Este script eliminará:"
Write-Host "  - Contenedores del proyecto"
Write-Host "  - Imágenes del proyecto"
Write-Host "  - Volúmenes del proyecto (modelos y datos)"
Write-Host "  - Red del proyecto"
Write-Host ""

$confirmation = Read-Host "¿Estás seguro? (yes/no)"
if ($confirmation -ne "yes") {
    Write-Info "Operación cancelada"
    exit 0
}
Write-Host ""

# Stop and remove containers
Write-Info "🛑 Deteniendo contenedores..."
docker-compose down
if ($LASTEXITCODE -eq 0) {
    Write-Success "✓ Contenedores detenidos"
} else {
    Write-Warning "⚠ Algunos contenedores pueden no haberse detenido"
}
Write-Host ""

# Ask about volumes
$removeVolumes = Read-Host "¿Eliminar volúmenes (modelos y datos)? (yes/no)"
if ($removeVolumes -eq "yes") {
    Write-Info "🗑️  Eliminando volúmenes..."
    docker-compose down -v
    
    # Remove named volumes explicitly
    docker volume rm chatbot-ollama-models 2>$null
    docker volume rm chatbot-hf-models 2>$null
    docker volume rm chatbot-vectorstore-data 2>$null
    
    Write-Success "✓ Volúmenes eliminados"
} else {
    Write-Info "⏭️  Volúmenes conservados"
}
Write-Host ""

# Remove images
Write-Info "🗑️  Eliminando imágenes del proyecto..."
docker rmi chatbotproyecto-chatbot 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Success "✓ Imágenes eliminadas"
} else {
    Write-Warning "⚠ No se encontraron imágenes para eliminar"
}
Write-Host ""

# Remove network
Write-Info "🗑️  Eliminando red..."
docker network rm chatbot-network 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Success "✓ Red eliminada"
} else {
    Write-Warning "⚠ Red no encontrada o ya eliminada"
}
Write-Host ""

# Ask about system cleanup
$systemCleanup = Read-Host "¿Ejecutar limpieza general de Docker? (yes/no)"
if ($systemCleanup -eq "yes") {
    Write-Info "🧹 Ejecutando limpieza general..."
    docker system prune -f
    Write-Success "✓ Limpieza general completada"
}
Write-Host ""

Write-Success "✅ Cleanup completado!"
Write-Host ""
Write-Info "Para volver a empezar:"
Write-Host "  .\scripts\docker-setup.ps1"
Write-Host ""
