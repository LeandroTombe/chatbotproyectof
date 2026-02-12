# ============================================================================
# Docker Setup Script for Windows PowerShell
# ChatBot RAG Project
# ============================================================================

# Colors
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }
function Write-Info { Write-Host $args -ForegroundColor Cyan }

Write-Info "ChatBot RAG - Docker Setup Script (Windows)"
Write-Info "=============================================="
Write-Host ""

# Check if Docker is running
try {
    $dockerVersion = docker --version 2>&1
    Write-Success "Docker instalado: $dockerVersion"
} catch {
    Write-Error "Docker no esta instalado o no esta corriendo"
    Write-Host "Por favor, instala Docker Desktop desde: https://docs.docker.com/desktop/install/windows-install/"
    exit 1
}

# Check if Docker Compose is available
try {
    $composeVersion = docker-compose --version 2>&1
    Write-Success "Docker Compose instalado: $composeVersion"
} catch {
    Write-Error "Docker Compose no esta disponible"
    exit 1
}
Write-Host ""

# Create .env file if it doesn't exist
if (-not (Test-Path .env)) {
    Write-Warning "Archivo .env no encontrado"
    if (Test-Path .env.docker) {
        Write-Info "Copiando .env.docker a .env..."
        Copy-Item .env.docker .env
        Write-Success "Archivo .env creado"
    } else {
        Write-Error "No se encuentra .env.docker"
        exit 1
    }
} else {
    Write-Success "Archivo .env existe"
}
Write-Host ""

# Create necessary directories
Write-Info "Creando directorios necesarios..."
$directories = @("data", "logs", "documents", "models", "vectorstore_data")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Success "  Creado: $dir"
    } else {
        Write-Success "  Existe: $dir"
    }
}
Write-Host ""

# Build Docker images
Write-Info "Construyendo imagenes Docker..."
Write-Host "Esto puede tardar varios minutos la primera vez..."
docker-compose build
if ($LASTEXITCODE -eq 0) {
    Write-Success "Imagenes construidas exitosamente"
} else {
    Write-Error "Error construyendo imagenes"
    exit 1
}
Write-Host ""

# Start services
Write-Info "Iniciando servicios..."
docker-compose up -d
if ($LASTEXITCODE -eq 0) {
    Write-Success "Servicios iniciados"
} else {
    Write-Error "Error iniciando servicios"
    exit 1
}
Write-Host ""

# Wait for Ollama to be healthy
Write-Info "Esperando que Ollama este listo..."
$maxAttempts = 30
$attempt = 0
$ready = $false

while ($attempt -lt $maxAttempts -and -not $ready) {
    try {
        docker-compose exec -T ollama ollama list 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $ready = $true
            Write-Success "Ollama esta listo"
        } else {
            $attempt++
            Write-Host "  Intento $attempt/$maxAttempts..."
            Start-Sleep -Seconds 2
        }
    } catch {
        $attempt++
        Write-Host "  Intento $attempt/$maxAttempts..."
        Start-Sleep -Seconds 2
    }
}

if (-not $ready) {
    Write-Error "Timeout esperando a Ollama"
    Write-Host "Ver logs con: docker-compose logs ollama"
    exit 1
}
Write-Host ""

# Download Ollama model
Write-Info "Descargando modelo de Ollama (llama3.2)..."
Write-Host "Esto puede tardar varios minutos dependiendo de tu conexion..."
docker-compose exec -T ollama ollama pull llama3.2
if ($LASTEXITCODE -eq 0) {
    Write-Success "Modelo llama3.2 descargado"
} else {
    Write-Warning "No se pudo descargar llama3.2 automaticamente"
    Write-Host "Puedes descargarlo manualmente despues con:"
    Write-Host "  docker-compose exec ollama ollama pull llama3.2"
}
Write-Host ""

# Show status
Write-Info "Estado de los servicios:"
docker-compose ps
Write-Host ""

# Show next steps
Write-Success "Setup completado!"
Write-Host ""
Write-Info "Proximos pasos:"
Write-Host "  1. Ver logs:          docker-compose logs -f"
Write-Host "  2. Ejecutar chatbot:  docker-compose exec chatbot python main.py"
Write-Host "  3. Ejecutar tests:    docker-compose run --rm chatbot python -m pytest"
Write-Host "  4. Acceder shell:     docker-compose exec chatbot bash"
Write-Host ""
Write-Info "Para mas informacion, ver:"
Write-Host "  - README.docker.md          (Documentacion completa)"
Write-Host "  - README.docker.windows.md  (Guia especifica Windows)"
Write-Host ""
Write-Success "Listo para usar!"
