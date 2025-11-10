#!/bin/bash

# Occupation Data Reports - Deployment Script
# This script provides various deployment options for the application

set -e  # Exit on any error

# Configuration
APP_NAME="occupation-data-reports"
VERSION="${VERSION:-1.0.0}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Occupation Data Reports - Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    local           Install locally using pip
    docker          Build and run Docker container
    docker-dev      Build and run development Docker container
    docker-compose  Deploy using Docker Compose
    clean           Clean up build artifacts and containers
    test            Run tests before deployment
    help            Show this help message

Options:
    --version VERSION    Set application version (default: $VERSION)
    --no-cache          Don't use Docker build cache
    --force             Force rebuild/reinstall
    --dev               Use development configuration
    --verbose           Enable verbose output

Examples:
    $0 local                    # Install locally
    $0 docker --no-cache        # Build Docker image without cache
    $0 docker-compose --dev     # Deploy with development profile
    $0 clean                    # Clean up everything
    $0 test                     # Run tests

Environment Variables:
    VERSION         Application version (default: $VERSION)
    BUILD_DATE      Build timestamp (auto-generated)
    VCS_REF         Git commit hash (auto-detected)
    LOG_LEVEL       Logging level (default: INFO)

EOF
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2)
    local major_version=$(echo $python_version | cut -d'.' -f1)
    local minor_version=$(echo $python_version | cut -d'.' -f2)
    
    if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 9 ]); then
        log_error "Python 3.9 or higher is required (found: $python_version)"
        exit 1
    fi
    
    log_success "Python $python_version found"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check Docker if needed
    if [[ "$1" == "docker"* ]]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker is required but not installed"
            exit 1
        fi
        
        if [[ "$1" == "docker-compose" ]]; then
            if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
                log_error "Docker Compose is required but not installed"
                exit 1
            fi
        fi
        
        log_success "Docker found"
    fi
}

# Local installation
deploy_local() {
    log_info "Starting local installation..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install dependencies
    log_info "Installing dependencies..."
    if [ "$FORCE" = true ]; then
        pip install --force-reinstall -r requirements.txt
    else
        pip install -r requirements.txt
    fi
    
    # Install application
    log_info "Installing application..."
    if [ "$DEV" = true ]; then
        pip install -e ".[dev,docs,all]"
    else
        pip install -e .
    fi
    
    # Create default configuration
    log_info "Creating default configuration..."
    python -m src.main create-config
    
    # Run basic validation
    log_info "Validating installation..."
    python -m src.main --help > /dev/null
    
    log_success "Local installation completed successfully!"
    log_info "To activate the environment, run: source venv/bin/activate"
    log_info "To test the installation, run: python -m src.main --help"
}

# Docker deployment
deploy_docker() {
    log_info "Starting Docker deployment..."
    
    local docker_args=""
    if [ "$NO_CACHE" = true ]; then
        docker_args="--no-cache"
    fi
    
    # Build Docker image
    log_info "Building Docker image..."
    docker build $docker_args \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$VCS_REF" \
        --target production \
        -t "$APP_NAME:$VERSION" \
        -t "$APP_NAME:latest" \
        .
    
    log_success "Docker image built successfully!"
    
    # Create necessary directories
    log_info "Creating data directories..."
    mkdir -p data reports logs config
    
    # Run container
    log_info "Running Docker container..."
    docker run -it --rm \
        --name "$APP_NAME-container" \
        -v "$(pwd)/data:/app/data:ro" \
        -v "$(pwd)/reports:/app/reports:rw" \
        -v "$(pwd)/logs:/app/logs:rw" \
        -v "$(pwd)/config:/app/config:rw" \
        -e LOG_LEVEL="${LOG_LEVEL:-INFO}" \
        "$APP_NAME:$VERSION" \
        python -m src.main --help
    
    log_success "Docker deployment completed!"
    log_info "To run reports, use: docker run -it --rm -v \$(pwd)/data:/app/data:ro -v \$(pwd)/reports:/app/reports:rw $APP_NAME:$VERSION python -m src.main --list-reports"
}

# Docker development deployment
deploy_docker_dev() {
    log_info "Starting Docker development deployment..."
    
    local docker_args=""
    if [ "$NO_CACHE" = true ]; then
        docker_args="--no-cache"
    fi
    
    # Build development Docker image
    log_info "Building development Docker image..."
    docker build $docker_args \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$VERSION-dev" \
        --build-arg VCS_REF="$VCS_REF" \
        --target development \
        -t "$APP_NAME:$VERSION-dev" \
        -t "$APP_NAME:dev" \
        .
    
    log_success "Development Docker image built successfully!"
    
    # Create necessary directories
    log_info "Creating data directories..."
    mkdir -p data reports logs config
    
    # Run development container
    log_info "Running development Docker container..."
    docker run -it --rm \
        --name "$APP_NAME-dev-container" \
        -v "$(pwd):/app:rw" \
        -v "$(pwd)/data:/app/data:ro" \
        -v "$(pwd)/reports:/app/reports:rw" \
        -v "$(pwd)/logs:/app/logs:rw" \
        -v "$(pwd)/config:/app/config:rw" \
        -e LOG_LEVEL="DEBUG" \
        "$APP_NAME:$VERSION-dev" \
        bash
    
    log_success "Development environment ready!"
}

# Docker Compose deployment
deploy_docker_compose() {
    log_info "Starting Docker Compose deployment..."
    
    # Set environment variables
    export BUILD_DATE="$BUILD_DATE"
    export VERSION="$VERSION"
    export VCS_REF="$VCS_REF"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    
    # Create necessary directories
    log_info "Creating data directories..."
    mkdir -p data reports logs config
    
    # Build and start services
    local compose_args=""
    if [ "$NO_CACHE" = true ]; then
        compose_args="--no-cache"
    fi
    
    if [ "$DEV" = true ]; then
        log_info "Starting development services..."
        docker-compose --profile development build $compose_args
        docker-compose --profile development up -d
        log_info "Development services started. Use 'docker-compose --profile development exec occupation-reports-dev bash' to access the development container."
    else
        log_info "Building and starting production services..."
        docker-compose build $compose_args
        docker-compose up -d
        log_info "Production services started. Use 'docker-compose exec occupation-reports python -m src.main --help' to run commands."
    fi
    
    log_success "Docker Compose deployment completed!"
    log_info "To view logs: docker-compose logs -f"
    log_info "To stop services: docker-compose down"
}

# Clean up
clean_deployment() {
    log_info "Cleaning up deployment artifacts..."
    
    # Stop and remove Docker containers
    if command -v docker &> /dev/null; then
        log_info "Stopping Docker containers..."
        docker stop "$APP_NAME-container" "$APP_NAME-dev-container" 2>/dev/null || true
        docker rm "$APP_NAME-container" "$APP_NAME-dev-container" 2>/dev/null || true
        
        # Remove Docker images
        if [ "$FORCE" = true ]; then
            log_info "Removing Docker images..."
            docker rmi "$APP_NAME:$VERSION" "$APP_NAME:latest" "$APP_NAME:$VERSION-dev" "$APP_NAME:dev" 2>/dev/null || true
        fi
        
        # Clean up Docker Compose
        if [ -f "docker-compose.yml" ]; then
            log_info "Cleaning up Docker Compose..."
            docker-compose down --volumes --remove-orphans 2>/dev/null || true
        fi
    fi
    
    # Clean up Python artifacts
    log_info "Cleaning up Python artifacts..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    rm -rf build/ dist/ .pytest_cache/ .coverage 2>/dev/null || true
    
    # Clean up virtual environment if force is specified
    if [ "$FORCE" = true ] && [ -d "venv" ]; then
        log_info "Removing virtual environment..."
        rm -rf venv/
    fi
    
    log_success "Cleanup completed!"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Check if virtual environment exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Install test dependencies if needed
    if ! python -c "import pytest" 2>/dev/null; then
        log_info "Installing test dependencies..."
        pip install pytest pytest-cov pytest-mock
    fi
    
    # Run tests
    log_info "Executing test suite..."
    python -m pytest tests/ -v --cov=src --cov-report=term-missing
    
    log_success "Tests completed!"
}

# Parse command line arguments
COMMAND=""
NO_CACHE=false
FORCE=false
DEV=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        local|docker|docker-dev|docker-compose|clean|test|help)
            COMMAND="$1"
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dev)
            DEV=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            set -x
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "Occupation Data Reports Deployment Script"
    log_info "Version: $VERSION"
    log_info "Build Date: $BUILD_DATE"
    log_info "VCS Ref: $VCS_REF"
    echo
    
    case $COMMAND in
        local)
            check_prerequisites "local"
            deploy_local
            ;;
        docker)
            check_prerequisites "docker"
            deploy_docker
            ;;
        docker-dev)
            check_prerequisites "docker"
            deploy_docker_dev
            ;;
        docker-compose)
            check_prerequisites "docker-compose"
            deploy_docker_compose
            ;;
        clean)
            clean_deployment
            ;;
        test)
            check_prerequisites "local"
            run_tests
            ;;
        help|"")
            show_help
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main