#!/bin/bash
# Deployment script for Mininet-WiFi Authority and Client components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AUTHORITY_IMAGE="mininet-wifi-authority:latest"
CLIENT_IMAGE="mininet-wifi-client:latest"
REGISTRY_URL=""
COMPOSE_FILE="docker-compose.yml"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are available"
}

# Function to build images
build_images() {
    print_status "Building Docker images..."
    
    # Build authority image
    print_status "Building authority image..."
    docker build -f Dockerfile.authority -t $AUTHORITY_IMAGE .
    
    # Build client image
    print_status "Building client image..."
    docker build -f Dockerfile.client -t $CLIENT_IMAGE .
    
    print_success "Images built successfully"
}

# Function to push images to registry
push_images() {
    if [ -z "$REGISTRY_URL" ]; then
        print_warning "No registry URL specified, skipping image push"
        return
    fi
    
    print_status "Pushing images to registry..."
    
    # Tag images for registry
    docker tag $AUTHORITY_IMAGE $REGISTRY_URL/$AUTHORITY_IMAGE
    docker tag $CLIENT_IMAGE $REGISTRY_URL/$CLIENT_IMAGE
    
    # Push images
    docker push $REGISTRY_URL/$AUTHORITY_IMAGE
    docker push $REGISTRY_URL/$CLIENT_IMAGE
    
    print_success "Images pushed to registry"
}

# Function to deploy using Docker Compose
deploy_compose() {
    print_status "Deploying with Docker Compose..."
    
    # Create necessary directories
    mkdir -p config logs
    
    # Start services
    docker-compose -f $COMPOSE_FILE up -d
    
    print_success "Services deployed successfully"
    print_status "Authority available at: http://localhost:8080"
    print_status "Client available at: http://localhost:8081"
}

# Function to deploy authority only
deploy_authority() {
    print_status "Deploying authority only..."
    
    docker run -d \
        --name mininet-wifi-authority \
        --privileged \
        --cap-add=NET_ADMIN \
        --cap-add=SYS_ADMIN \
        -p 8080:8080 \
        -p 12345:12345 \
        -v $(pwd)/config:/app/config:ro \
        -v $(pwd)/logs:/app/logs \
        -e AUTHORITY_HOST=0.0.0.0 \
        -e AUTHORITY_PORT=8080 \
        -e LOG_LEVEL=INFO \
        $AUTHORITY_IMAGE
    
    print_success "Authority deployed successfully"
    print_status "Authority available at: http://localhost:8080"
}

# Function to deploy client only
deploy_client() {
    local authority_url=${1:-"http://localhost:8080"}
    
    print_status "Deploying client only..."
    print_status "Connecting to authority at: $authority_url"
    
    docker run -d \
        --name mininet-wifi-client \
        --privileged \
        --cap-add=NET_ADMIN \
        --cap-add=SYS_ADMIN \
        -p 8081:8081 \
        -v $(pwd)/config:/app/config:ro \
        -v $(pwd)/logs:/app/logs \
        -e CLIENT_HOST=0.0.0.0 \
        -e CLIENT_PORT=8081 \
        -e AUTHORITY_URL=$authority_url \
        -e LOG_LEVEL=INFO \
        $CLIENT_IMAGE
    
    print_success "Client deployed successfully"
    print_status "Client available at: http://localhost:8081"
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    
    if [ -f "$COMPOSE_FILE" ]; then
        docker-compose -f $COMPOSE_FILE down
    else
        docker stop mininet-wifi-authority mininet-wifi-client 2>/dev/null || true
        docker rm mininet-wifi-authority mininet-wifi-client 2>/dev/null || true
    fi
    
    print_success "Services stopped"
}

# Function to show logs
show_logs() {
    local service=${1:-"all"}
    
    case $service in
        "authority")
            docker logs -f mininet-wifi-authority
            ;;
        "client")
            docker logs -f mininet-wifi-client
            ;;
        "all")
            docker-compose -f $COMPOSE_FILE logs -f
            ;;
        *)
            print_error "Invalid service: $service. Use 'authority', 'client', or 'all'"
            exit 1
            ;;
    esac
}

# Function to show status
show_status() {
    print_status "Service Status:"
    echo "=================="
    
    if [ -f "$COMPOSE_FILE" ]; then
        docker-compose -f $COMPOSE_FILE ps
    else
        docker ps --filter "name=mininet-wifi"
    fi
    
    echo ""
    print_status "Network Status:"
    echo "=================="
    curl -s http://localhost:8080/health 2>/dev/null || echo "Authority not accessible"
    curl -s http://localhost:8081/health 2>/dev/null || echo "Client not accessible"
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build           Build Docker images"
    echo "  deploy          Deploy all services using Docker Compose"
    echo "  deploy-auth     Deploy authority only"
    echo "  deploy-client   Deploy client only (requires authority URL)"
    echo "  stop            Stop all services"
    echo "  logs [SERVICE]  Show logs (authority, client, or all)"
    echo "  status          Show service status"
    echo "  push            Push images to registry"
    echo "  help            Show this help message"
    echo ""
    echo "Options:"
    echo "  --registry URL  Set registry URL for pushing images"
    echo "  --auth-url URL  Set authority URL for client deployment"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 deploy"
    echo "  $0 deploy-client --auth-url http://192.168.1.100:8080"
    echo "  $0 logs authority"
}

# Parse command line arguments
COMMAND=""
AUTHORITY_URL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        build|deploy|deploy-auth|deploy-client|stop|logs|status|push|help)
            COMMAND="$1"
            shift
            ;;
        --registry)
            REGISTRY_URL="$2"
            shift 2
            ;;
        --auth-url)
            AUTHORITY_URL="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if command is provided
if [ -z "$COMMAND" ]; then
    print_error "No command specified"
    show_help
    exit 1
fi

# Execute command
case $COMMAND in
    "build")
        check_docker
        build_images
        ;;
    "deploy")
        check_docker
        build_images
        deploy_compose
        ;;
    "deploy-auth")
        check_docker
        build_images
        deploy_authority
        ;;
    "deploy-client")
        check_docker
        build_images
        deploy_client "$AUTHORITY_URL"
        ;;
    "stop")
        stop_services
        ;;
    "logs")
        show_logs "$1"
        ;;
    "status")
        show_status
        ;;
    "push")
        check_docker
        push_images
        ;;
    "help")
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac