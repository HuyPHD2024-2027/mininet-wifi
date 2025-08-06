# Mininet-WiFi Authority and Client Deployment Guide

This guide explains how to deploy the Mininet-WiFi Authority and Client components on different machines and devices.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Docker Deployment](#docker-deployment)
4. [Python Package Deployment](#python-package-deployment)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Troubleshooting](#troubleshooting)

## Overview

The Mininet-WiFi project now includes exportable Authority and Client components that can be deployed on different machines:

- **Authority**: Central management component that controls the Mininet-WiFi network
- **Client**: Distributed component that connects to the Authority for network operations

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 16.04+ recommended)
- **Python**: 3.9 or higher
- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+ (for orchestrated deployment)

### Network Requirements

- Network connectivity between Authority and Client machines
- Port 8080 (Authority API)
- Port 8081 (Client API)
- Port 12345 (Legacy socket communication)

## Docker Deployment

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/intrig-unicamp/mininet-wifi
   cd mininet-wifi
   ```

2. **Deploy all components**:
   ```bash
   ./deploy.sh deploy
   ```

3. **Check status**:
   ```bash
   ./deploy.sh status
   ```

### Individual Component Deployment

#### Deploy Authority Only

```bash
# On the authority machine
./deploy.sh deploy-auth
```

#### Deploy Client Only

```bash
# On the client machine
./deploy.sh deploy-client --auth-url http://AUTHORITY_IP:8080
```

### Docker Compose Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Custom Registry Deployment

```bash
# Build and push to registry
./deploy.sh build
./deploy.sh push --registry your-registry.com

# Deploy from registry
docker-compose -f docker-compose.yml up -d
```

## Python Package Deployment

### Install from Source

#### Authority Component

```bash
# Install authority component
pip install -e . -f setup_authority.py

# Run authority
mininet-wifi-authority
```

#### Client Component

```bash
# Install client component
pip install -e . -f setup_client.py

# Run client
mininet-wifi-client
```

### Install from PyPI (Future)

```bash
# Install authority
pip install mininet-wifi-authority

# Install client
pip install mininet-wifi-client
```

## Configuration

### Environment Variables

#### Authority Environment Variables

```bash
export AUTHORITY_HOST="0.0.0.0"
export AUTHORITY_PORT="8080"
export LOG_LEVEL="INFO"
export JWT_SECRET="your-secret-key"
```

#### Client Environment Variables

```bash
export CLIENT_HOST="0.0.0.0"
export CLIENT_PORT="8081"
export AUTHORITY_URL="http://AUTHORITY_IP:8080"
export LOG_LEVEL="INFO"
```

### Configuration Files

Configuration files are located in the `config/` directory:

- `config/authority_config.yaml` - Authority configuration
- `config/client_config.yaml` - Client configuration

### Custom Configuration

Create custom configuration files:

```bash
# Copy default configs
cp config/authority_config.yaml config/my_authority_config.yaml
cp config/client_config.yaml config/my_client_config.yaml

# Edit configurations
nano config/my_authority_config.yaml
nano config/my_client_config.yaml

# Use custom configs
AUTHORITY_CONFIG=config/my_authority_config.yaml mininet-wifi-authority
CLIENT_CONFIG=config/my_client_config.yaml mininet-wifi-client
```

## Usage Examples

### Basic Network Operations

#### Start Network via Authority

```bash
# Using curl
curl -X POST http://localhost:8080/api/v1/network/start \
  -H "Content-Type: application/json" \
  -d '{"ssid": "my-network", "channel": 6}'

# Using Python
import requests
response = requests.post('http://localhost:8080/api/v1/network/start', 
                        json={'ssid': 'my-network', 'channel': 6})
print(response.json())
```

#### Check Network Status

```bash
# Using curl
curl http://localhost:8080/api/v1/network/status

# Using Python
import requests
response = requests.get('http://localhost:8080/api/v1/network/status')
print(response.json())
```

#### Register Client

```bash
# Using curl
curl -X POST http://localhost:8080/api/v1/clients/register \
  -H "Content-Type: application/json" \
  -d '{"client_id": "client-001", "capabilities": ["monitoring"]}'
```

### Advanced Operations

#### Set Network Policies

```bash
# Define policy
policy = {
    "id": "bandwidth_limit",
    "type": "bandwidth_control",
    "parameters": {
        "max_bandwidth": "10Mbps",
        "stations": ["sta1", "sta2"]
    }
}

# Apply policy
curl -X POST http://localhost:8080/api/v1/policies \
  -H "Content-Type: application/json" \
  -d "$(echo $policy | jq -c .)"
```

#### Monitor Network Performance

```bash
# Get network metrics
curl http://localhost:8080/api/v1/metrics/network

# Get client statistics
curl http://localhost:8080/api/v1/metrics/clients
```

### Multi-Machine Deployment

#### Scenario: Authority on Server, Clients on Multiple Devices

1. **Deploy Authority on Server**:
   ```bash
   # On server (192.168.1.100)
   ./deploy.sh deploy-auth
   ```

2. **Deploy Clients on Devices**:
   ```bash
   # On device 1 (192.168.1.101)
   ./deploy.sh deploy-client --auth-url http://192.168.1.100:8080
   
   # On device 2 (192.168.1.102)
   ./deploy.sh deploy-client --auth-url http://192.168.1.100:8080
   ```

3. **Verify Connectivity**:
   ```bash
   # Check authority status
   curl http://192.168.1.100:8080/health
   
   # Check client status
   curl http://192.168.1.101:8081/health
   curl http://192.168.1.102:8081/health
   ```

## Troubleshooting

### Common Issues

#### Docker Permission Issues

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo (not recommended for production)
sudo docker-compose up -d
```

#### Network Connectivity Issues

```bash
# Check if ports are open
netstat -tulpn | grep :8080
netstat -tulpn | grep :8081

# Check firewall settings
sudo ufw status
sudo ufw allow 8080
sudo ufw allow 8081
```

#### Mininet-WiFi Dependencies

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y git make help2man pyflakes3 python3-pep8 tcpdump wpan-tools

# Run Mininet-WiFi installation
sudo util/install.sh -Wlnfv
```

### Logs and Debugging

#### View Docker Logs

```bash
# View all logs
./deploy.sh logs all

# View specific service logs
./deploy.sh logs authority
./deploy.sh logs client

# Follow logs in real-time
docker-compose logs -f
```

#### Debug Mode

```bash
# Run in debug mode
AUTHORITY_DEBUG=true ./deploy.sh deploy-auth
CLIENT_DEBUG=true ./deploy.sh deploy-client --auth-url http://AUTHORITY_IP:8080
```

### Health Checks

#### Manual Health Checks

```bash
# Check authority health
curl http://localhost:8080/health

# Check client health
curl http://localhost:8081/health

# Check network status
curl http://localhost:8080/api/v1/network/status
```

#### Automated Health Monitoring

```bash
# Create health check script
cat > health_check.sh << 'EOF'
#!/bin/bash
AUTHORITY_URL=${1:-"http://localhost:8080"}
CLIENT_URL=${2:-"http://localhost:8081"}

echo "Checking Authority..."
curl -s $AUTHORITY_URL/health | jq .

echo "Checking Client..."
curl -s $CLIENT_URL/health | jq .

echo "Checking Network Status..."
curl -s $AUTHORITY_URL/api/v1/network/status | jq .
EOF

chmod +x health_check.sh
./health_check.sh
```

## Security Considerations

### Production Deployment

1. **Use HTTPS**:
   ```bash
   # Enable SSL in configuration
   security:
     enable_ssl: true
     ssl_cert: "/path/to/cert.pem"
     ssl_key: "/path/to/key.pem"
   ```

2. **Authentication**:
   ```bash
   # Enable JWT authentication
   security:
     enable_auth: true
     jwt_secret: "your-secure-secret-key"
   ```

3. **Network Security**:
   ```bash
   # Restrict access to specific IPs
   api:
     cors_origins: ["https://your-domain.com"]
   ```

### Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw allow 8080/tcp  # Authority API
sudo ufw allow 8081/tcp  # Client API
sudo ufw allow 12345/tcp # Legacy socket
sudo ufw enable
```

## Performance Optimization

### Resource Limits

```yaml
# In docker-compose.yml
services:
  authority:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### Scaling

```bash
# Scale client instances
docker-compose up -d --scale client=3
```

## Support and Community

- **Documentation**: https://mininet-wifi.github.io/
- **GitHub**: https://github.com/intrig-unicamp/mininet-wifi
- **Mailing List**: https://groups.google.com/forum/#!forum/mininet-wifi-discuss
- **Issues**: https://github.com/intrig-unicamp/mininet-wifi/issues