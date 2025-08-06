#!/usr/bin/env python3
"""
Mininet-WiFi Client Component
Client that connects to the authority for network operations
"""

import os
import sys
import logging
import json
import time
import requests
import socket
from flask import Flask, request, jsonify
from threading import Thread
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Client:
    """Client component for connecting to Mininet-WiFi authority"""
    
    def __init__(self, authority_url='http://localhost:8080', host='0.0.0.0', port=8081):
        self.authority_url = authority_url
        self.host = host
        self.port = port
        self.client_id = str(uuid.uuid4())
        self.app = Flask(__name__)
        self.connected = False
        self.capabilities = ['network_control', 'policy_enforcement', 'monitoring']
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes for client API endpoints"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy', 
                'client_id': self.client_id,
                'connected_to_authority': self.connected,
                'timestamp': time.time()
            })
        
        @self.app.route('/api/v1/connect', methods=['POST'])
        def connect_to_authority():
            try:
                success = self.register_with_authority()
                if success:
                    return jsonify({'status': 'success', 'message': 'Connected to authority'})
                else:
                    return jsonify({'status': 'error', 'message': 'Failed to connect to authority'}), 500
            except Exception as e:
                logger.error(f"Failed to connect: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/v1/network/status', methods=['GET'])
        def get_network_status():
            try:
                response = requests.get(f"{self.authority_url}/api/v1/network/status")
                return jsonify(response.json())
            except Exception as e:
                logger.error(f"Failed to get network status: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/v1/network/start', methods=['POST'])
        def start_network():
            try:
                config = request.json or {}
                response = requests.post(f"{self.authority_url}/api/v1/network/start", json=config)
                return jsonify(response.json())
            except Exception as e:
                logger.error(f"Failed to start network: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/v1/network/stop', methods=['POST'])
        def stop_network():
            try:
                response = requests.post(f"{self.authority_url}/api/v1/network/stop")
                return jsonify(response.json())
            except Exception as e:
                logger.error(f"Failed to stop network: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/v1/policies', methods=['GET'])
        def get_policies():
            try:
                response = requests.get(f"{self.authority_url}/api/v1/policies")
                return jsonify(response.json())
            except Exception as e:
                logger.error(f"Failed to get policies: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/v1/policies', methods=['POST'])
        def set_policy():
            try:
                policy = request.json
                response = requests.post(f"{self.authority_url}/api/v1/policies", json=policy)
                return jsonify(response.json())
            except Exception as e:
                logger.error(f"Failed to set policy: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def register_with_authority(self):
        """Register this client with the authority"""
        try:
            client_data = {
                'client_id': self.client_id,
                'capabilities': self.capabilities,
                'host': self.host,
                'port': self.port
            }
            
            response = requests.post(f"{self.authority_url}/api/v1/clients/register", json=client_data)
            
            if response.status_code == 200:
                self.connected = True
                logger.info(f"Successfully registered with authority as {self.client_id}")
                return True
            else:
                logger.error(f"Failed to register with authority: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to register with authority: {e}")
            return False
    
    def socket_client(self, host='127.0.0.1', port=12345):
        """Legacy socket client for compatibility"""
        try:
            s = socket.socket()
            s.connect((host, port))
            logger.info(f"Socket client connected to {host}:{port}")
            
            while True:
                message = input('-> ')
                if message.lower() in ['q', 'exit', 'quit']:
                    break
                    
                s.send(str(message).encode('utf-8'))
                data = s.recv(1024).decode('utf-8')
                print(f'Received from server: {data}')
                
            s.close()
            
        except Exception as e:
            logger.error(f"Socket client error: {e}")
    
    def run(self):
        """Run the client server"""
        logger.info(f"Starting Client server on {self.host}:{self.port}")
        logger.info(f"Client ID: {self.client_id}")
        
        # Try to connect to authority
        if self.register_with_authority():
            logger.info("Successfully connected to authority")
        else:
            logger.warning("Failed to connect to authority, will retry later")
        
        # Start Flask app
        self.app.run(host=self.host, port=self.port, debug=False)

def main():
    """Main entry point"""
    authority_url = os.getenv('AUTHORITY_URL', 'http://localhost:8080')
    host = os.getenv('CLIENT_HOST', '0.0.0.0')
    port = int(os.getenv('CLIENT_PORT', 8081))
    
    client = Client(authority_url=authority_url, host=host, port=port)
    client.run()

if __name__ == '__main__':
    main()