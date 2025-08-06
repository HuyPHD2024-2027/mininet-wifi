#!/usr/bin/env python3
"""
Mininet-WiFi Authority Component
Central authority for managing network policies and configurations
"""

import os
import sys
import logging
import json
import time
from flask import Flask, request, jsonify
from threading import Thread
import socket
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Authority:
    """Central authority for Mininet-WiFi network management"""
    
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.net = None
        self.clients = {}
        self.policies = {}
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes for API endpoints"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy', 'timestamp': time.time()})
        
        @self.app.route('/api/v1/network/status', methods=['GET'])
        def get_network_status():
            if self.net:
                nodes = [node.name for node in self.net.stations + self.net.aps]
                return jsonify({
                    'status': 'running',
                    'nodes': nodes,
                    'clients_connected': len(self.clients)
                })
            return jsonify({'status': 'stopped'})
        
        @self.app.route('/api/v1/network/start', methods=['POST'])
        def start_network():
            try:
                config = request.json or {}
                self.start_mininet_wifi(config)
                return jsonify({'status': 'success', 'message': 'Network started'})
            except Exception as e:
                logger.error(f"Failed to start network: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/v1/network/stop', methods=['POST'])
        def stop_network():
            try:
                if self.net:
                    self.net.stop()
                    self.net = None
                return jsonify({'status': 'success', 'message': 'Network stopped'})
            except Exception as e:
                logger.error(f"Failed to stop network: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/v1/clients/register', methods=['POST'])
        def register_client():
            try:
                client_data = request.json
                client_id = client_data.get('client_id')
                client_info = {
                    'id': client_id,
                    'ip': request.remote_addr,
                    'timestamp': time.time(),
                    'capabilities': client_data.get('capabilities', [])
                }
                self.clients[client_id] = client_info
                logger.info(f"Client {client_id} registered from {request.remote_addr}")
                return jsonify({'status': 'success', 'client_id': client_id})
            except Exception as e:
                logger.error(f"Failed to register client: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/v1/policies', methods=['GET'])
        def get_policies():
            return jsonify(self.policies)
        
        @self.app.route('/api/v1/policies', methods=['POST'])
        def set_policy():
            try:
                policy = request.json
                policy_id = policy.get('id')
                self.policies[policy_id] = policy
                logger.info(f"Policy {policy_id} set")
                return jsonify({'status': 'success', 'policy_id': policy_id})
            except Exception as e:
                logger.error(f"Failed to set policy: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def start_mininet_wifi(self, config=None):
        """Start Mininet-WiFi network"""
        try:
            self.net = Mininet_wifi()
            
            # Add stations
            sta1 = self.net.addStation('sta1', mac='00:00:00:00:00:02', 
                                     ip='10.0.0.1/8', position='30,60,0')
            sta2 = self.net.addStation('sta2', mac='00:00:00:00:00:03', 
                                     ip='10.0.0.2/8', position='70,30,0')
            
            # Add access point
            ap1 = self.net.addAccessPoint('ap1', ssid='authority-network', 
                                        mode='g', channel='1',
                                        failMode="standalone", position='50,50,0')
            
            # Add host
            h1 = self.net.addHost('h1', ip='10.0.0.3/8')
            
            # Set propagation model
            self.net.setPropagationModel(model="logDistance", exp=4.5)
            
            # Configure nodes
            self.net.configureNodes()
            
            # Create links
            self.net.addLink(ap1, h1)
            
            # Start network
            self.net.addNAT(linkTo='ap1').configDefault()
            self.net.build()
            ap1.start([])
            
            logger.info("Mininet-WiFi network started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Mininet-WiFi: {e}")
            raise
    
    def start_socket_server(self, host='127.0.0.1', port=12345):
        """Start socket server for legacy compatibility"""
        try:
            if self.net:
                self.net.socketServer(ip=host, port=port)
                logger.info(f"Socket server started on {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to start socket server: {e}")
    
    def run(self):
        """Run the authority server"""
        logger.info(f"Starting Authority server on {self.host}:{self.port}")
        
        # Start socket server in background
        if self.net:
            socket_thread = Thread(target=self.start_socket_server)
            socket_thread.daemon = True
            socket_thread.start()
        
        # Start Flask app
        self.app.run(host=self.host, port=self.port, debug=False)

def main():
    """Main entry point"""
    host = os.getenv('AUTHORITY_HOST', '0.0.0.0')
    port = int(os.getenv('AUTHORITY_PORT', 8080))
    
    authority = Authority(host=host, port=port)
    authority.run()

if __name__ == '__main__':
    main()