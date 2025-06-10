#!/bin/bash

# Install CX Consulting AI as a systemd service
echo "🚀 Installing CX Consulting AI as a systemd service..."

# Copy service file to systemd directory
sudo cp cx-consulting-ai.service /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable cx-consulting-ai.service

# Start the service
sudo systemctl start cx-consulting-ai.service

# Check status
echo "✅ Service installation complete!"
echo ""
echo "📋 Service Management Commands:"
echo "  Start:   sudo systemctl start cx-consulting-ai"
echo "  Stop:    sudo systemctl stop cx-consulting-ai"
echo "  Restart: sudo systemctl restart cx-consulting-ai"
echo "  Status:  sudo systemctl status cx-consulting-ai"
echo "  Logs:    sudo journalctl -u cx-consulting-ai -f"
echo ""
echo "🔍 Current Status:"
sudo systemctl status cx-consulting-ai --no-pager
