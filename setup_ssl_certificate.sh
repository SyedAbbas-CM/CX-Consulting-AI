#!/bin/bash

# Setup SSL Certificate with Let's Encrypt for CX Consulting AI
# This script sets up a proper SSL certificate to replace the self-signed one

set -e

echo "🔒 Setting up SSL Certificate with Let's Encrypt..."

# Configuration
DOMAIN="ec2-51-20-53-151.eu-north-1.compute.amazonaws.com"
EMAIL="support@cxconsulting.ai"  # Valid email for Let's Encrypt
NGINX_CONF="/etc/nginx/sites-available/cx-consulting-ai"
NGINX_LINK="/etc/nginx/sites-enabled/cx-consulting-ai"

echo "📋 Configuration:"
echo "   Domain: $DOMAIN"
echo "   Email: $EMAIL"
echo ""

# Update system packages
echo "📦 Updating system packages..."
sudo apt update

# Install Certbot and Nginx plugin
echo "🔧 Installing Certbot..."
sudo apt install -y certbot python3-certbot-nginx

# Stop nginx temporarily
echo "⏹️  Stopping nginx..."
sudo systemctl stop nginx

# Get SSL certificate using standalone mode
echo "🔐 Obtaining SSL certificate..."
sudo certbot certonly \
    --standalone \
    --non-interactive \
    --agree-tos \
    --email "$EMAIL" \
    -d "$DOMAIN"

# Create nginx configuration with SSL
echo "⚙️  Creating nginx configuration..."
sudo tee "$NGINX_CONF" > /dev/null << EOF
# HTTP server - redirect to HTTPS
server {
    listen 80;
    server_name $DOMAIN;

    # Redirect all HTTP requests to HTTPS
    return 301 https://\$server_name\$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name $DOMAIN;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;

    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Proxy to FastAPI backend
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Host \$server_name;

        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
EOF

# Enable the site
echo "🔗 Enabling nginx site..."
sudo ln -sf "$NGINX_CONF" "$NGINX_LINK"

# Remove default nginx site if it exists
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
echo "🧪 Testing nginx configuration..."
sudo nginx -t

# Start nginx
echo "🚀 Starting nginx..."
sudo systemctl start nginx
sudo systemctl enable nginx

# Set up automatic certificate renewal
echo "🔄 Setting up automatic certificate renewal..."
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

# Test certificate renewal
echo "🧪 Testing certificate renewal..."
sudo certbot renew --dry-run

echo ""
echo "✅ SSL Certificate setup complete!"
echo ""
echo "🔒 Your site is now available at:"
echo "   https://$DOMAIN"
echo ""
echo "📋 Certificate details:"
sudo certbot certificates
echo ""
echo "🔄 Certificate will auto-renew every 60 days"
echo "💡 You can check renewal status with: sudo certbot renew --dry-run"
