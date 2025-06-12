#!/bin/bash

# Setup Dynamic DNS and SSL Certificate for CX Consulting AI
# Uses a free dynamic DNS service to get a proper domain name

set -e

echo "🌐 Setting up Dynamic DNS and SSL Certificate..."

# Configuration
EC2_IP="51.20.53.151"
SUBDOMAIN="cx-consulting-ai"
DDNS_SERVICE="duckdns.org"  # Free dynamic DNS service
EMAIL="support@cxconsulting.ai"

echo "📋 This script will help you set up:"
echo "   1. A free subdomain: ${SUBDOMAIN}.duckdns.org"
echo "   2. Point it to your EC2 IP: $EC2_IP"
echo "   3. Get a valid SSL certificate for the subdomain"
echo ""
echo "🔗 First, you need to:"
echo "   1. Go to https://www.duckdns.org/"
echo "   2. Sign in with Google/GitHub/etc."
echo "   3. Create a subdomain: $SUBDOMAIN"
echo "   4. Set the IP to: $EC2_IP"
echo "   5. Get your DuckDNS token"
echo ""
echo "💡 Alternative free DNS services:"
echo "   - duckdns.org (recommended)"
echo "   - noip.com"
echo "   - freedns.afraid.org"
echo ""

read -p "Have you set up the subdomain and have your DuckDNS token? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please set up the subdomain first, then run this script again."
    exit 1
fi

read -p "Enter your DuckDNS token: " DUCKDNS_TOKEN
read -p "Enter your subdomain (without .duckdns.org): " USER_SUBDOMAIN

FULL_DOMAIN="${USER_SUBDOMAIN}.duckdns.org"

echo ""
echo "📋 Configuration:"
echo "   Domain: $FULL_DOMAIN"
echo "   IP: $EC2_IP"
echo "   Email: $EMAIL"
echo ""

# Update the IP (in case it changed)
echo "🔄 Updating DuckDNS IP..."
curl "https://www.duckdns.org/update?domains=${USER_SUBDOMAIN}&token=${DUCKDNS_TOKEN}&ip=${EC2_IP}"
echo ""

# Test DNS resolution
echo "🧪 Testing DNS resolution..."
sleep 5
if nslookup "$FULL_DOMAIN" | grep -q "$EC2_IP"; then
    echo "✅ DNS resolution working!"
else
    echo "❌ DNS not resolving yet. Please wait a few minutes and try again."
    exit 1
fi

# Install Certbot if not already installed
echo "🔧 Installing Certbot..."
sudo apt update
sudo apt install -y certbot python3-certbot-nginx

# Stop nginx temporarily
echo "⏹️  Stopping nginx..."
sudo systemctl stop nginx

# Get SSL certificate
echo "🔐 Obtaining SSL certificate for $FULL_DOMAIN..."
sudo certbot certonly \
    --standalone \
    --non-interactive \
    --agree-tos \
    --email "$EMAIL" \
    -d "$FULL_DOMAIN"

# Create nginx configuration
NGINX_CONF="/etc/nginx/sites-available/cx-consulting-ai"
NGINX_LINK="/etc/nginx/sites-enabled/cx-consulting-ai"

echo "⚙️  Creating nginx configuration..."
sudo tee "$NGINX_CONF" > /dev/null << EOF
# HTTP server - redirect to HTTPS
server {
    listen 80;
    server_name $FULL_DOMAIN;

    # Redirect all HTTP requests to HTTPS
    return 301 https://\$server_name\$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name $FULL_DOMAIN;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/$FULL_DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$FULL_DOMAIN/privkey.pem;

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

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

# Enable the site
echo "🔗 Enabling nginx site..."
sudo ln -sf "$NGINX_CONF" "$NGINX_LINK"
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

echo ""
echo "✅ SSL Certificate setup complete!"
echo ""
echo "🔒 Your site is now available at:"
echo "   https://$FULL_DOMAIN"
echo ""
echo "📝 Update your frontend to use this URL:"
echo "   NEXT_PUBLIC_API_URL=https://$FULL_DOMAIN"
echo ""
echo "🔄 Certificate will auto-renew every 60 days"
