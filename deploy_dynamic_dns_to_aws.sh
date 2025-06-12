#!/bin/bash

# Deploy Dynamic DNS SSL setup to AWS
echo "🌐 Deploying Dynamic DNS SSL setup to AWS..."

# Configuration
AWS_KEY="CX-Consulting-AI.pem"
AWS_HOST="ubuntu@ec2-51-20-53-151.eu-north-1.compute.amazonaws.com"

# Check if SSH key exists
if [ ! -f "$AWS_KEY" ]; then
    echo "❌ SSH key $AWS_KEY not found!"
    echo "💡 Please put your CX-Consulting-AI.pem file in this directory"
    exit 1
fi

chmod 600 "$AWS_KEY"

# Test connection
echo "🧪 Testing SSH connection..."
if ! ssh -i "$AWS_KEY" -o ConnectTimeout=10 "$AWS_HOST" "echo 'Connected!'" 2>/dev/null; then
    echo "❌ Can't connect to AWS. Check your key and instance."
    exit 1
fi

echo "✅ Connected successfully"

# Transfer dynamic DNS SSL setup script
echo "📤 Transferring Dynamic DNS SSL setup script..."
scp -i "$AWS_KEY" setup_dynamic_dns_ssl.sh "$AWS_HOST:~/"

# Run the setup script on AWS
echo "🚀 Running Dynamic DNS SSL setup on AWS..."
echo ""
echo "📋 The script will ask you for:"
echo "   1. Your DuckDNS token"
echo "   2. Your subdomain name"
echo ""
echo "🔗 Make sure you've already set up your subdomain at https://www.duckdns.org/"
echo ""

ssh -i "$AWS_KEY" "$AWS_HOST" "chmod +x setup_dynamic_dns_ssl.sh && ./setup_dynamic_dns_ssl.sh"

echo ""
echo "✅ Dynamic DNS SSL deployment complete!"
