#!/bin/bash

# Deploy SSL Certificate to AWS
echo "🔒 Deploying SSL Certificate to AWS..."

# Configuration - Update these if needed
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

# Transfer SSL setup script
echo "📤 Transferring SSL setup script..."
scp -i "$AWS_KEY" setup_ssl_certificate.sh "$AWS_HOST:~/"

# Run the SSL setup script on AWS
echo "🚀 Running SSL setup on AWS..."
ssh -i "$AWS_KEY" "$AWS_HOST" "chmod +x setup_ssl_certificate.sh && sudo ./setup_ssl_certificate.sh"

echo ""
echo "✅ SSL Certificate deployment complete!"
echo ""
echo "🔒 Your backend should now be available with a valid SSL certificate at:"
echo "   https://ec2-51-20-53-151.eu-north-1.compute.amazonaws.com"
echo ""
echo "🧪 Test it with:"
echo "   curl https://ec2-51-20-53-151.eu-north-1.compute.amazonaws.com/api/auth/login"
