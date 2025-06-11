#!/bin/bash

# Deploy frontend to Azure with correct environment configuration
echo "🚀 Deploying frontend to Azure Static Web Apps..."

# Navigate to frontend directory
cd app/frontend/cx-consulting-ai-3

# Set environment variable for build
export NEXT_PUBLIC_API_URL="http://ec2-51-20-53-151.eu-north-1.compute.amazonaws.com:8000"

echo "✅ Environment configured:"
echo "   NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL"

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf .next
rm -rf out

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Build the application
echo "🔨 Building application..."
npm run build

echo "✅ Frontend build completed!"
echo "📁 Built files are in .next directory"
echo ""
echo "ℹ️  Next steps:"
echo "   1. Commit and push the changes to trigger Azure deployment"
echo "   2. Or manually deploy the built files to Azure Static Web Apps"
