#!/bin/bash

# Configure frontend for Azure deployment
echo "Configuring frontend for Azure deployment with AWS backend..."

# Set the API base URL for production
export NEXT_PUBLIC_API_BASE_URL="http://ec2-51-20-53-151.eu-north-1.compute.amazonaws.com:8000/api"

echo "Frontend configured to use AWS backend: $NEXT_PUBLIC_API_BASE_URL"

# Build the frontend for production
cd app/frontend/cx-consulting-ai-3
npm run build

echo "Frontend build completed!"
