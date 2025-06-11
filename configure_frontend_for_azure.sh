#!/bin/bash

# Configure frontend for Azure deployment
echo "Configuring frontend for Azure deployment with AWS backend..."

# Set the API base URL for production (note: should NOT include /api at the end)
export NEXT_PUBLIC_API_URL="http://ec2-51-20-53-151.eu-north-1.compute.amazonaws.com:8000"

echo "Frontend configured to use AWS backend: $NEXT_PUBLIC_API_URL"

# Build the frontend for production
cd app/frontend/cx-consulting-ai-3
npm run build

echo "Frontend build completed!"
