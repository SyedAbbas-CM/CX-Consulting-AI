#!/bin/bash
# Upload to Google Drive using rclone

PACKAGE_DIR="$1"
if [ -z "$PACKAGE_DIR" ]; then
    echo "Usage: $0 <package_directory>"
    exit 1
fi

echo "📤 Uploading $PACKAGE_DIR to Google Drive..."

# Install rclone if not present
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
fi

# Configure rclone for Google Drive (interactive)
echo "Setting up Google Drive connection..."
rclone config

# Upload the package
echo "Uploading package..."
rclone copy "$PACKAGE_DIR" gdrive:CX-Consulting-AI-Shared/ --progress

echo "✅ Upload complete!"
echo "🔗 Share the Google Drive folder with your collaborator"
