#!/bin/bash
# Create BitTorrent file for P2P sharing

PACKAGE_DIR="$1"
if [ -z "$PACKAGE_DIR" ]; then
    echo "Usage: $0 <package_directory>"
    exit 1
fi

echo "🌐 Creating torrent for $PACKAGE_DIR..."

# Check if transmission-create is available
if ! command -v transmission-create &> /dev/null; then
    echo "Installing transmission-cli..."
    # For macOS
    if command -v brew &> /dev/null; then
        brew install transmission-cli
    # For Ubuntu/Debian
    elif command -v apt &> /dev/null; then
        sudo apt install transmission-cli
    else
        echo "Please install transmission-cli manually"
        exit 1
    fi
fi

# Create torrent file
TORRENT_FILE="${PACKAGE_DIR}.torrent"
transmission-create -o "$TORRENT_FILE" "$PACKAGE_DIR"

echo "✅ Torrent created: $TORRENT_FILE"
echo "📤 Share this .torrent file with your collaborator"
echo "💡 They can download it using any BitTorrent client"
