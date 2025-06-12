#!/bin/bash
# Upload to AWS S3

PACKAGE_DIR="$1"
BUCKET_NAME="cx-consulting-ai-shared"

if [ -z "$PACKAGE_DIR" ]; then
    echo "Usage: $0 <package_directory>"
    exit 1
fi

echo "📤 Uploading $PACKAGE_DIR to AWS S3..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Please install AWS CLI first:"
    echo "pip install awscli"
    exit 1
fi

# Create bucket if it doesn't exist
aws s3 mb s3://$BUCKET_NAME 2>/dev/null || true

# Upload the package
aws s3 sync "$PACKAGE_DIR" s3://$BUCKET_NAME/$(basename "$PACKAGE_DIR")/ --delete

# Generate presigned URLs for sharing (valid for 7 days)
echo ""
echo "🔗 Shareable links (valid for 7 days):"
aws s3 ls s3://$BUCKET_NAME/$(basename "$PACKAGE_DIR")/ --recursive | while read line; do
    file=$(echo $line | awk '{print $4}')
    url=$(aws s3 presign s3://$BUCKET_NAME/$file --expires-in 604800)
    echo "  $(basename $file): $url"
done

echo "✅ Upload complete!"
