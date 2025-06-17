#!/bin/bash
# Backend Switcher for CX Consulting AI
# Usage: ./switch_backend.sh [local|aws]

set -e

FRONTEND_DIR="app/frontend/cx-consulting-ai-3"
ENV_FILE="$FRONTEND_DIR/.env.local"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo -e "${BLUE}🔄 Backend Switcher for CX Consulting AI${NC}"
    echo ""
    echo "Usage: $0 [local|aws|status]"
    echo ""
    echo "Commands:"
    echo "  local   - Switch to local backend (localhost:8000)"
    echo "  aws     - Switch to AWS backend (cx.cloudprimerolabs.com)"
    echo "  status  - Show current backend configuration"
    echo ""
}

get_current_backend() {
    if [ -f "$ENV_FILE" ]; then
        if grep -q "localhost:8000" "$ENV_FILE"; then
            echo "local"
        elif grep -q "cx.cloudprimerolabs.com" "$ENV_FILE"; then
            echo "aws"
        else
            echo "unknown"
        fi
    else
        echo "none"
    fi
}

show_status() {
    current=$(get_current_backend)
    echo -e "${BLUE}📊 Current Backend Status:${NC}"
    echo ""

    if [ "$current" = "local" ]; then
        echo -e "  Backend: ${GREEN}LOCAL${NC} (localhost:8000)"
        echo -e "  Status: $(curl -s http://localhost:8000/ >/dev/null 2>&1 && echo -e "${GREEN}✅ Running${NC}" || echo -e "${RED}❌ Not Running${NC}")"
    elif [ "$current" = "aws" ]; then
        echo -e "  Backend: ${YELLOW}AWS${NC} (cx.cloudprimerolabs.com)"
        echo -e "  Status: $(curl -s https://cx.cloudprimerolabs.com/ >/dev/null 2>&1 && echo -e "${GREEN}✅ Running${NC}" || echo -e "${RED}❌ Not Running${NC}")"
    else
        echo -e "  Backend: ${RED}Not Configured${NC}"
    fi
    echo ""
}

switch_to_local() {
    echo -e "${BLUE}🔄 Switching to LOCAL backend...${NC}"

    cat > "$ENV_FILE" << EOF
# Frontend Environment - LOCAL Backend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# Backend Mode
BACKEND_MODE=local
EOF

    echo -e "${GREEN}✅ Switched to LOCAL backend (localhost:8000)${NC}"
    echo -e "${YELLOW}⚠️  Make sure your local backend is running: python start.py${NC}"
}

switch_to_aws() {
    echo -e "${BLUE}🔄 Switching to AWS backend...${NC}"

    cat > "$ENV_FILE" << EOF
# Frontend Environment - AWS Backend
NEXT_PUBLIC_API_URL=https://cx.cloudprimerolabs.com
NEXT_PUBLIC_API_BASE_URL=https://cx.cloudprimerolabs.com

# Backend Mode
BACKEND_MODE=aws
EOF

    echo -e "${GREEN}✅ Switched to AWS backend (cx.cloudprimerolabs.com)${NC}"
    echo -e "${YELLOW}⚠️  Make sure AWS backend is running${NC}"
}

# Main logic
case "${1:-status}" in
    "local")
        switch_to_local
        echo ""
        show_status
        ;;
    "aws")
        switch_to_aws
        echo ""
        show_status
        ;;
    "status")
        show_status
        ;;
    *)
        print_usage
        exit 1
        ;;
esac

echo -e "${BLUE}💡 Tip: Restart your frontend (npm run dev) to apply changes${NC}"
