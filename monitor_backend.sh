#!/bin/bash
# Backend Monitor for CX Consulting AI
# Monitors LLM responses, timeouts, and performance issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BACKEND_URL="http://localhost:8000"
LOG_FILE="backend_monitor.log"
TIMEOUT_THRESHOLD=30 # seconds

print_header() {
    clear
    echo -e "${BLUE}🔍 CX Consulting AI Backend Monitor${NC}"
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo -e "Backend: ${GREEN}$BACKEND_URL${NC}"
    echo -e "Timeout Threshold: ${YELLOW}${TIMEOUT_THRESHOLD}s${NC}"
    echo -e "Log File: ${CYAN}$LOG_FILE${NC}"
    echo ""
}

check_backend_status() {
    echo -e "${BLUE}📊 Backend Status Check:${NC}"

    # Health check
    if curl -s "$BACKEND_URL/" >/dev/null 2>&1; then
        echo -e "  Health: ${GREEN}✅ Running${NC}"
    else
        echo -e "  Health: ${RED}❌ Not Running${NC}"
        return 1
    fi

    # Check models endpoint (requires auth)
    response=$(curl -s -w "%{http_code}" "$BACKEND_URL/api/models" -o /dev/null)
    if [ "$response" = "401" ]; then
        echo -e "  Auth: ${GREEN}✅ Working (401 expected)${NC}"
    else
        echo -e "  Auth: ${YELLOW}⚠️  Unexpected response: $response${NC}"
    fi

    echo ""
}

monitor_logs() {
    echo -e "${BLUE}📝 Real-time Backend Logs:${NC}"
    echo -e "${BLUE}─────────────────────────────${NC}"

    # Find the backend process and monitor its logs
    if pgrep -f "uvicorn app.main:app" >/dev/null; then
        echo -e "${GREEN}✅ Backend process found${NC}"
        echo ""

        # Monitor logs with filtering for important events
        tail -f /dev/null 2>/dev/null &
        TAIL_PID=$!

        # Use journalctl or direct log monitoring
        if command -v journalctl >/dev/null 2>&1; then
            journalctl -f -u cx-consulting-ai 2>/dev/null | while read line; do
                timestamp=$(date '+%H:%M:%S')

                # Color code different types of messages
                if echo "$line" | grep -q "ERROR\|error\|Error"; then
                    echo -e "${RED}[$timestamp] $line${NC}"
                elif echo "$line" | grep -q "WARNING\|warning\|Warning"; then
                    echo -e "${YELLOW}[$timestamp] $line${NC}"
                elif echo "$line" | grep -q "LLM\|llm\|model\|Model"; then
                    echo -e "${PURPLE}[$timestamp] $line${NC}"
                elif echo "$line" | grep -q "timeout\|Timeout\|TIMEOUT"; then
                    echo -e "${RED}[$timestamp] ⏰ TIMEOUT: $line${NC}"
                elif echo "$line" | grep -q "JWT\|jwt\|token\|Token"; then
                    echo -e "${CYAN}[$timestamp] 🔐 AUTH: $line${NC}"
                else
                    echo -e "${NC}[$timestamp] $line${NC}"
                fi

                # Log to file
                echo "[$timestamp] $line" >> "$LOG_FILE"
            done
        else
            echo -e "${YELLOW}⚠️  journalctl not available, monitoring process output...${NC}"

            # Alternative: monitor the process directly
            ps aux | grep "uvicorn app.main:app" | grep -v grep | while read line; do
                echo -e "${GREEN}Process: $line${NC}"
            done
        fi

        kill $TAIL_PID 2>/dev/null || true
    else
        echo -e "${RED}❌ Backend process not found${NC}"
        echo -e "${YELLOW}💡 Start the backend with: python start.py${NC}"
        return 1
    fi
}

test_llm_response() {
    echo -e "${BLUE}🧪 Testing LLM Response Time:${NC}"
    echo -e "${BLUE}─────────────────────────────${NC}"

    # First, we need to authenticate
    echo -e "${CYAN}🔐 Authenticating...${NC}"

    auth_response=$(curl -s -X POST "$BACKEND_URL/api/auth/login" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=monitor&password=monitor123" \
        -w "%{http_code}" -o /tmp/auth_response.json)

    http_code="${auth_response: -3}"

    if [ "$http_code" = "200" ]; then
        # Try multiple ways to extract the token
        if command -v jq >/dev/null 2>&1; then
            token=$(cat /tmp/auth_response.json | jq -r '.access_token' 2>/dev/null)
        else
            # Fallback without jq
            token=$(cat /tmp/auth_response.json | sed -n 's/.*"access_token":"\([^"]*\)".*/\1/p')
        fi

        if [ -n "$token" ] && [ "$token" != "null" ]; then
            echo -e "${GREEN}✅ Authentication successful${NC}"
            echo -e "${CYAN}Token: ${token:0:20}...${NC}"

            # Test a simple LLM request
            echo -e "${PURPLE}🤖 Testing LLM response...${NC}"

            start_time=$(date +%s)

            llm_response=$(timeout $TIMEOUT_THRESHOLD curl -s -X POST "$BACKEND_URL/api/ask" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $token" \
                -d '{"message":"Hello, how are you?","project_id":"test"}' \
                -w "%{http_code}" -o /tmp/llm_response.json 2>/dev/null)

            llm_exit_code=$?
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            llm_http_code="${llm_response: -3}"

            if [ $llm_exit_code -eq 124 ]; then
                echo -e "${RED}⏰ TIMEOUT: LLM took longer than ${TIMEOUT_THRESHOLD}s${NC}"
            elif [ "$llm_http_code" = "200" ]; then
                echo -e "${GREEN}✅ LLM responded in ${duration}s${NC}"
                if [ -f /tmp/llm_response.json ]; then
                    response_preview=$(cat /tmp/llm_response.json | head -c 100)
                    echo -e "${CYAN}Preview: $response_preview...${NC}"
                fi
            elif [ "$llm_http_code" = "401" ]; then
                echo -e "${RED}🔐 JWT TOKEN EXPIRED or INVALID${NC}"
                cat /tmp/llm_response.json 2>/dev/null || echo "No response body"
            elif [ "$llm_http_code" = "504" ]; then
                echo -e "${RED}⏰ GATEWAY TIMEOUT: Backend took too long${NC}"
            else
                echo -e "${RED}❌ LLM request failed with code: $llm_http_code${NC}"
                cat /tmp/llm_response.json 2>/dev/null || echo "No response body"
            fi
        else
            echo -e "${RED}❌ Could not extract access token${NC}"
            cat /tmp/auth_response.json
        fi
    else
        echo -e "${RED}❌ Authentication failed (HTTP $http_code)${NC}"
        cat /tmp/auth_response.json 2>/dev/null || echo "No response body"
    fi

    # Cleanup
    rm -f /tmp/auth_response.json /tmp/llm_response.json
    echo ""
}

monitor_performance() {
    echo -e "${BLUE}📈 Performance Monitoring:${NC}"
    echo -e "${BLUE}─────────────────────────${NC}"

    while true; do
        # CPU and Memory usage
        if pgrep -f "uvicorn app.main:app" >/dev/null; then
            pid=$(pgrep -f "uvicorn app.main:app")
            cpu=$(ps -p $pid -o %cpu --no-headers 2>/dev/null | tr -d ' ')
            mem=$(ps -p $pid -o %mem --no-headers 2>/dev/null | tr -d ' ')

            timestamp=$(date '+%H:%M:%S')
            echo -e "${GREEN}[$timestamp] CPU: ${cpu}% | Memory: ${mem}%${NC}"
        else
            echo -e "${RED}[$(date '+%H:%M:%S')] Backend process not found${NC}"
        fi

        sleep 5
    done
}

# Main menu
case "${1:-status}" in
    "status")
        print_header
        check_backend_status
        ;;
    "logs")
        print_header
        check_backend_status
        monitor_logs
        ;;
    "test")
        print_header
        check_backend_status
        test_llm_response
        ;;
    "performance")
        print_header
        check_backend_status
        monitor_performance
        ;;
    "full")
        print_header
        check_backend_status
        echo -e "${BLUE}🚀 Starting full monitoring...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""

        # Run test first
        test_llm_response

        # Then start log monitoring
        monitor_logs &
        LOG_PID=$!

        # And performance monitoring
        monitor_performance &
        PERF_PID=$!

        # Wait for interrupt
        trap "kill $LOG_PID $PERF_PID 2>/dev/null; exit 0" INT
        wait
        ;;
    *)
        echo -e "${BLUE}🔍 Backend Monitor Usage:${NC}"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  status      - Check backend status"
        echo "  logs        - Monitor real-time logs"
        echo "  test        - Test LLM response time"
        echo "  performance - Monitor CPU/Memory usage"
        echo "  full        - Full monitoring (logs + performance + test)"
        echo ""
        exit 1
        ;;
esac
