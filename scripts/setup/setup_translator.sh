#!/bin/bash
# Translator setup script for PyDAC
# Configure the DACPP translator environment variable

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DACPP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TRANSLATOR_PATH="$DACPP_DIR/translator/bin/translator"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  set, s          Set translator environment variable"
    echo "  status, st      Show current translator status"
    echo "  help, h         Show help information"
    echo ""
    echo "Environment Variables:"
    echo "  DACPP_TRANSLATOR    Path to the translator executable"
    echo ""
}

check_translator() {
    local path=$1
    if [ -f "$path" ] && [ -x "$path" ]; then
        return 0
    else
        return 1
    fi
}

set_translator() {
    if check_translator "$TRANSLATOR_PATH"; then
        export DACPP_TRANSLATOR="$TRANSLATOR_PATH"
        echo -e "${GREEN}[SUCCESS] Translator configured${NC}"
        echo "   DACPP_TRANSLATOR=$TRANSLATOR_PATH"
        echo ""
        echo "This setting is active for the current terminal session."
        echo "To make it permanent, add the following to your ~/.bashrc:"
        echo "   export DACPP_TRANSLATOR=\"$TRANSLATOR_PATH\""
        return 0
    else
        echo -e "${RED}[ERROR] Translator not found or not executable${NC}"
        echo "   Path: $TRANSLATOR_PATH"
        echo ""
        echo "Please ensure the translator is built and available at:"
        echo "   $TRANSLATOR_PATH"
        return 1
    fi
}


show_status() {
    echo "=================="
    echo "Translator Status"
    echo "=================="
    echo ""

    # Check current environment variable
    if [ -n "$DACPP_TRANSLATOR" ]; then
        echo -e "${BLUE}Current translator:${NC} $DACPP_TRANSLATOR"
        if check_translator "$DACPP_TRANSLATOR"; then
            echo -e "${GREEN}[OK] Translator is available and executable${NC}"
        else
            echo -e "${RED}[ERROR] Translator not found or not executable${NC}"
        fi
    else
        echo -e "${YELLOW}[WARNING] DACPP_TRANSLATOR environment variable not set${NC}"
        echo ""
        echo "Checking default translator..."
        if check_translator "$TRANSLATOR_PATH"; then
            echo -e "${GREEN}[OK] Default translator available at: $TRANSLATOR_PATH${NC}"
            echo "   Run '$0 set' to configure it"
        else
            echo -e "${RED}[ERROR] Default translator not available${NC}"
            echo "   Expected location: $TRANSLATOR_PATH"
        fi
    fi

    echo ""
    echo "Usage:"
    echo "  $0 set      Configure translator environment variable"
    echo "  $0 status   Show current status (this output)"
    echo "  $0 help     Show help information"
}

# Main logic
if [ $# -eq 0 ]; then
    # If no arguments, show status
    show_status
    exit 0
fi

case "$1" in
    set|s)
        set_translator
        ;;
    status|st)
        show_status
        ;;
    help|h|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}[ERROR] Unknown option: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
