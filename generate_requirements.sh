#!/usr/bin/env bash
# Simple helper to snapshot the active Python environment into requirements.txt
# Usage: source your virtualenv, then run ./generate_requirements.sh
set -euo pipefail

echo "Generating requirements.txt from current environment…"

pip freeze > requirements.txt

echo "requirements.txt written with $(wc -l < requirements.txt) packages."
