#!/usr/bin/env python3
"""
Entry point for the CX Consulting AI application.
This script starts the FastAPI server with uvicorn.
"""

import logging
import sys

import uvicorn

from app.core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cx_consulting_ai")


def main():
    """Main function to start the application."""
    # Get settings
    settings = get_settings()

    logger.info(f"Starting {settings.APP_NAME} on {settings.HOST}:{settings.PORT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # Start the API server
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )


if __name__ == "__main__":
    main()
