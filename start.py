#!/usr/bin/env python
"""
CX Consulting AI Startup Script

This script helps with starting all required services for the CX Consulting AI application.
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cx_startup")


def check_environment():
    """Check that the environment is properly set up."""
    # Check for required directories
    app_dir = Path("app")
    data_dir = app_dir / "data"

    if not app_dir.exists():
        logger.error(f"App directory not found: {app_dir.absolute()}")
        return False

    # Create data directories if they don't exist
    for subdir in ["documents", "chunked", "vectorstore"]:
        dir_path = data_dir / subdir
        if not dir_path.exists():
            logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)

    return True


def start_redis():
    """Start Redis server if not already running."""
    try:
        # Import the Redis manager from the app
        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
        from app.utils.redis_manager import ensure_redis_running

        if ensure_redis_running():
            logger.info("Redis server is running")
            return True
        else:
            logger.error("Failed to start Redis server")
            return False

    except ImportError:
        logger.error(
            "Could not import Redis manager. Make sure the app is properly installed."
        )
        return False


def start_app(dev_mode=False):
    """Start the FastAPI application."""
    try:
        logger.info(
            f"Starting CX Consulting AI in {'development' if dev_mode else 'production'} mode"
        )

        # Base command
        cmd = ["uvicorn", "app.main:app"]

        # Add options
        if dev_mode:
            cmd.extend(["--reload", "--host", "0.0.0.0", "--port", "8000"])
        else:
            cmd.extend(["--host", "0.0.0.0", "--port", "8000", "--workers", "1"])

        # Start the process
        process = subprocess.Popen(cmd)
        logger.info(f"Server started with PID {process.pid}")

        return process

    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        return None


def main():
    """Main entry point for startup script."""
    parser = argparse.ArgumentParser(description="Start CX Consulting AI")
    parser.add_argument(
        "--dev", action="store_true", help="Start in development mode with auto-reload"
    )
    parser.add_argument(
        "--skip-redis", action="store_true", help="Skip Redis startup check"
    )
    args = parser.parse_args()

    # Banner
    print("\n" + "=" * 50)
    print("CX Consulting AI Startup")
    print("=" * 50 + "\n")

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Start Redis
    if not args.skip_redis:
        if not start_redis():
            print("\nWARNING: Redis failed to start. Continue anyway? (y/n)")
            response = input().strip().lower()
            if response != "y":
                sys.exit(1)

    # Start application server
    app_process = start_app(dev_mode=args.dev)
    if not app_process:
        sys.exit(1)

    print("\n" + "=" * 50)
    print("CX Consulting AI is running!")
    print("Press Ctrl+C to stop the server")
    print("=" * 50 + "\n")

    try:
        # Keep the script running to maintain the process
        while True:
            time.sleep(1)
            # Check if the process is still alive
            if app_process.poll() is not None:
                logger.error("Server process has terminated")
                sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if app_process:
            app_process.terminate()
        print("\nCX Consulting AI has been stopped")


if __name__ == "__main__":
    main()
