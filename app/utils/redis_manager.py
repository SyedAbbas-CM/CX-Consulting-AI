#!/usr/bin/env python
"""
Redis Manager Utility

This script provides utilities for checking and starting a Redis server.
"""
import os
import sys
import logging
import subprocess
import shutil
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cx_consulting_ai.redis_manager")

def check_redis_running():
    """
    Check if Redis server is running.
    
    Returns:
        bool: True if Redis server is running, False otherwise
    """
    try:
        import redis
        
        # Try to connect to Redis
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            socket_timeout=2  # Short timeout for quick check
        )
        
        # Ping Redis
        redis_client.ping()
        return True
    
    except Exception as e:
        logger.warning(f"Redis connection check failed: {str(e)}")
        return False

def start_redis_server():
    """
    Start the Redis server.
    
    Returns:
        bool: True if Redis was started successfully, False otherwise
    """
    logger.info("Attempting to start Redis server...")
    
    # Check if Redis is installed via Homebrew
    if shutil.which("brew"):
        try:
            # Check if Redis is installed
            result = subprocess.run(["brew", "list", "redis"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
            
            if result.returncode == 0:
                # Redis is installed, start it
                subprocess.run(["brew", "services", "start", "redis"])
                logger.info("Redis server started via Homebrew")
                
                # Give it a moment to start
                time.sleep(2)
                return check_redis_running()
            else:
                # Redis is not installed, install it
                logger.info("Redis not installed. Installing via Homebrew...")
                subprocess.run(["brew", "install", "redis"])
                subprocess.run(["brew", "services", "start", "redis"])
                logger.info("Redis installed and started via Homebrew")
                
                # Give it a moment to start
                time.sleep(3)
                return check_redis_running()
        except Exception as e:
            logger.error(f"Error managing Redis with Homebrew: {str(e)}")
    
    # Try to start Redis directly if available
    if shutil.which("redis-server"):
        try:
            # Start Redis in background
            subprocess.Popen(["redis-server"], 
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            logger.info("Redis server started directly")
            
            # Give it a moment to start
            time.sleep(2)
            return check_redis_running()
        except Exception as e:
            logger.error(f"Error starting Redis server: {str(e)}")
    
    logger.error("Failed to start Redis server. Please install and start Redis manually.")
    return False

def ensure_redis_running():
    """
    Check if Redis is running and start it if not.
    
    Returns:
        bool: True if Redis is running (or was started), False otherwise
    """
    if check_redis_running():
        logger.info("Redis server is already running")
        return True
    else:
        logger.warning("Redis server is not running, attempting to start...")
        return start_redis_server()

if __name__ == "__main__":
    """
    Run as a standalone script to check and start Redis.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--force-start":
        # Force start Redis
        success = start_redis_server()
    else:
        # Ensure Redis is running
        success = ensure_redis_running()
    
    if success:
        print("✅ Redis server is running")
        sys.exit(0)
    else:
        print("❌ Failed to start Redis server")
        sys.exit(1) 