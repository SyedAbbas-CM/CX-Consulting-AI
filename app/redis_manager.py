import os
import subprocess
import asyncio
import logging
from typing import Tuple, Optional
import redis

# Configure logger
logger = logging.getLogger("cx_consulting_ai.redis_manager")

async def check_redis_status() -> bool:
    """
    Check if Redis server is running.
    
    Returns:
        bool: True if Redis server is running, False otherwise
    """
    try:
        # Try to ping Redis
        redis_instance = redis.from_url("redis://localhost:6379/0")
        redis_instance.ping()
        logger.info("Redis server is already running")
        return True
    except redis.exceptions.ConnectionError:
        logger.warning("Redis server is not running")
        
        # Try to start Redis server
        try:
            logger.info("Attempting to start Redis server...")
            
            # Check if we're on macOS
            if os.uname().sysname == "Darwin":
                # macOS: try to start Redis using brew services
                result = await run_command("brew services start redis")
                if result[0] == 0:
                    logger.info("Started Redis server using brew services")
                    return True
                
                # If brew services fails, try redis-server directly
                result = await run_command("redis-server --daemonize yes")
                if result[0] == 0:
                    logger.info("Started Redis server directly")
                    return True
            else:
                # Linux/Unix: try to start Redis using systemd
                result = await run_command("systemctl start redis")
                if result[0] == 0:
                    logger.info("Started Redis server using systemd")
                    return True
                
                # Try redis-server directly
                result = await run_command("redis-server --daemonize yes")
                if result[0] == 0:
                    logger.info("Started Redis server directly")
                    return True
            
            # If we got here, Redis could not be started
            logger.error("Failed to start Redis server")
            return False
        except Exception as e:
            logger.error(f"Error starting Redis server: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error checking Redis status: {str(e)}")
        return False

async def run_command(command: str) -> Tuple[int, str, str]:
    """
    Run a shell command asynchronously.
    
    Args:
        command: Command to run
        
    Returns:
        Tuple[int, str, str]: Exit code, stdout, stderr
    """
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    return (
        process.returncode,
        stdout.decode().strip() if stdout else "",
        stderr.decode().strip() if stderr else ""
    ) 