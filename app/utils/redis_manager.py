#!/usr/bin/env python
"""
Redis Manager Utility

This script provides utilities for checking and starting a Redis server.
"""
import logging
import os

import redis.asyncio as aioredis  # Correct import based on requirements

logger = logging.getLogger(__name__)


async def check_redis_status(redis_url: str = None) -> bool:
    """
    Ping the Redis instance.
    Returns True if the server responds, else False.
    Never raises errors.
    """
    if not redis_url:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    conn = None  # Initialize conn to None
    try:
        # Use timeout settings to prevent hanging indefinitely
        conn = await aioredis.from_url(
            redis_url, socket_timeout=2, socket_connect_timeout=2
        )
        pong = await conn.ping()
        if pong:
            logger.info(f"Redis ping OK to {redis_url}")
            return True
        else:
            logger.warning(
                f"Redis ping to {redis_url} returned False (unexpected). Treating as failure."
            )
            return False
    except (aioredis.RedisError, ConnectionRefusedError, TimeoutError, OSError) as e:
        logger.warning(f"Redis connection/ping failed for {redis_url}: {e}")
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error during Redis check for {redis_url}: {e}", exc_info=True
        )
        return False
    finally:
        if conn:
            try:
                await conn.close()
            except Exception as close_e:
                logger.warning(
                    f"Error closing Redis connection during check: {close_e}"
                )


async def ensure_redis_running(redis_url: str = None):
    """Checks Redis status and raises an exception if unavailable."""
    if not await check_redis_status(redis_url):
        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        logger.critical(
            f"Redis server at {url} is not responding. Application cannot proceed."
        )
        # Optionally, attempt to start Redis here if desired and feasible
        raise ConnectionError(f"Redis server at {url} is unavailable.")
    else:
        logger.info("Redis server connection confirmed.")
