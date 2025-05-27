# app/api/middleware/auth_logging.py
import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logger
logger = logging.getLogger("cx_consulting_ai.auth_logging")


class AuthLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log authentication attempts and API access."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timer
        start_time = time.time()

        # Get path and method
        path = request.url.path
        method = request.method

        # Check if this is an auth endpoint
        is_auth_endpoint = path.startswith("/api/auth/")
        is_admin_endpoint = path.startswith("/api/admin/")

        # Process the request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log authentication attempts
        if is_auth_endpoint:
            status_code = response.status_code
            client_ip = request.client.host

            # Login attempt
            if path == "/api/auth/login" and method == "POST":
                if status_code == 200:
                    logger.info(f"Successful login from {client_ip}")
                else:
                    logger.warning(f"Failed login attempt from {client_ip}")

            # Register attempt
            elif path == "/api/auth/register" and method == "POST":
                if status_code == 201:
                    logger.info(f"New user registered from {client_ip}")
                else:
                    logger.warning(f"Failed registration attempt from {client_ip}")

        # Log admin access
        elif is_admin_endpoint:
            client_ip = request.client.host
            logger.info(
                f"Admin endpoint accessed: {method} {path} from {client_ip} (status: {response.status_code})"
            )

        # Return response
        return response
