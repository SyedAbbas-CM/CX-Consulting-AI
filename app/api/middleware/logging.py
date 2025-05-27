# app/api/middleware/logging.py
import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("cx_consulting_ai")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and metrics."""

    async def dispatch(self, request: Request, call_next):
        # Start timer
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log request details
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s"
        )

        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)

        return response
