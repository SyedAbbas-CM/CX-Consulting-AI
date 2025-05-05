"""
Error handling framework for the CX Consulting AI application.

This module provides a centralized approach to error handling across the application,
with standardized error types, logging, and reporting mechanisms.
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, Type, List
from fastapi import HTTPException, status

# Configure logger
logger = logging.getLogger("cx_consulting_ai.error_handling")

# Error types dictionary for mapping error codes to descriptive names
ERROR_TYPES = {
    "REDIS_CONNECTION_ERROR": "Redis connection error",
    "VECTOR_DB_ERROR": "Vector database error",
    "LLM_SERVICE_ERROR": "LLM service error",
    "DOCUMENT_PROCESSING_ERROR": "Document processing error",
    "MEMORY_MANAGER_ERROR": "Memory manager error",
    "CONTEXT_OPTIMIZER_ERROR": "Context optimizer error",
    "RAG_ENGINE_ERROR": "RAG engine error",
    "AUTHENTICATION_ERROR": "Authentication error",
    "AUTHORIZATION_ERROR": "Authorization error",
    "ASYNC_AWAIT_ERROR": "Async/await error",
    "VALIDATION_ERROR": "Validation error",
    "NOT_FOUND_ERROR": "Resource not found",
    "INTERNAL_SERVER_ERROR": "Internal server error"
}

class ApplicationError(Exception):
    """Base exception class for application-specific errors."""
    
    def __init__(
        self,
        message: str,
        error_type: str = "INTERNAL_SERVER_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize the application error.
        
        Args:
            message: Human-readable error message
            error_type: Type of error from ERROR_TYPES
            status_code: HTTP status code for API responses
            details: Additional error details/context
            original_exception: Original exception if this is wrapping another
        """
        self.message = message
        self.error_type = error_type
        self.error_name = ERROR_TYPES.get(error_type, "Unknown error")
        self.status_code = status_code
        self.details = details or {}
        self.original_exception = original_exception
        
        # For proper stack trace
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        error_dict = {
            "error": self.error_name,
            "error_type": self.error_type,
            "message": self.message,
            "status_code": self.status_code
        }
        
        if self.details:
            error_dict["details"] = self.details
            
        return error_dict
    
    def log(self, level: int = logging.ERROR) -> None:
        """Log the error with appropriate detail level."""
        log_message = f"{self.error_name}: {self.message}"
        
        if self.details:
            log_message += f" | Details: {self.details}"
        
        if self.original_exception:
            log_message += f" | Original exception: {str(self.original_exception)}"
            
        logger.log(level, log_message)
        
        # Log stack trace for server errors
        if level >= logging.ERROR:
            logger.log(level, "".join(traceback.format_exception(*sys.exc_info())))
    
    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        return HTTPException(
            status_code=self.status_code,
            detail=self.to_dict()
        )


# Specific error types - these are helpful for more granular error handling
class RedisConnectionError(ApplicationError):
    """Redis connection error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_exception: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_type="REDIS_CONNECTION_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
            original_exception=original_exception
        )


class VectorDBError(ApplicationError):
    """Vector database error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_exception: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_type="VECTOR_DB_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
            original_exception=original_exception
        )


class LLMServiceError(ApplicationError):
    """LLM service error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_exception: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_type="LLM_SERVICE_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
            original_exception=original_exception
        )


class DocumentProcessingError(ApplicationError):
    """Document processing error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_exception: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_type="DOCUMENT_PROCESSING_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
            original_exception=original_exception
        )


class AsyncAwaitError(ApplicationError):
    """Async/await pattern error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_exception: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_type="ASYNC_AWAIT_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
            original_exception=original_exception
        )


class ResourceNotFoundError(ApplicationError):
    """Resource not found error."""
    
    def __init__(self, message: str, resource_type: str, resource_id: str, original_exception: Optional[Exception] = None):
        super().__init__(
            message=message,
            error_type="NOT_FOUND_ERROR",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource_type": resource_type, "resource_id": resource_id},
            original_exception=original_exception
        )


# Utility functions for error handling

async def safe_await(coro, error_message: str, default_value: Any = None) -> Any:
    """
    Safely await a coroutine with proper error handling.
    
    Args:
        coro: The coroutine to await
        error_message: Error message if the await fails
        default_value: Default value to return on error
        
    Returns:
        The awaited result or default value on error
    """
    import asyncio
    
    try:
        return await coro
    except asyncio.CancelledError:
        # Re-raise cancellation as it's usually part of normal control flow
        raise
    except Exception as e:
        error = AsyncAwaitError(
            message=error_message,
            details={"coroutine": str(coro)},
            original_exception=e
        )
        error.log()
        return default_value


def global_exception_handler(request=None, exc=None):
    """
    Global exception handler for the application.
    
    Args:
        request: The FastAPI request
        exc: The exception
        
    Returns:
        A structured error response
    """
    # If it's already our ApplicationError, use it directly
    if isinstance(exc, ApplicationError):
        exc.log()
        return exc.to_dict()
    
    # If it's a HTTPException, convert to our format but maintain status code
    if isinstance(exc, HTTPException):
        error = ApplicationError(
            message=str(exc.detail),
            status_code=exc.status_code,
            error_type="HTTP_ERROR"
        )
        error.log()
        return error.to_dict()
    
    # Otherwise, it's an unexpected error
    error = ApplicationError(
        message="An unexpected error occurred",
        original_exception=exc
    )
    error.log(logging.CRITICAL)
    return error.to_dict() 