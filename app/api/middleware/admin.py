# app/api/middleware/admin.py
from fastapi import HTTPException, status, Depends
from typing import Optional, Callable
from app.api.auth import get_current_user

def admin_required(func: Callable):
    """Dependency to check if a user is an admin."""
    async def wrapper(*args, current_user: dict = Depends(get_current_user), **kwargs):
        if not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await func(*args, current_user=current_user, **kwargs)
    return wrapper 