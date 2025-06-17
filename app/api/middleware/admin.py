# app/api/middleware/admin.py
"""Admin dependency to ensure the current user has admin privileges.

Usage (route example):

    @router.get("/admin-only")
    async def admin_only_route(current_user: dict = Depends(admin_required)):
        ...

`admin_required` simply verifies the user is admin and returns the user dict for
down-stream usage.
"""

from fastapi import Depends, HTTPException, status

from app.api.auth import get_current_user


async def admin_required(current_user: dict = Depends(get_current_user)) -> dict:  # type: ignore
    """Raise 403 unless the authenticated user has is_admin = True.

    Returns the `current_user` dict so the route handler can access user info
    after the check passes (just like get_current_user).
    """

    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return current_user
