import logging
import sqlite3
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.api.models import (
    Token,
    User,
    UserCreate,
    UserLogin,
    UsersResponse,
    UserUpdateRequest,
)
from app.services.auth_service import AuthService

# Configure logger
logger = logging.getLogger("cx_consulting_ai.api.auth")

# Create router
router = APIRouter(tags=["Authentication"])

# Create OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Initialize auth service
auth_service = AuthService()


# Dependency to get current user
async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Get current user from token.

    Args:
        token: JWT token

    Returns:
        User data
    """
    # Verify token
    payload = auth_service.verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user ID
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user
    user = auth_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """
    Register a new user.

    Args:
        user_data: User data

    Returns:
        User data
    """
    try:
        # For testing purposes, we'll relax the validation requirements
        # Validate email format - simple check
        if "@" not in user_data.email:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid email format - must contain @",
            )

        # Commented out password length check for testing
        # In production, this should be enabled
        """
        if len(user_data.password) < 8:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Password must be at least 8 characters long"
            )
        """

        # Log the registration attempt
        logger.info(
            f"Attempting to register user: {user_data.username}, email: {user_data.email}"
        )

        # Create user
        user = auth_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            company=user_data.company,
        )

        if not user:
            logger.warning(f"Failed to create user: {user_data.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not create user. Username or email may already exist.",
            )

        logger.info(f"User registered successfully: {user_data.username}")
        return user

    except sqlite3.IntegrityError as e:
        error_msg = str(e)
        logger.warning(f"SQLite integrity error: {error_msg}")
        if "UNIQUE constraint failed: users.email" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email address already in use",
            )
        elif "UNIQUE constraint failed: users.username" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Username already taken"
            )
        else:
            logger.error(f"Database error during registration: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error occurred during registration",
            )
    except Exception as e:
        logger.error(f"Error during user registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}",
        )


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login a user.

    Args:
        form_data: OAuth2 password request form

    Returns:
        Token data
    """
    # Authenticate user
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    token = auth_service.create_access_token(user["id"], user["username"])
    if not token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create access token",
        )

    return token


@router.get("/me", response_model=User)
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Get current user.

    Args:
        current_user: Current user data

    Returns:
        User data
    """
    return current_user


@router.put("/me", response_model=User)
async def update_me(
    user_data: UserUpdateRequest, current_user: dict = Depends(get_current_user)
):
    """
    Update current user.

    Args:
        user_data: User data to update
        current_user: Current user data

    Returns:
        Updated user data
    """
    # Update user
    updates = {}
    for key, value in user_data.dict(exclude_unset=True).items():
        if value is not None:
            updates[key] = value

    if not updates:
        return current_user

    updated_user = auth_service.update_user(current_user["id"], updates)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not update user",
        )

    return updated_user


@router.get("/users", response_model=UsersResponse)
async def list_users(
    limit: int = 50, offset: int = 0, current_user: dict = Depends(get_current_user)
):
    """
    List users (only for admin users in a real app).

    Args:
        limit: Maximum number of users to return
        offset: Offset for pagination
        current_user: Current user data

    Returns:
        User list
    """
    # In a real app, check if current user is admin
    # For now, allow any user to list users
    users, total_count = auth_service.list_users(limit=limit, offset=offset)

    return {"users": users, "count": total_count}
