import json
import logging
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import ExpiredSignatureError, InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel  # For User model

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger("cx_consulting_ai.services.auth")

# Define OAuth2PasswordBearer for token dependency
# Token URL should match your login endpoint, e.g., "/api/auth/login"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# Pydantic model for User (can be expanded)
class User(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    company: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False


class AuthService:
    """Service for user authentication and authorization."""

    def __init__(self):
        """Initialize the auth service."""
        # JWT settings
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key")
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(
            os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
        )

        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Database setup
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "app", "data"
        )
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "users.db")
        logger.info(f"Using SQLite database at: {self.db_path}")
        self._init_db()

    def _init_db(self):
        """Initialize the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create users table if it doesn't exist
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                full_name TEXT,
                company TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_admin BOOLEAN DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
            )

            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")

    def _hash_password(self, password: str) -> str:
        """
        Hash a password for storing.

        Args:
            password: Password to hash

        Returns:
            Hashed password
        """
        return self.pwd_context.hash(password)

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against a hash.

        Args:
            plain_password: Password to verify
            hashed_password: Hashed password to verify against

        Returns:
            True if password matches hash, False otherwise
        """
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        company: Optional[str] = None,
        is_admin: bool = False,
    ) -> Optional[dict]:
        """
        Create a new user.

        Args:
            username: Username
            email: Email
            password: Password
            full_name: Full name
            company: Company
            is_admin: Whether the user is an admin

        Returns:
            User data or None if creation failed
        """
        try:
            # Validate inputs
            if not username or not email or not password:
                logger.error("Missing required fields for user creation")
                return None

            logger.info(f"Creating user with username: {username}, email: {email}")

            # Hash password
            hashed_password = self._hash_password(password)

            # Generate user ID
            user_id = str(uuid.uuid4())

            # Get current time
            now = datetime.utcnow().isoformat()

            # Create user
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO users (
                    id, username, email, hashed_password, full_name, company,
                    is_admin, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    username,
                    email,
                    hashed_password,
                    full_name,
                    company,
                    is_admin,
                    now,
                    now,
                ),
            )

            conn.commit()
            conn.close()

            logger.info(f"User created successfully with ID: {user_id}")

            # Return user data
            return {
                "id": user_id,
                "username": username,
                "email": email,
                "full_name": full_name,
                "company": company,
                "is_admin": is_admin,
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            }

        except sqlite3.IntegrityError as e:
            error_msg = str(e)
            logger.error(f"SQLite integrity error creating user: {error_msg}")
            raise
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return None

    def get_user(self, user_id: str) -> Optional[dict]:
        """
        Get a user by ID.

        Args:
            user_id: User ID

        Returns:
            User data or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM users WHERE id = ? AND is_active = 1", (user_id,)
            )

            user = cursor.fetchone()
            conn.close()

            if not user:
                return None

            # Convert to dict and remove hashed_password
            user_dict = dict(user)
            del user_dict["hashed_password"]

            return user_dict

        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            return None

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """
        Get a user by username.

        Args:
            username: Username

        Returns:
            User data or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM users WHERE username = ? AND is_active = 1", (username,)
            )

            user = cursor.fetchone()
            conn.close()

            if not user:
                return None

            # Convert to dict
            return dict(user)

        except Exception as e:
            logger.error(f"Error getting user by username: {str(e)}")
            return None

    def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Optional[dict]:
        """
        Update a user.

        Args:
            user_id: User ID
            user_data: User data to update

        Returns:
            Updated user data or None if update failed
        """
        try:
            # Check if user exists
            user = self.get_user(user_id)
            if not user:
                return None

            # Get current time
            now = datetime.utcnow().isoformat()

            # Prepare update query
            update_fields = []
            update_values = []

            for key, value in user_data.items():
                if key == "password":
                    update_fields.append("hashed_password = ?")
                    update_values.append(self._hash_password(value))
                elif key in [
                    "username",
                    "email",
                    "full_name",
                    "company",
                    "is_active",
                    "is_admin",
                ]:
                    update_fields.append(f"{key} = ?")
                    update_values.append(value)

            if not update_fields:
                return user

            # Add updated_at
            update_fields.append("updated_at = ?")
            update_values.append(now)

            # Add user_id
            update_values.append(user_id)

            # Update user
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?",
                tuple(update_values),
            )

            conn.commit()
            conn.close()

            # Get updated user
            return self.get_user(user_id)

        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            return None

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user by ID.

        Args:
            user_id: User ID

        Returns:
            True if deleted, False otherwise
        """
        try:
            # Update user as inactive instead of deleting
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE users SET is_active = 0, updated_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), user_id),
            )

            success = cursor.rowcount > 0
            conn.commit()
            conn.close()

            return success

        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            return False

    def list_users(self, limit: int = 50, offset: int = 0) -> Tuple[List[dict], int]:
        """
        List users.

        Args:
            limit: Maximum number of users to return
            offset: Offset for pagination

        Returns:
            Tuple of (users, total_count)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get total count
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
            total_count = cursor.fetchone()[0]

            # Get users
            cursor.execute(
                "SELECT * FROM users WHERE is_active = 1 ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )

            users = cursor.fetchall()
            conn.close()

            # Convert to dict and remove hashed_password
            user_list = []
            for user in users:
                user_dict = dict(user)
                del user_dict["hashed_password"]
                user_list.append(user_dict)

            return user_list, total_count

        except Exception as e:
            logger.error(f"Error listing users: {str(e)}")
            return [], 0

    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        """
        Authenticate a user.

        Args:
            username: Username
            password: Password

        Returns:
            User data or None if authentication failed
        """
        # Get user by username
        user = self.get_user_by_username(username)
        if not user:
            return None

        # Verify password
        if not self._verify_password(password, user["hashed_password"]):
            return None

        # Convert to response dict and remove hashed_password
        user_dict = dict(user)
        del user_dict["hashed_password"]

        return user_dict

    def create_access_token(self, user_id: str, username: str) -> dict:
        """
        Create an access token.

        Args:
            user_id: User ID
            username: Username

        Returns:
            Token data
        """
        # Create token data
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        payload = {"sub": user_id, "username": username, "exp": expire}

        # Create token
        encoded_jwt = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        return {
            "access_token": encoded_jwt,
            "token_type": "bearer",
            "expires_at": expire.isoformat(),
            "user_id": user_id,
            "username": username,
        }

    def verify_token(self, token: str) -> Optional[dict]:
        """
        Verify a token.

        Args:
            token: JWT token

        Returns:
            Token payload or None if invalid
        """
        logger.debug(f"Attempting to verify token: {token[:10]}...")  # Log start
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            logger.debug(
                f"Token verified successfully. Payload: {payload}"
            )  # Log success
            return payload

        except ExpiredSignatureError:
            logger.warning(
                "Token verification failed: Expired signature"
            )  # Log specific error
            return None
        except InvalidTokenError as e:
            logger.warning(
                f"Token verification failed: {type(e).__name__} - {str(e)}"
            )  # Log other JWT errors
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error during token verification: {str(e)}", exc_info=True
            )  # Log unexpected errors
            return None

    # New method to get active user or raise HTTP 401
    def get_current_active_user(
        self, token: str = Depends(oauth2_scheme)
    ) -> Optional[User]:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        payload = self.verify_token(token)
        if payload is None:
            logger.debug("Token verification failed or payload is None")
            raise credentials_exception

        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            logger.debug("User ID (sub) not found in token payload")
            raise credentials_exception

        user_data = self.get_user(user_id)
        if user_data is None:
            logger.debug(
                f"User with ID {user_id} not found or inactive after token verification"
            )
            raise credentials_exception

        # Ensure is_active if it's part of your user model/DB check
        # The get_user method already filters for is_active = 1
        # if not user_data.get("is_active", False):
        #     logger.debug(f"User {user_id} is inactive.")
        #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")

        return User(**user_data)  # Cast to User model

    # Dependency function to be used in path operations
    async def get_current_user_or_raise(
        self, token: str = Depends(oauth2_scheme)
    ) -> User:
        user = self.get_current_active_user(token)  # Reuses the logic above
        if not user:
            # This case should be caught by exceptions within get_current_active_user
            # but as a safeguard:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive after token verification.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
