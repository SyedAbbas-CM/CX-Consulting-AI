#!/usr/bin/env python
"""
Database Initialization Script

This script initializes the SQLite database and creates a default admin user.
"""
import os
import sys
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add app directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.services.auth_service import AuthService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("init_db")

async def init_database():
    """Initialize the database and create a default admin user."""
    print("\n========== DATABASE INITIALIZATION ==========\n")
    
    # Initialize auth service (this will create the database)
    auth_service = AuthService()
    print("✅ Database initialized")
    
    # Check if default admin user exists
    default_username = "admin"
    existing_user = auth_service.get_user_by_username(default_username)
    
    if existing_user:
        print(f"✅ Default admin user '{default_username}' already exists")
    else:
        # Create default admin user
        default_password = "admin123"  # For testing only
        
        user = auth_service.create_user(
            username=default_username,
            email="admin@example.com",
            password=default_password,
            full_name="Default Admin",
            is_admin=True
        )
        
        if user:
            print(f"✅ Created default admin user:")
            print(f"   Username: {default_username}")
            print(f"   Password: {default_password}")
            print(f"   User ID: {user['id']}")
        else:
            print("❌ Failed to create default admin user")
    
    # Create a default test user
    test_username = "test"
    existing_test_user = auth_service.get_user_by_username(test_username)
    
    if existing_test_user:
        print(f"✅ Default test user '{test_username}' already exists")
    else:
        # Create default test user
        test_password = "test123"  # For testing only
        
        user = auth_service.create_user(
            username=test_username,
            email="test@example.com",
            password=test_password,
            full_name="Test User"
        )
        
        if user:
            print(f"✅ Created default test user:")
            print(f"   Username: {test_username}")
            print(f"   Password: {test_password}")
            print(f"   User ID: {user['id']}")
        else:
            print("❌ Failed to create default test user")
    
    print("\n========== INITIALIZATION COMPLETE ==========\n")

if __name__ == "__main__":
    asyncio.run(init_database()) 