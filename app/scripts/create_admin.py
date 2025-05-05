#!/usr/bin/env python
# app/scripts/create_admin.py

import argparse
import sys
import os
import getpass
from pathlib import Path

# Add the parent directory to the path so we can import app modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, parent_dir)

from app.services.auth_service import AuthService

def create_admin():
    """Create an admin user interactively."""
    print("=== Create Admin User ===")
    
    # Get user input
    username = input("Username: ")
    email = input("Email: ")
    
    # Get password securely (not echoed to terminal)
    while True:
        password = getpass.getpass("Password: ")
        confirm_password = getpass.getpass("Confirm password: ")
        
        if password == confirm_password:
            break
        print("Passwords do not match. Please try again.")
    
    full_name = input("Full name (optional): ")
    company = input("Company (optional): ")
    
    # Initialize auth service
    auth_service = AuthService()
    
    # Create admin user
    user = auth_service.create_user(
        username=username,
        email=email,
        password=password,
        full_name=full_name if full_name else None,
        company=company if company else None,
        is_admin=True
    )
    
    if user:
        print(f"\nAdmin user created successfully!")
        print(f"Username: {user['username']}")
        print(f"User ID: {user['id']}")
    else:
        print("\nFailed to create admin user. Username or email may already exist.")
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(description="Create an admin user")
    parser.parse_args()
    
    return create_admin()

if __name__ == "__main__":
    sys.exit(main()) 