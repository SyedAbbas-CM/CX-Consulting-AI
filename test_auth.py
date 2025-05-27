#!/usr/bin/env python3
import json
import os
import sqlite3

from app.services.auth_service import AuthService


def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


# Initialize auth service
print_header("Auth Service Initialization")
auth = AuthService()
print(f"JWT Secret Key: {auth.secret_key[:10]}... (length: {len(auth.secret_key)})")
print(f"JWT Algorithm: {auth.algorithm}")
print(f"JWT Access Token Expire Minutes: {auth.access_token_expire_minutes}")
print(f"Database Path: {auth.db_path}")

# Check database connection
print_header("Database Check")
try:
    conn = sqlite3.connect(auth.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check users table
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    print(f"Number of users in database: {count}")

    # List users
    print("\nUsers in database:")
    cursor.execute("SELECT id, username, email, is_active, is_admin FROM users")
    users = cursor.fetchall()
    for user in users:
        print(f"- {dict(user)}")

    conn.close()
    print("\nDatabase connection successful")
except Exception as e:
    print(f"Database error: {str(e)}")

# Test authentication
print_header("Authentication Test")
test_username = "newuser"  # Using our newly created user
test_password = "password123"

print(f"Testing authentication for user '{test_username}'")
user = auth.get_user_by_username(test_username)
if user:
    print(f"User found: {user['username']} (ID: {user['id']})")
    print(f"Active: {bool(user['is_active'])}")
    print(f"Admin: {bool(user['is_admin'])}")

    # Test password verification
    print("\nTesting password verification")
    hashed_pwd = user.get("hashed_password", "")
    print(f"Hashed password: {hashed_pwd[:20]}... (length: {len(hashed_pwd)})")

    is_valid = auth._verify_password(test_password, hashed_pwd)
    print(f"Password valid: {is_valid}")

    # Test full authentication
    print("\nTesting full authentication")
    auth_result = auth.authenticate_user(test_username, test_password)
    print(f"Authentication result: {'Success' if auth_result else 'Failed'}")

    if auth_result:
        # Create token
        print("\nCreate access token")
        token = auth.create_access_token(user["id"], user["username"])
        print(f"Token Type: {token['token_type']}")
        print(f"Expires At: {token['expires_at']}")
        print(
            f"Access Token: {token['access_token'][:20]}... (length: {len(token['access_token'])})"
        )

        # Verify token
        print("\nVerify token")
        payload = auth.verify_token(token["access_token"])
        print(f"Token verification: {'Success' if payload else 'Failed'}")
        if payload:
            print(f"Token payload: {json.dumps(payload, indent=2)}")
else:
    print(f"User '{test_username}' not found")

print_header("Test Complete")
