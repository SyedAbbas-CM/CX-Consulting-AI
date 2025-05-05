#!/usr/bin/env python
# app/scripts/test_auth.py

import argparse
import sys
import os
import json
import requests
from pathlib import Path

def test_auth(base_url, username, password):
    """Test the authentication API endpoints."""
    print(f"Testing authentication at {base_url}")
    
    session = requests.Session()
    
    # Step 1: Login
    print("\n1. Testing login...")
    login_data = {
        "username": username,
        "password": password
    }
    
    login_resp = session.post(
        f"{base_url}/api/auth/login",
        data=login_data
    )
    
    if login_resp.status_code != 200:
        print(f"Login failed: {login_resp.status_code}")
        print(login_resp.text)
        return 1
    
    token_data = login_resp.json()
    access_token = token_data["access_token"]
    token_type = token_data["token_type"]
    user_id = token_data["user_id"]
    print(f"Login successful! User ID: {user_id}")
    
    # Update session headers with token
    session.headers.update({
        "Authorization": f"{token_type} {access_token}"
    })
    
    # Step 2: Get current user
    print("\n2. Testing /me endpoint...")
    me_resp = session.get(f"{base_url}/api/auth/me")
    
    if me_resp.status_code != 200:
        print(f"Get user failed: {me_resp.status_code}")
        print(me_resp.text)
        return 1
    
    user_data = me_resp.json()
    print(f"User info: {json.dumps(user_data, indent=2)}")
    
    # Step 3: Test a protected endpoint
    print("\n3. Testing protected endpoint...")
    protected_resp = session.get(f"{base_url}/api/projects")
    
    if protected_resp.status_code != 200:
        print(f"Protected endpoint failed: {protected_resp.status_code}")
        print(protected_resp.text)
        return 1
    
    print("Protected endpoint access successful!")
    
    # Step 4: Test admin endpoint (may fail if not admin)
    print("\n4. Testing admin endpoint...")
    admin_resp = session.get(f"{base_url}/api/admin/users")
    
    if admin_resp.status_code == 200:
        print("Admin access successful!")
        print(f"Users: {json.dumps(admin_resp.json(), indent=2)}")
    else:
        print(f"Admin access failed: {admin_resp.status_code}")
        print(admin_resp.text)
        if admin_resp.status_code == 403:
            print("This is expected if the user is not an admin.")
    
    print("\nAuthentication testing completed successfully!")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Test the authentication system")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for the API")
    parser.add_argument("--username", required=True, help="Username for login")
    parser.add_argument("--password", required=True, help="Password for login")
    
    args = parser.parse_args()
    
    return test_auth(args.url, args.username, args.password)

if __name__ == "__main__":
    sys.exit(main()) 