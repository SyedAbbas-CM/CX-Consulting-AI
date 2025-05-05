# Authentication System

CX Consulting Agent uses a JWT-based authentication system to secure API endpoints. This document explains how to set up, configure and use the authentication system.

## Setup

### Environment Variables

The authentication system requires the following environment variables to be set:

```
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
AUTH_DB_PATH=users.db
```

- `JWT_SECRET_KEY`: A secret key used to sign JWT tokens (keep this secure!)
- `JWT_ALGORITHM`: The algorithm used for JWT tokens (default: HS256)
- `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time in minutes
- `AUTH_DB_PATH`: Path to the SQLite database file for user storage

### Creating the First Admin User

A script is provided to create the initial admin user:

```bash
cd /path/to/CX-Consulting-Agent
python app/scripts/create_admin.py
```

Follow the prompts to create your admin user.

## API Endpoints

### Authentication Endpoints

- `POST /api/auth/register`: Register a new user
- `POST /api/auth/login`: Login with username and password
- `GET /api/auth/me`: Get the current user's information
- `PUT /api/auth/me`: Update the current user's information

### Admin Endpoints

The following endpoints are only accessible to admin users:

- `GET /api/admin/users`: List all users
- `POST /api/admin/users`: Create a new user
- `PUT /api/admin/users/{user_id}`: Update a user
- `DELETE /api/admin/users/{user_id}`: Delete a user

## Authentication Flow

1. Register a user using `/api/auth/register` or create an admin using the script
2. Login using `/api/auth/login` to receive a JWT token
3. Include the token in the Authorization header of each request:
   ```
   Authorization: Bearer <your-token-here>
   ```

## User Roles

There are two types of users:
- Regular users: Can access most API endpoints
- Admin users: Can access all endpoints including user management

## Security Considerations

- The JWT secret key should be kept secure and not shared
- The system uses bcrypt for password hashing
- Users are never actually deleted; they are marked as inactive
- Admin endpoints are protected with additional middleware

## Integrating with Frontend

For frontend integration, store the JWT token securely (preferably using HttpOnly cookies) and include it in every API request in the Authorization header. 