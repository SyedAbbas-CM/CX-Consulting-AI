export interface LoginRequest {
  username: string;
  password?: string; // Assuming password might be optional depending on auth method
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface User {
  id: number; // Or string, depending on your backend ID type
  username: string;
  email?: string; // Optional email
  full_name?: string; // Optional full name
  disabled?: boolean;
  // Add other relevant user properties
} 