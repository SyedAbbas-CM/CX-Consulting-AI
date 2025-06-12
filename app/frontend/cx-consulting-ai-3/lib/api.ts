/**
 * API client for communicating with the CX Consulting AI backend
 */

// For production, always use the AWS backend with HTTPS. For local development, you can override with NEXT_PUBLIC_API_URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://ec2-51-20-53-151.eu-north-1.compute.amazonaws.com';

// Types
export interface SearchResult {
  source: string;
  score: number;
  text_snippet: string;
}

export interface QuestionResponse {
  answer: string;
  conversation_id: string;
  sources: SearchResult[];
  processing_time: number;
}

export interface DocumentUploadResponse {
  filename: string;
  chunks_created: number;
  message: string;
}

export interface DeliverableResponse {
  content: string;
  conversation_id: string;
  processing_time: number;
}

export interface ConversationInfo {
  id: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationsResponse {
  conversations: ConversationInfo[];
  count: number;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  expires_at: string;
  user_id: string;
  username: string;
}

/**
 * Helper function to get authentication headers
 */
export function getAuthHeaders(contentType = 'application/json') {
  const token = typeof window !== 'undefined' ? localStorage.getItem('authToken') : null;
  const headers: Record<string, string> = {};

  if (contentType !== 'multipart/form-data') {
    headers['Content-Type'] = contentType;
  }

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  return headers;
}

/**
 * Login with username and password
 */
export async function login(username: string, password: string): Promise<AuthResponse> {
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);

  const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Login failed' }));
    throw new Error(errorData.detail || 'Login failed');
  }

  return response.json();
}

/**
 * Register a new user
 */
export async function register(userData: {
  username: string;
  email: string;
  password: string;
  full_name?: string;
  company?: string;
}): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(userData),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Registration failed' }));
    throw new Error(errorData.detail || 'Registration failed');
  }

  return response.json();
}

/**
 * Get current user profile
 */
export async function getCurrentUser(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
    method: 'GET',
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Failed to get user profile' }));
    throw new Error(errorData.detail || 'Failed to get user profile');
  }

  return response.json();
}

/**
 * Ask a question to the CX Consulting AI
 */
export async function askQuestion(query: string, conversationId?: string): Promise<QuestionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/ask`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({
      query,
      conversation_id: conversationId,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Failed to ask question' }));
    throw new Error(errorData.detail || 'Failed to ask question');
  }

  return response.json();
}

/**
 * Upload a document to the CX Consulting AI
 */
export async function uploadDocument(file: File, projectId?: string): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  // Add project ID if provided
  if (projectId) {
    formData.append('project_id', projectId);
  }

  const response = await fetch(`${API_BASE_URL}/api/documents`, {
    method: 'POST',
    headers: getAuthHeaders('multipart/form-data'),
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Failed to upload document' }));
    throw new Error(errorData.detail || 'Failed to upload document');
  }

  return response.json();
}

/**
 * Generate a CX strategy
 */
export async function generateCXStrategy(
  clientName: string,
  industry: string,
  challenges: string,
  conversationId?: string
): Promise<DeliverableResponse> {
  const response = await fetch(`${API_BASE_URL}/api/cx-strategy`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({
      client_name: clientName,
      industry,
      challenges,
      conversation_id: conversationId,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Failed to generate CX strategy' }));
    throw new Error(errorData.detail || 'Failed to generate CX strategy');
  }

  return response.json();
}

/**
 * Generate an ROI analysis
 */
export async function generateROIAnalysis(
  clientName: string,
  industry: string,
  projectDescription: string,
  currentMetrics: string,
  conversationId?: string
): Promise<DeliverableResponse> {
  const response = await fetch(`${API_BASE_URL}/api/roi-analysis`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({
      client_name: clientName,
      industry,
      project_description: projectDescription,
      current_metrics: currentMetrics,
      conversation_id: conversationId,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Failed to generate ROI analysis' }));
    throw new Error(errorData.detail || 'Failed to generate ROI analysis');
  }

  return response.json();
}

/**
 * Generate a customer journey map
 */
export async function generateJourneyMap(
  clientName: string,
  industry: string,
  persona: string,
  scenario: string,
  conversationId?: string
): Promise<DeliverableResponse> {
  const response = await fetch(`${API_BASE_URL}/api/journey-map`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({
      client_name: clientName,
      industry,
      persona,
      scenario,
      conversation_id: conversationId,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Failed to generate journey map' }));
    throw new Error(errorData.detail || 'Failed to generate journey map');
  }

  return response.json();
}

/**
 * Get conversation history
 */
export async function getConversations(): Promise<ConversationsResponse> {
  const response = await fetch(`${API_BASE_URL}/api/conversations`, {
    method: 'GET',
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Failed to get conversations' }));
    throw new Error(errorData.detail || 'Failed to get conversations');
  }

  return response.json();
}

/**
 * Delete a conversation
 */
export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/conversations/${conversationId}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Failed to delete conversation' }));
    throw new Error(errorData.detail || 'Failed to delete conversation');
  }
}

/**
 * Health check
 */
export async function checkHealth(): Promise<{ status: string; model: string; version: string }> {
  const response = await fetch(`${API_BASE_URL}/api/health`, {
    method: 'GET',
  });

  if (!response.ok) {
    throw new Error('Health check failed');
  }

  return response.json();
}
