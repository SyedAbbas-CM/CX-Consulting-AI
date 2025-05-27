import { useState } from 'react';
import * as api from '@/lib/api';

export interface ApiClientState {
  loading: boolean;
  error: string | null;
  conversation: {
    id: string | null;
    messages: Array<{ role: 'user' | 'assistant'; content: string; timestamp?: string }>;
  };
}

export function useApiClient() {
  const [state, setState] = useState<ApiClientState>({
    loading: false,
    error: null,
    conversation: {
      id: null,
      messages: [],
    },
  });

  const resetError = () => {
    setState((prev) => ({ ...prev, error: null }));
  };

  const resetConversation = () => {
    setState((prev) => ({
      ...prev,
      conversation: {
        id: null,
        messages: [],
      },
    }));
  };

  const checkHealth = async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const health = await api.checkHealth();
      setState((prev) => ({ ...prev, loading: false }));
      return health;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to check health',
      }));
      return null;
    }
  };

  const login = async (username: string, password: string) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const authResponse = await api.login(username, password);
      setState((prev) => ({ ...prev, loading: false }));
      return authResponse;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Login failed',
      }));
      return null;
    }
  };

  const register = async (userData: {
    username: string;
    email: string;
    password: string;
    full_name?: string;
    company?: string;
  }) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const response = await api.register(userData);
      setState((prev) => ({ ...prev, loading: false }));
      return response;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Registration failed',
      }));
      return null;
    }
  };

  const getCurrentUser = async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const userData = await api.getCurrentUser();
      setState((prev) => ({ ...prev, loading: false }));
      return userData;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to get user profile',
      }));
      return null;
    }
  };

  const askQuestion = async (query: string) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));

    // Create timestamp
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Add user message to conversation
    setState((prev) => ({
      ...prev,
      conversation: {
        ...prev.conversation,
        messages: [
          ...prev.conversation.messages,
          { role: 'user', content: query, timestamp },
        ],
      },
    }));

    try {
      // Only pass conversation ID if it's not null
      const conversationId = state.conversation.id || undefined;
      const response = await api.askQuestion(query, conversationId);

      // Add assistant response to conversation
      setState((prev) => ({
        ...prev,
        loading: false,
        conversation: {
          id: response.conversation_id,
          messages: [
            ...prev.conversation.messages,
            {
              role: 'assistant',
              content: response.answer,
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            },
          ],
        },
      }));

      return response;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to ask question',
      }));
      return null;
    }
  };

  const uploadDocument = async (file: File) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const response = await api.uploadDocument(file);
      setState((prev) => ({ ...prev, loading: false }));
      return response;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to upload document',
      }));
      return null;
    }
  };

  const generateCXStrategy = async (
    clientName: string,
    industry: string,
    challenges: string
  ) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      // Convert null to undefined
      const conversationId = state.conversation.id || undefined;
      const response = await api.generateCXStrategy(
        clientName,
        industry,
        challenges,
        conversationId
      );

      // Create timestamp
      const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      setState((prev) => ({
        ...prev,
        loading: false,
        conversation: {
          id: response.conversation_id,
          messages: [
            ...prev.conversation.messages,
            {
              role: 'assistant',
              content: `Generated CX Strategy:\n\n${response.content}`,
              timestamp
            },
          ],
        },
      }));
      return response;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to generate CX strategy',
      }));
      return null;
    }
  };

  const generateROIAnalysis = async (
    clientName: string,
    industry: string,
    projectDescription: string,
    currentMetrics: string
  ) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      // Convert null to undefined
      const conversationId = state.conversation.id || undefined;
      const response = await api.generateROIAnalysis(
        clientName,
        industry,
        projectDescription,
        currentMetrics,
        conversationId
      );

      // Create timestamp
      const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      setState((prev) => ({
        ...prev,
        loading: false,
        conversation: {
          id: response.conversation_id,
          messages: [
            ...prev.conversation.messages,
            {
              role: 'assistant',
              content: `Generated ROI Analysis:\n\n${response.content}`,
              timestamp
            },
          ],
        },
      }));
      return response;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to generate ROI analysis',
      }));
      return null;
    }
  };

  const generateJourneyMap = async (
    clientName: string,
    industry: string,
    persona: string,
    scenario: string
  ) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      // Convert null to undefined
      const conversationId = state.conversation.id || undefined;
      const response = await api.generateJourneyMap(
        clientName,
        industry,
        persona,
        scenario,
        conversationId
      );

      // Create timestamp
      const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      setState((prev) => ({
        ...prev,
        loading: false,
        conversation: {
          id: response.conversation_id,
          messages: [
            ...prev.conversation.messages,
            {
              role: 'assistant',
              content: `Generated Journey Map:\n\n${response.content}`,
              timestamp
            },
          ],
        },
      }));
      return response;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to generate journey map',
      }));
      return null;
    }
  };

  const getConversations = async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      const response = await api.getConversations();
      setState((prev) => ({ ...prev, loading: false }));
      return response;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to get conversations',
      }));
      return null;
    }
  };

  const deleteConversation = async (conversationId: string) => {
    setState((prev) => ({ ...prev, loading: true, error: null }));
    try {
      await api.deleteConversation(conversationId);
      setState((prev) => ({ ...prev, loading: false }));
      return true;
    } catch (error) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to delete conversation',
      }));
      return false;
    }
  };

  return {
    state,
    resetError,
    resetConversation,
    checkHealth,
    login,
    register,
    getCurrentUser,
    askQuestion,
    uploadDocument,
    generateCXStrategy,
    generateROIAnalysis,
    generateJourneyMap,
    getConversations,
    deleteConversation,
  };
}
