import { create } from 'zustand';
import { ChatSummaryResponse } from '../types/chat';
import { listChats } from '../lib/apiClient'; 

interface ChatState {
  currentProjectId: string | null;
  chats: ChatSummaryResponse[];
  currentChatId: string | null;
  isLoadingChats: boolean;
  error: string | null;
  setProjectId: (projectId: string | null) => void;
  setCurrentChatId: (chatId: string | null) => void;
  setChats: (chats: ChatSummaryResponse[]) => void;
  addChat: (newChat: ChatSummaryResponse) => void;
  removeChat: (chatId: string) => void;
  fetchChats: (projectId: string) => Promise<void>; // Action to load chats
  clearChats: () => void; // Add clearChats action type
}

export const useChatStore = create<ChatState>((set, get) => ({
  currentProjectId: null, // Initialize to null, no project selected by default
  chats: [], // Initial chat list is empty
  currentChatId: null,
  isLoadingChats: false,
  error: null,

  setProjectId: (projectId) => {
    if (get().currentProjectId !== projectId) {
      set({ currentProjectId: projectId, chats: [], currentChatId: null, isLoadingChats: false, error: null });
      console.log(`Chat Store: Project ID set to ${projectId}, chats cleared.`);
    } else {
      console.log(`Chat Store: Project ID already ${projectId}.`);
    }
  },
  
  setCurrentChatId: (chatId) => set({ currentChatId: chatId }),

  setChats: (chats) => set({ chats: chats, isLoadingChats: false }),

  addChat: (newChat) => set((state) => ({
    // Add new chat to the beginning of the list
    chats: [newChat, ...state.chats]
  })),

  removeChat: (chatId) => set((state) => ({
    // Filter out the chat with the given ID (use chat.id based on updated type)
    chats: state.chats.filter(chat => chat.id !== chatId), 
    // If the deleted chat was the current one, reset currentChatId
    currentChatId: state.currentChatId === chatId ? null : state.currentChatId
  })),

  fetchChats: async (projectId: string) => {
    if (get().currentProjectId !== projectId || !projectId) {
       console.log(`Chat Store: Skipping fetch for ${projectId} as current is ${get().currentProjectId}`);
       if (get().currentProjectId === null && get().chats.length > 0) {
         // Clear chats if project becomes null
         set({ chats: [], currentChatId: null }); 
       }
       return; 
    }
    
    console.log(`Chat Store: Fetching chats for project ${projectId}...`);
    set({ isLoadingChats: true, error: null });
    try {
      const fetchedChats = await listChats(projectId);
      // Ensure we only set chats if the project ID hasn't changed during the fetch
      if (get().currentProjectId === projectId) {
         set({ chats: fetchedChats || [], isLoadingChats: false });
         console.log(`Chat Store: Fetched ${fetchedChats?.length || 0} chats for ${projectId}.`);
      } else {
         console.log(`Chat Store: Project ID changed during fetch for ${projectId}. Discarding results.`);
         set({ isLoadingChats: false }); // Still turn off loading
      }
    } catch (err: any) {
      console.error(`Chat Store: Error fetching chats for ${projectId}:`, err);
      // Ensure we only set error if the project ID hasn't changed
      if (get().currentProjectId === projectId) {
         set({ error: 'Failed to load chats', isLoadingChats: false, chats: [] });
      } else {
         set({ isLoadingChats: false }); // Turn off loading even if project changed
      }
    }
  },

  clearChats: () => {
    set({ chats: [], currentChatId: null, error: null });
    console.log("Chat Store: Cleared all chats.");
  },
}));

// Optional: Initialize: Fetch chats for the default project ID when the store is first used/app loads
// This might need adjustment based on app structure (e.g., call fetchChats in a layout component)
// useChatStore.getState().fetchChats(useChatStore.getState().currentProjectId ?? ""); 