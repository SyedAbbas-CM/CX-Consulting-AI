import { create } from 'zustand';
import { Project } from '../types/project';
import { listProjects } from '../lib/apiClient';
import { useChatStore } from './chatStore';
import { useDocumentStore } from './documentStore';

interface ProjectState {
  projects: Project[];
  currentProjectId: string | null;
  isLoadingProjects: boolean;
  error: string | null;
  fetchProjects: () => Promise<void>;
  addProject: (newProject: Project) => void;
  removeProject: (projectId: string) => void;
  setCurrentProjectId: (projectId: string | null) => void;
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  projects: [],
  currentProjectId: null, // No default project selected initially
  isLoadingProjects: false,
  error: null,

  fetchProjects: async () => {
    set({ isLoadingProjects: true, error: null });
    console.log("Project Store: Fetching projects...");
    try {
      const data = await listProjects();
      set({ projects: data.projects, isLoadingProjects: false });
      console.log(`Project Store: Fetched ${data.projects.length} projects.`);
      // If no project is currently selected, select the first one by default?
      // if (!get().currentProjectId && data.projects.length > 0) {
      //   set({ currentProjectId: data.projects[0].id });
      // }
    } catch (error) {
      console.error("Failed to fetch projects:", error);
      set({ isLoadingProjects: false, projects: [], error: 'Failed to load projects' });
    }
  },

  addProject: (newProject) => set((state) => ({
    projects: [newProject, ...state.projects] // Add to beginning
  })),

  removeProject: (projectId) => {
    set((state) => ({
      projects: state.projects.filter((p) => p.id !== projectId),
      currentProjectId: state.currentProjectId === projectId ? null : state.currentProjectId,
    }));
    // If the deleted project was the current one, also clear chats in chatStore
    const { currentProjectId: currentChatStoreProjectId, setProjectId: setChatStoreProjectId } = useChatStore.getState();
    if (currentChatStoreProjectId === projectId) {
      setChatStoreProjectId(null);
    }
  },

  setCurrentProjectId: (projectId) => {
    console.log("Project Store: Setting current project ID to:", projectId);
    set({ currentProjectId: projectId });
    // Notify ChatStore about the project change
    useChatStore.getState().setProjectId(projectId);
    // Notify DocumentStore about the project change
    useDocumentStore.getState().setProjectId(projectId);

    // Trigger fetch chats immediately (Alternatively, UI component can trigger fetch based on ID change)
    if (projectId) {
      // Use timeout to allow state update to propagate before fetching
      setTimeout(() => {
        useChatStore.getState().fetchChats(projectId);
        // Documents are fetched by DocumentStore's setProjectId automatically
      }, 0);
    } else {
       useChatStore.getState().clearChats(); // Clear chats if no project selected
       useDocumentStore.getState().clearDocuments(); // Clear documents if no project selected
    }
  }
}));

// Optional: Fetch projects immediately when store is loaded
// useProjectStore.getState().fetchProjects();
