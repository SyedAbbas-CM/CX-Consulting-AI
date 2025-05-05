import { create } from 'zustand';
import { Project } from '../types/project';
import { listProjects } from '../lib/apiClient';

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
  },

  setCurrentProjectId: (projectId) => {
    console.log("Project Store: Setting current project ID to:", projectId);
    set({ currentProjectId: projectId });
  }
}));

// Optional: Fetch projects immediately when store is loaded
// useProjectStore.getState().fetchProjects(); 