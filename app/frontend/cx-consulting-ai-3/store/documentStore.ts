import { create } from 'zustand';
import { ProjectDocument } from '../types/project';
import {
  listProjectDocuments,
  getDocument,
  updateDocumentContent,
  refineDocument
} from '../lib/apiClient';

interface DocumentState {
  currentProjectId: string | null;
  documents: ProjectDocument[];
  currentDocument: ProjectDocument | null;
  currentDocumentContent: string | null; // To store fetched content
  isLoadingDocuments: boolean;
  isLoadingContent: boolean;
  error: string | null;
  setProjectId: (projectId: string | null) => void;
  fetchDocumentsForProject: (projectId: string) => Promise<void>;
  setCurrentDocumentById: (documentId: string) => Promise<void>; // Fetches content too
  updateCurrentDocumentContent: (documentId: string, newContent: string) => Promise<void>; // New action
  refineCurrentDocument: (prompt: string, replaceEmbeddings?: boolean) => Promise<void>; // New action
  clearDocuments: () => void;
}

export const useDocumentStore = create<DocumentState>((set, get) => ({
  currentProjectId: null,
  documents: [],
  currentDocument: null,
  currentDocumentContent: null,
  isLoadingDocuments: false,
  isLoadingContent: false,
  error: null,

  setProjectId: (projectId) => {
    if (get().currentProjectId !== projectId) {
      set({
        currentProjectId: projectId,
        documents: [],
        currentDocument: null,
        currentDocumentContent: null,
        isLoadingDocuments: false,
        isLoadingContent: false,
        error: null
      });
      console.log(`Document Store: Project ID set to ${projectId}, documents cleared.`);
      if (projectId) {
        get().fetchDocumentsForProject(projectId);
      }
    }
  },

  fetchDocumentsForProject: async (projectId: string) => {
    if (!projectId) {
      set({ documents: [], isLoadingDocuments: false, currentDocument: null, currentDocumentContent: null });
      return;
    }
    console.log(`Document Store: Fetching documents for project ${projectId}...`);
    set({ isLoadingDocuments: true, error: null, currentDocument: null, currentDocumentContent: null });
    try {
      const response = await listProjectDocuments(projectId);
      // Ensure we only set documents if the project ID hasn't changed during the fetch
      if (get().currentProjectId === projectId) {
         set({ documents: response.documents || [], isLoadingDocuments: false });
         console.log(`Document Store: Fetched ${response.documents?.length || 0} documents for ${projectId}.`);
      } else {
         console.log(`Document Store: Project ID changed during fetch for ${projectId}. Discarding results.`);
         set({ isLoadingDocuments: false }); // Still turn off loading
      }
    } catch (err: any) {
      console.error(`Document Store: Error fetching documents for ${projectId}:`, err);
      if (get().currentProjectId === projectId) {
         set({ error: 'Failed to load documents', isLoadingDocuments: false, documents: [] });
      } else {
         set({ isLoadingDocuments: false });
      }
    }
  },

  setCurrentDocumentById: async (documentId: string) => {
    const currentDocs = get().documents;
    // Try to find metadata from the list first, but we will fetch full content anyway
    const selectedDocMetadata = currentDocs.find(doc => doc.id === documentId) || null;

    // If it's already current and content is loaded or loading, do nothing
    if (selectedDocMetadata && selectedDocMetadata.id === get().currentDocument?.id && (get().currentDocumentContent || get().isLoadingContent)) {
        console.log(`Document Store: Document ${documentId} is already current/loading.`);
        return;
    }

    set({ currentDocument: selectedDocMetadata, currentDocumentContent: null, isLoadingContent: true, error: null });

    if (documentId) { // We need a documentId to fetch
      console.log(`Document Store: Fetching content for document ${documentId}...`);
      try {
        const fullDoc = await getDocument(documentId);
        if (fullDoc) {
          set({
            currentDocument: fullDoc,
            currentDocumentContent: fullDoc.content || 'Content not available.',
            isLoadingContent: false
          });
          console.log(`Document Store: Content loaded for document ${documentId}.`);
        } else {
          throw new Error("Document not found after fetch.");
        }
      } catch (err: any) {
        console.error("Document Store: Failed to fetch document content:", err);
        set({ error: err.message || 'Failed to load document content', isLoadingContent: false, currentDocument: selectedDocMetadata, currentDocumentContent: null });
      }
    } else {
      // This case should ideally not be reached if documentId is always passed from a valid selection
      console.log(`Document Store: No document ID provided to setCurrentDocumentById.`);
      set({ currentDocument: null, currentDocumentContent: null, isLoadingContent: false });
    }
  },

  updateCurrentDocumentContent: async (documentId: string, newContent: string) => {
    const currentDoc = get().currentDocument;
    if (!currentDoc || currentDoc.id !== documentId) {
      console.warn("Document Store: Attempted to update content for a document that is not current.");
      return;
    }

    set({ isLoadingContent: true, error: null }); // Indicate loading state for saving
    try {
      const updatedDoc = await updateDocumentContent(documentId, newContent);
      if (updatedDoc) {
        set({
          currentDocument: updatedDoc,
          currentDocumentContent: updatedDoc.content || '',
          isLoadingContent: false,
          // Optionally, update the document in the main `documents` list as well
          documents: get().documents.map(doc => doc.id === documentId ? updatedDoc : doc)
        });
        console.log("Document Store: Document content updated successfully.");
      } else {
        throw new Error("Failed to update document on the server.");
      }
    } catch (err: any) {
      console.error("Document Store: Failed to update document content:", err);
      set({ error: err.message || 'Failed to save document content', isLoadingContent: false });
    }
  },

  refineCurrentDocument: async (prompt: string, replaceEmbeddings: boolean = false) => {
    const currentDoc = get().currentDocument;
    if (!currentDoc || !currentDoc.id) {
      set({ error: "No current document selected to refine." });
      return;
    }
    set({ isLoadingContent: true, error: null }); // Indicate loading for refinement
    try {
      const updatedDoc = await refineDocument(currentDoc.id, prompt, replaceEmbeddings);
      if (updatedDoc) {
        set({
          currentDocument: updatedDoc,
          currentDocumentContent: updatedDoc.content, // Update content as well
          documents: get().documents.map(doc => doc.id === updatedDoc.id ? updatedDoc : doc),
          isLoadingContent: false,
        });
      } else {
        throw new Error("Refinement returned no document data.");
      }
    } catch (err: any) {
      set({ error: err.message || "Failed to refine document", isLoadingContent: false });
    }
  },

  clearDocuments: () => {
    set({
      documents: [],
      currentDocument: null,
      currentDocumentContent: null,
      isLoadingDocuments: false,
      isLoadingContent: false,
      error: null
    });
    console.log("Document Store: Cleared all documents.");
  },
}));

// O3/User Fix: When currentProjectId in ProjectStore changes, update DocumentStore's project ID.
// This can be done by subscribing to projectStore changes or by calling documentStore.setProjectId
// from projectStore's setCurrentProjectId method.
// For simplicity, we'll assume projectStore's setCurrentProjectId will call documentStore.setProjectId.
// See projectStore.ts for the corresponding call.
