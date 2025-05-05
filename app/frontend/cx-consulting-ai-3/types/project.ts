export interface ProjectDocument {
  id: string;           // Or document_id?
  title: string;
  filename?: string;      // Made optional, update if always present
  document_type?: string; // Made optional
  created_at: string;    // ISO String
  // Add other relevant fields based on backend ProjectDocument schema
}

export interface Project {
  id: string;
  name: string;
  client_name?: string;
  industry?: string;
  description?: string;
  created_at: string;    // ISO String
  updated_at: string;    // ISO String
  // Assuming documents might be nested or fetched separately
  // documents?: ProjectDocument[];
  // conversations?: any[]; // Or ConversationSummary[]
}

export interface ProjectCreateRequest {
  name: string;
  description?: string;
  client_name: string;
  industry: string;
  metadata?: Record<string, any>;
}

// Added export for listing projects
export interface ProjectsResponse {
  projects: Project[];
}

// Added export for listing project documents
export interface ProjectDocumentsResponse {
  documents: ProjectDocument[];
} 