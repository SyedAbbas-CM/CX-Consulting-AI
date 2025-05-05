// types/deliverables.ts

// Matches backend app/api/models.py

export interface CXStrategyRequest {
  client_name: string;
  industry: string;
  challenges: string;
  conversation_id?: string | null;
  project_id?: string | null;
}

export interface ROIAnalysisRequest {
  client_name: string;
  industry: string;
  project_description: string;
  current_metrics: string;
  conversation_id?: string | null;
  project_id?: string | null;
}

export interface JourneyMapRequest {
  client_name: string; // Added based on usage in routes
  industry: string;    // Added based on usage in routes
  persona: string;
  scenario: string;
  conversation_id?: string | null;
  project_id?: string | null;
}

export interface DeliverableResponse {
  content: string;
  conversation_id: string;
  project_id?: string | null;
  document_id?: string | null;
  processing_time: number;
} 