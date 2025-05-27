export interface DocumentGenerationConfig {
  deliverable_type: "proposal" | "roi_analysis" | "journey_map" | "cx_strategy";
  parameters: Record<string, any>;
}

export interface QuestionRequest {
  query: string;
  conversation_id?: string | null;
  project_id?: string | null;
  mode: "chat" | "document";
  doc_config?: DocumentGenerationConfig | null;
  active_deliverable_type?: string;
}

export interface SearchResult {
  source: string;
  score: number;
  text_snippet: string;
}

export interface QuestionResponse {
  answer: string;
  conversation_id: string;
  project_id?: string | null;
  sources: SearchResult[];
  processing_time: number;
}

// Optional: Keep these if you might use the dedicated deliverable endpoints for other purposes.
// Otherwise, they can be removed if only /ask with mode: "document" is used.
export interface CXStrategyRequestParams {
  client_name: string;
  industry: string;
  challenges: string;
}

export interface ROIAnalysisRequestParams {
  client_name: string;
  industry: string;
  project_description: string;
  current_metrics: string;
}

export interface JourneyMapRequestParams {
  client_name: string;
  industry: string;
  persona: string;
  scenario: string;
}

// Used by deprecated functions in apiClient.ts, keep if those functions are kept for any reason.
// Also used by the dedicated deliverable endpoints in routes.py
export interface DeliverableResponse {
  content: string;
  conversation_id: string;
  project_id?: string | null;
  document_id?: string | null;
  processing_time: number;
}
