// apiClient.ts – single‑source API wrapper (no duplicates, strict typing)
// --------------------------------------------------------------
// NOTE: Adjust type imports if your project already declares
//       ChatHistoryResponse or ModelListResponse elsewhere.
//       The inline definitions here prevent TS errors out‑of‑the‑box.
// --------------------------------------------------------------

// **Imports** --------------------------------------------------
import {
    Project,
    ProjectCreateRequest,
    ProjectsResponse,
    ProjectDocumentsResponse,
    ProjectDocument,
  } from "../types/project";
  import {
    ChatSummaryResponse,
    Message,
    ChatCreateRequest,
  } from "../types/chat";
  import {
    DeliverableResponse,
  } from "../types/deliverables";
  import { LlmConfigResponse } from "../types/config";
  import { ModelInfo, ModelStatus } from "../types/model";
  import { QuestionRequest, QuestionResponse } from "@/src/types/api";

  // **Fallback type aliases (remove if you have real ones)** ------
  export interface ChatHistoryResponse {
    messages: Message[];
  }
  export interface ModelListResponse {
    available_models: ModelInfo[];
    active_model_path: string;
  }

  // **Refinement types** -----------------------------------------
  export interface RefinementPayload {
    message: string;
    interaction_id: string;
    file_path: string;
  }
  export interface RefinementInteraction {
    id: string;
    chat_id: string;
    user_query: string;
    assistant_response: string;
    user_message_timestamp?: string;
    assistant_message_timestamp?: string;
    marked_at: string;
    marked_by: string;
    context?: unknown[];
  }
  export interface RefinementListResponse {
    interactions: RefinementInteraction[];
    total: number;
  }

  // **Globals & helpers** ----------------------------------------
  const API_BASE_URL =
    process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") || "/api";

  const VERY_LONG_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes for RAG/Agent calls

  const getAuthToken = (): string | null =>
    typeof window !== "undefined" ? localStorage.getItem("authToken") : null;

  const fetchWithAuth = async (
    url: string,
    options: RequestInit = {},
    timeoutMs: number = VERY_LONG_TIMEOUT_MS // Add a default timeout, can be overridden per call if needed
  ): Promise<Response> => {
    const token = getAuthToken();
    const headers = new Headers(options.headers || {});

    if (token) headers.set("Authorization", `Bearer ${token}`);
    if (!(options.body instanceof FormData) && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
        console.warn(`API call to ${url} timed out after ${timeoutMs / 1000}s (frontend defined)`);
        controller.abort();
    }, timeoutMs);

    try {
        const response = await fetch(url, { ...options, headers, signal: controller.signal });
        clearTimeout(timeoutId); // Clear timeout if response received

        if (!response.ok) {
          let errorData: any = { detail: response.statusText };
          try {
            errorData = await response.json();
          } catch (_) {/* ignore */}
          const err: any = new Error(errorData.detail || "API request failed without specific detail");
          err.status = response.status;
          err.errorData = errorData;
          throw err;
        }
        return response;
    } catch (error: any) {
        clearTimeout(timeoutId); // Ensure timeout is cleared on any error
        if (error.name === 'AbortError') {
            // Re-throw a more specific error for timeouts
            const timeoutError: any = new Error(`Request to ${url} timed out after ${timeoutMs / 1000}s.`);
            timeoutError.status = 408; // Request Timeout status
            timeoutError.errorData = { detail: `Frontend timeout: ${timeoutMs / 1000}s` };
            throw timeoutError;
        }
        throw error; // Re-throw other errors
    }
  };

  async function handleResponse<T>(resp: Response): Promise<T> {
    // Some endpoints (DELETE) return 204 w/ empty body
    const text = await resp.text();
    if (text.trim() === "") return undefined as unknown as T;
    try {
      return JSON.parse(text) as T;
    } catch (_) {
      throw new Error("Failed to parse JSON from server.");
    }
  }

  // **Project endpoints** ----------------------------------------
  export async function listProjects(
    limit = 50,
    offset = 0,
  ): Promise<ProjectsResponse> {
    const r = await fetchWithAuth(
      `${API_BASE_URL}/projects?limit=${limit}&offset=${offset}`,
    );
    return (await handleResponse<ProjectsResponse>(r)) ?? {
      projects: [],
      count: 0,
    };
  }

  export async function createProject(
    payload: ProjectCreateRequest,
  ): Promise<Project> {
    const r = await fetchWithAuth(`${API_BASE_URL}/projects`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    return handleResponse<Project>(r);
  }

  export async function deleteProject(projectId: string): Promise<void> {
    await fetchWithAuth(`${API_BASE_URL}/projects/${projectId}`, {
      method: "DELETE",
    });
  }

  // **Chat endpoints** -------------------------------------------
  export async function listChats(
    projectId: string,
    limit = 50,
    offset = 0,
  ): Promise<ChatSummaryResponse[]> {
    const r = await fetchWithAuth(
      `${API_BASE_URL}/projects/${projectId}/chats?limit=${limit}&offset=${offset}`,
    );
    return (await handleResponse<ChatSummaryResponse[]>(r)) ?? [];
  }

  export async function createChat(
    projectId: string,
    data?: ChatCreateRequest,
  ): Promise<ChatSummaryResponse> {
    const r = await fetchWithAuth(`${API_BASE_URL}/projects/${projectId}/chats`, {
      method: "POST",
      body: data ? JSON.stringify(data) : undefined,
    });
    return handleResponse<ChatSummaryResponse>(r);
  }

  export async function getChatHistory(
    chatId: string,
    limit = 100,
    offset = 0,
  ): Promise<Message[]> {
    const r = await fetchWithAuth(
      `${API_BASE_URL}/chats/${chatId}/history?limit=${limit}&offset=${offset}`,
    );
    return (
      (await handleResponse<ChatHistoryResponse>(r))?.messages ?? []
    );
  }

  export async function deleteChat(projectId: string, chatId: string): Promise<void> {
    await fetchWithAuth(`${API_BASE_URL}/projects/${projectId}/chats/${chatId}`, { method: "DELETE" });
  }

  // **Document endpoints** ---------------------------------------
  export async function uploadDocument(
    file: File,
    projectId?: string,
    isGlobal = false,
  ): Promise<any> {
    const form = new FormData();
    form.append("file", file);
    if (projectId) form.append("project_id", projectId);
    form.append("is_global", String(isGlobal));

    const r = await fetchWithAuth(`${API_BASE_URL}/documents`, {
      method: "POST",
      body: form,
    });
    return handleResponse<any>(r);
  }

  export async function listProjectDocuments(
    projectId: string,
  ): Promise<ProjectDocumentsResponse> {
    const r = await fetchWithAuth(
      `${API_BASE_URL}/projects/${projectId}/documents`,
    );
    return (
      (await handleResponse<ProjectDocumentsResponse>(r)) ?? {
        documents: [],
        count: 0,
      }
    );
  }

  export async function deleteDocument(documentId: string): Promise<void> {
    await fetchWithAuth(`${API_BASE_URL}/documents/${documentId}`, {
      method: "DELETE",
    });
  }

  export async function getDocument(
    documentId: string,
  ): Promise<ProjectDocument | null> {
    const r = await fetchWithAuth(
      `${API_BASE_URL}/documents/${documentId}`,
    );
    return handleResponse<ProjectDocument | null>(r);
  }

  export async function updateDocumentContent(
    documentId: string,
    content: string,
  ): Promise<ProjectDocument | null> {
    const r = await fetchWithAuth(
      `${API_BASE_URL}/documents/${documentId}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }), // Send only the content for update
      },
    );
    return handleResponse<ProjectDocument | null>(r);
  }

  export async function refineDocument(
    documentId: string,
    prompt: string,
    replaceEmbeddings: boolean = false,
  ): Promise<ProjectDocument | null> {
    const r = await fetchWithAuth(
      `${API_BASE_URL}/documents/${documentId}/refine`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, replace_embeddings: replaceEmbeddings }),
      },
    );
    return handleResponse<ProjectDocument | null>(r);
  }

  // **LLM config** -----------------------------------------------
  export async function getLlmConfig(): Promise<LlmConfigResponse> {
    const r = await fetchWithAuth(`${API_BASE_URL}/config/llm`);
    return handleResponse<LlmConfigResponse>(r);
  }

  // **Refinement** -----------------------------------------------
  export async function markInteractionForRefinement(
    chatId: string,
    messageIndex: number,
  ): Promise<RefinementPayload> {
    const r = await fetchWithAuth(
      `${API_BASE_URL}/chats/${chatId}/messages/${messageIndex}/refine`,
      { method: "POST" },
    );
    return handleResponse<RefinementPayload>(r);
  }

  export async function getRefinementInteractions(
    limit = 100,
    offset = 0,
  ): Promise<RefinementListResponse> {
    const r = await fetchWithAuth(
      `${API_BASE_URL}/improvement/interactions?limit=${limit}&offset=${offset}`,
    );
    return handleResponse<RefinementListResponse>(r);
  }

  // **Model management** -----------------------------------------
  export async function listModels(): Promise<ModelListResponse> {
    const r = await fetchWithAuth(`${API_BASE_URL}/models`);
    return handleResponse<ModelListResponse>(r);
  }

  export async function downloadModel(
    modelId: string,
    force = false,
  ): Promise<{ message: string }> {
    const r = await fetchWithAuth(`${API_BASE_URL}/models/download`, {
      method: "POST",
      body: JSON.stringify({ model_id: modelId, force_download: force }),
    });
    return handleResponse<{ message: string }>(r);
  }

  export async function setActiveModel(
    modelId: string,
  ): Promise<{ message: string }> {
    const r = await fetchWithAuth(`${API_BASE_URL}/models/set_active`, {
      method: "POST",
      body: JSON.stringify({ model_id: modelId }),
    });
    return handleResponse<{ message: string }>(r);
  }

  export async function getModelStatus(modelId: string): Promise<ModelStatus> {
    const r = await fetchWithAuth(`${API_BASE_URL}/models/${modelId}/status`);
    return handleResponse<ModelStatus>(r);
  }

  // **DEPRECATED Deliverable Endpoints** -------------------------
  // These functions call the old dedicated endpoints.
  // For new UI flows generating deliverables via chat/modal,
  // use askQuestion({ mode: "document", doc_config: {...} })

  /**
   * @deprecated Use askQuestion with mode: "document" and appropriate doc_config instead.
   */
  export async function generateProposal(
    payload: any, // Replace 'any' with actual CXStrategyRequest from ../types/deliverables if keeping
  ): Promise<DeliverableResponse> {
    console.warn("generateProposal is deprecated. Use askQuestion with mode: \"document\".")
    const r = await fetchWithAuth(`${API_BASE_URL}/cx-strategy`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    return handleResponse<DeliverableResponse>(r);
  }

  /**
   * @deprecated Use askQuestion with mode: "document" and appropriate doc_config instead.
   */
  export async function generateRoiAnalysis(
    payload: any, // Replace 'any' with actual ROIAnalysisRequest from ../types/deliverables if keeping
  ): Promise<DeliverableResponse> {
    console.warn("generateRoiAnalysis is deprecated. Use askQuestion with mode: \"document\".")
    const r = await fetchWithAuth(`${API_BASE_URL}/roi-analysis`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    return handleResponse<DeliverableResponse>(r);
  }

  /**
   * @deprecated Use askQuestion with mode: "document" and appropriate doc_config instead.
   */
  export async function generateJourneyMap(
    payload: any, // Replace 'any' with actual JourneyMapRequest from ../types/deliverables if keeping
  ): Promise<DeliverableResponse> {
    console.warn("generateJourneyMap is deprecated. Use askQuestion with mode: \"document\".")
    const r = await fetchWithAuth(`${API_BASE_URL}/journey-map`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    return handleResponse<DeliverableResponse>(r);
  }

  // **Core QA and Document Generation endpoint** -----------------
  // Removed old inline AskRequestPayload and AskResponse interfaces

  export async function askQuestion(payload: QuestionRequest): Promise<QuestionResponse> {
    const r = await fetchWithAuth(`${API_BASE_URL}/ask`, {
      method: "POST",
      body: JSON.stringify(payload),
      // Using the default VERY_LONG_TIMEOUT_MS for RAG/Agent calls from fetchWithAuth
    });
    return handleResponse<QuestionResponse>(r);
  }

  // --------------------------------------------------------------
  //  End of apiClient.ts
  // --------------------------------------------------------------
