export interface Message {
    id: string; // Or number
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp?: string; // Optional timestamp
    // Add other relevant message properties if needed
}

export interface ChatCreateRequest {
    project_id: string | number; // Assuming project ID is needed
    first_message_content: string;
}

export interface ChatCreateResponse {
    chat_id: string; // Or number
    first_message: Message;
    // Potentially include project info or other relevant data
}

export interface ChatSummaryResponse {
    id: string; // Or number
    title: string; // Often the first user message or a generated summary
    created_at: string;
    last_updated_at: string;
    project_id: string | number;
}

// Assuming this is related to refining a specific message or the chat context
export interface RefinementRequest {
    chat_id: string | number;
    message_id?: string | number; // Optional, if refining a specific message
    refinement_prompt: string;
    context?: string; // Optional additional context
}

export interface RefinementResponse {
    refined_content: string;
    chat_id: string | number;
    message_id?: string | number;
    // Include other relevant data, like tokens used, etc.
} 