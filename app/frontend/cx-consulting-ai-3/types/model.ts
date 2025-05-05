// types/model.ts

// Corresponds to backend app/schemas/model.py
export interface LlmConfigResponse {
    backend: string;
    model_id?: string | null;
    model_path?: string | null;
    max_model_len: number;
    gpu_count: number;
}

// You can also define ModelActionRequest here if needed by frontend
export interface ModelActionRequest {
    model_id: string;
    force_download?: boolean;
}

export interface ModelInfo {
    id: string;
    name: string;
    size_gb?: number;
    description?: string;
    status?: string;
    message?: string;
    path?: string;
    downloaded?: boolean;
    // Add other relevant properties like quantization, family, etc.
}

export interface ModelStatus {
    model_id: string;
    status: 'downloading' | 'available' | 'unavailable' | 'error' | 'pending';
    download_progress?: number; // Optional: 0-100
    error_message?: string; // Optional: message if status is 'error'
} 