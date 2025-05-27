export interface LlmConfigResponse {
    backend: string;
    model_id?: string | null;
    model_path?: string | null;
    max_model_len: number;
    gpu_count: number;
}
