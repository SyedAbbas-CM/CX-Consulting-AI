from pydantic import BaseModel, Field


class AzureOpenAIConfigRequest(BaseModel):
    endpoint: str = Field(
        ...,
        description="Azure OpenAI resource endpoint, e.g. https://my-res.openai.azure.com",
    )
    api_key: str = Field(..., description="Azure OpenAI API key")
    deployment: str = Field(..., description="Deployment name (model) e.g. gpt-4o-mini")
    api_version: str | None = Field(
        None, description="Optional API version, defaults to 2023-12-01-preview"
    )
