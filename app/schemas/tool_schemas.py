# app/schemas/tool_schemas.py
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseToolInput(BaseModel):
    """Base model for tool inputs, providing a description for the LLM."""

    # This description can be used in the LLM prompt.
    description: str = Field(
        ..., description="Description of what this tool or deliverable does."
    )


class ResearchSummaryInput(BaseToolInput):
    """
    Inputs for generating a Research Summary Report.
    The LLM should provide content for each of these fields based on the conversation
    or retrieved documents.
    """

    description: str = "Generates a Research Summary Report based on provided insights."
    executive_summary: str = Field(
        ..., description="A brief overview of the research and its main conclusions."
    )
    research_objectives: str = Field(
        ..., description="The goals and aims of the research study."
    )
    methodology: str = Field(
        ...,
        description="How the research was conducted (e.g., surveys, interviews, data analysis).",
    )
    participant_demographics: Optional[str] = Field(
        None,
        description="Description of the participants involved in the research (if applicable).",
    )
    key_findings: str = Field(
        ..., description="The most important discoveries and results from the research."
    )
    detailed_insights: Optional[str] = Field(
        None,
        description="More in-depth explanation of the findings and their implications.",
    )
    customer_quotes: Optional[List[str]] = Field(
        None,
        description="Direct quotes from customers or research participants that highlight key points.",
    )
    patterns_themes: Optional[str] = Field(
        None, description="Recurring patterns or themes observed in the research data."
    )
    recommendations: str = Field(
        ..., description="Actionable recommendations based on the research findings."
    )
    next_steps: Optional[str] = Field(
        None, description="Suggested next actions or future research areas."
    )
    research_materials: Optional[str] = Field(
        None,
        description="Reference to any appendix or source materials used (e.g., survey questions, interview scripts).",
    )

    # To allow the model to be used as a "tool" definition for the LLM
    class Config:
        schema_extra = {
            "tool_name": "research_summary",  # This will be the identifier for the LLM
        }


# We can add more deliverable input schemas here following the same pattern.
# For example:
# class CXStrategyInput(BaseToolInput):
#     description: str = "Creates a comprehensive Customer Experience Strategy document."
#     executive_summary: str = Field(..., description="High-level overview of the CX strategy.")
#     current_state: str = Field(..., description="Assessment of the current customer experience.")
#     # ... add all other fields from the cx_strategy template
#     class Config:
#         schema_extra = {
#             "tool_name": "cx_strategy",
#         }

# A dictionary to hold all tool schemas, which can be used to generate the LLM prompt.
TOOL_SCHEMAS = {
    ResearchSummaryInput.Config.schema_extra["tool_name"]: ResearchSummaryInput
    # CXStrategyInput.Config.schema_extra["tool_name"]: CXStrategyInput,
}


def get_tool_schema_for_llm(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Returns a JSON schema representation of a tool's input model,
    formatted in a way that LLMs can understand for function/tool calling.
    """
    tool_model = TOOL_SCHEMAS.get(tool_name)
    if not tool_model:
        return None

    model_schema = tool_model.model_json_schema()  # Pydantic v2

    # Simplify for LLM: extract name, description, and parameters
    parameters = {
        "type": "object",
        "properties": model_schema.get("properties", {}),
    }
    if "required" in model_schema:
        parameters["required"] = model_schema.get("required")

    return {
        "name": tool_name,
        "description": tool_model.model_fields[
            "description"
        ].default,  # Get default from Pydantic field
        "parameters": parameters,
    }


def get_all_tool_schemas_for_llm() -> List[Dict[str, Any]]:
    """
    Returns a list of all tool schemas formatted for LLM tool/function calling.
    """
    return [
        get_tool_schema_for_llm(name)
        for name in TOOL_SCHEMAS.keys()
        if get_tool_schema_for_llm(name) is not None
    ]
