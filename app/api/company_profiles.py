from fastapi import APIRouter, Depends, HTTPException

from app.api.projects import (
    add_conversation,
    add_document,
    create_project,
    get_project,
    get_project_conversations,
    get_project_summary,
    list_projects,
)

# Create an alias router whose endpoints call the same underlying handlers but expose a different prefix
router = APIRouter(prefix="/company-profiles", tags=["company-profiles"])

# Simple wrappers -------------------------------------------------------------


@router.post("/", name="create_company_profile")
async def create_company_profile(*args, **kwargs):
    return await create_project(*args, **kwargs)


@router.get("/", name="list_company_profiles")
async def list_company_profiles(*args, **kwargs):
    return await list_projects(*args, **kwargs)


@router.get("/{company_id}", name="get_company_profile")
async def get_company_profile(company_id: str, **kwargs):
    return await get_project(company_id, **kwargs)


@router.get("/{company_id}/summary", name="company_profile_summary")
async def company_profile_summary(company_id: str, **kwargs):
    return await get_project_summary(company_id, **kwargs)


@router.post("/{company_id}/conversations", name="company_profile_add_conversation")
async def company_profile_add_conversation(company_id: str, *args, **kwargs):
    return await add_conversation(company_id, *args, **kwargs)


@router.get("/{company_id}/conversations", name="company_profile_conversations")
async def company_profile_conversations(company_id: str, **kwargs):
    return await get_project_conversations(company_id, **kwargs)


@router.post("/{company_id}/documents", name="company_profile_add_document")
async def company_profile_add_document(company_id: str, *args, **kwargs):
    return await add_document(company_id, *args, **kwargs)
