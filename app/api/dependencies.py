# app/api/dependencies.py
import logging
import os
from functools import lru_cache
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status

from app.agents.agent_runner import AgentRunner
from app.api.auth import get_current_user
from app.core.config import Settings, get_settings
from app.core.llm_service import LLMService
from app.services.auth_service import AuthService
from app.services.auth_service import User as AuthUser
from app.services.chat_service import ChatService
from app.services.context_optimizer import ContextOptimizer
from app.services.deliverable_service import DeliverableService
from app.services.document_service import DocumentService
from app.services.project_manager import ProjectManager
from app.services.project_memory_service import ProjectMemoryService
from app.services.rag_engine import RagEngine
from app.services.template_service import TemplateService
from app.template_wrappers.prompt_template import PromptTemplateManager

logger = logging.getLogger("cx_consulting_ai.dependencies")

# Remove global singleton variables like _llm_service, _document_service, etc.
# They will now be managed by app.state in main.py's startup_event.


def get_llm_service(request: Request) -> LLMService:
    if (
        not hasattr(request.app.state, "llm_service")
        or request.app.state.llm_service is None
    ):
        logger.critical("LLMService not found on app.state.")
        raise RuntimeError(
            "LLMService not available. Application did not start correctly."
        )
    return request.app.state.llm_service


def get_document_service(request: Request) -> DocumentService:
    if (
        not hasattr(request.app.state, "document_service")
        or request.app.state.document_service is None
    ):
        logger.critical("DocumentService not found on app.state.")
        raise RuntimeError(
            "DocumentService not available. Application did not start correctly."
        )
    return request.app.state.document_service


def get_template_manager(request: Request) -> PromptTemplateManager:
    if (
        not hasattr(request.app.state, "template_manager")
        or request.app.state.template_manager is None
    ):
        logger.critical("PromptTemplateManager not found on app.state.")
        raise RuntimeError(
            "PromptTemplateManager not available. Application did not start correctly."
        )
    return request.app.state.template_manager


def get_context_optimizer(request: Request) -> ContextOptimizer:
    if (
        not hasattr(request.app.state, "context_optimizer")
        or request.app.state.context_optimizer is None
    ):
        logger.critical("ContextOptimizer not found on app.state.")
        raise RuntimeError(
            "ContextOptimizer not available. Application did not start correctly."
        )
    return request.app.state.context_optimizer


def get_chat_service(request: Request) -> ChatService:
    if (
        not hasattr(request.app.state, "chat_service")
        or request.app.state.chat_service is None
    ):
        logger.critical("ChatService not found on app.state.")
        raise RuntimeError(
            "ChatService not available. Application did not start correctly."
        )
    return request.app.state.chat_service


def get_project_manager(request: Request) -> ProjectManager:
    if (
        not hasattr(request.app.state, "project_manager")
        or request.app.state.project_manager is None
    ):
        logger.critical("ProjectManager not found on app.state.")
        raise RuntimeError(
            "ProjectManager not available. Application did not start correctly."
        )
    return request.app.state.project_manager


def get_template_service(request: Request) -> TemplateService:
    if (
        not hasattr(request.app.state, "template_service")
        or request.app.state.template_service is None
    ):
        logger.critical("TemplateService not found on app.state.")
        raise RuntimeError(
            "TemplateService not available. Application did not start correctly."
        )
    return request.app.state.template_service


# --- Composite service getters refactored to fetch from app.state ---


def get_project_memory_service(request: Request) -> ProjectMemoryService:
    if (
        not hasattr(request.app.state, "project_memory_service")
        or request.app.state.project_memory_service is None
    ):
        logger.critical(
            "ProjectMemoryService not found on app.state. It should be initialized in main.py startup_event."
        )
        raise RuntimeError(
            "ProjectMemoryService not available. Application did not start correctly."
        )
    return request.app.state.project_memory_service


def get_deliverable_service(request: Request) -> DeliverableService:
    if (
        not hasattr(request.app.state, "deliverable_service")
        or request.app.state.deliverable_service is None
    ):
        logger.critical(
            "DeliverableService not found on app.state. It should be initialized in main.py startup_event."
        )
        raise RuntimeError(
            "DeliverableService not available. Application did not start correctly."
        )
    return request.app.state.deliverable_service


def get_rag_engine(request: Request) -> RagEngine:
    if (
        not hasattr(request.app.state, "rag_engine")
        or request.app.state.rag_engine is None
    ):
        logger.critical(
            "RagEngine not found on app.state. It should be initialized in main.py startup_event."
        )
        raise RuntimeError(
            "RagEngine not available. Application did not start correctly."
        )
    return request.app.state.rag_engine


def get_auth_service(request: Request) -> AuthService:
    if (
        not hasattr(request.app.state, "auth_service")
        or request.app.state.auth_service is None
    ):
        logger.critical("AuthService not found on app.state.")
        raise RuntimeError(
            "AuthService not available. Application did not start correctly."
        )
    return request.app.state.auth_service


# --- Added for Workstream 4: AgentRunner Dependency ---
def get_agent_runner(request: Request) -> AgentRunner:
    if (
        not hasattr(request.app.state, "agent_runner")
        or request.app.state.agent_runner is None
    ):
        # AgentRunner is not a singleton like other services in app.state currently in main.py
        # It's composed of other services. We can instantiate it here.
        # Or, it could be added to app.state in main.py if it's meant to be a shared instance.
        # For now, instantiating it on demand using other singleton services from app.state.
        logger.info("Creating new AgentRunner instance via dependency function.")
        llm_service = get_llm_service(request)
        document_service = get_document_service(
            request
        )  # This is the new searcher for RetrieverAgent
        chat_service = get_chat_service(request)
        project_manager = get_project_manager(request)
        prompt_template_manager = get_template_manager(
            request
        )  # Get PromptTemplateManager
        app_settings = get_settings()  # Get global app settings

        # HierarchicalSearcher is no longer used directly by AgentRunner;
        # RetrieverAgent now takes DocumentService.
        # If other agents needed HierarchicalSearcher, this would be an issue.
        # For now, assuming DocumentService is sufficient for RetrieverAgent via AgentRunner.

        request.app.state.agent_runner = AgentRunner(
            llm_service=llm_service,
            document_service=document_service,
            chat_service=chat_service,
            project_manager=project_manager,
            prompt_template_manager=prompt_template_manager,  # Pass it
            settings=app_settings.model_dump(),  # Pass settings as dict
        )
        logger.info(
            "AgentRunner instance created and cached on app.state for this request cycle, or if already present."
        )
    elif not isinstance(request.app.state.agent_runner, AgentRunner):
        # This case handles if app.state.agent_runner was somehow set but not to an AgentRunner instance
        # Or if we decide to make it a true singleton initialized at startup.
        # For now, this indicates an unexpected state if reached after the first if-branch.
        logger.error(
            f"app.state.agent_runner is not a valid AgentRunner instance. Type: {type(request.app.state.agent_runner)}. Re-initializing."
        )
        # Fallback to re-initialize, though this might hide setup issues.
        llm_service = get_llm_service(request)
        document_service = get_document_service(request)
        chat_service = get_chat_service(request)
        project_manager = get_project_manager(request)
        prompt_template_manager = get_template_manager(
            request
        )  # Get PromptTemplateManager
        app_settings = get_settings()  # Get global app settings
        request.app.state.agent_runner = AgentRunner(
            llm_service=llm_service,
            document_service=document_service,
            chat_service=chat_service,
            project_manager=project_manager,
            prompt_template_manager=prompt_template_manager,  # Pass it
            settings=app_settings.model_dump(),  # Pass settings as dict
        )

    return request.app.state.agent_runner


# --- End Workstream 4 ---


# Typedef for dependencies
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
TemplateManagerDep = Annotated[PromptTemplateManager, Depends(get_template_manager)]
ContextOptimizerDep = Annotated[ContextOptimizer, Depends(get_context_optimizer)]
TemplateServiceDep = Annotated[TemplateService, Depends(get_template_service)]
ProjectMemoryServiceDep = Annotated[
    ProjectMemoryService, Depends(get_project_memory_service)
]
DeliverableServiceDep = Annotated[DeliverableService, Depends(get_deliverable_service)]
RagEngineDep = Annotated[RagEngine, Depends(get_rag_engine)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
ProjectManagerDep = Annotated[ProjectManager, Depends(get_project_manager)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
CurrentUserDep = Annotated[dict, Depends(get_current_user)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]

# --- Added for Workstream 4 ---
AgentRunnerDep = Annotated[AgentRunner, Depends(get_agent_runner)]
# --- End Workstream 4 ---


# --- new current_user_dep function ---
async def current_user_dep(
    request: Request,
    auth_service: AuthServiceDep,  # Use the existing AuthServiceDep
) -> AuthUser:
    # Assuming get_current_user_or_raise is a method on AuthService instance
    # and it correctly handles the request object if needed, or just the token from it.
    # The original get_current_user_or_raise in auth_service.py takes a token.
    # We need to extract the token from the request here if that's what it expects.
    # For now, let's assume get_current_user_or_raise can derive token from request or is passed it.
    # This part might need adjustment based on the exact signature of get_current_user_or_raise.
    # Let's re-check auth_service.py for that method's signature.
    # For now, passing request, assuming the method handles it.
    token = request.headers.get("Authorization")
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]
    else:
        # This case should ideally be handled by OAuth2PasswordBearer if it was used directly
        # in get_current_user_or_raise, but since we are building it manually:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_service.get_current_user_or_raise(
        token=token
    )  # Pass the token explicitly


AuthenticatedUserDep = Annotated[AuthUser, Depends(current_user_dep)]


# Health check dependency
