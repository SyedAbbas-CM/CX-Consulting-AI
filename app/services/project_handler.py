import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.services.chat_service import ChatService

# Configure logger
logger = logging.getLogger("cx_consulting_ai.project_handler")


class ProjectHandler:
    """
    Handle project-related operations including creation, retrieval, and management
    of project-specific data and conversations.
    """

    def __init__(self, chat_service: ChatService, document_service=None):
        """
        Initialize the project handler.

        Args:
            chat_service: Chat service for conversation tracking
            document_service: Optional document service for project-related documents
        """
        self.chat_service = chat_service
        self.document_service = document_service
        self.projects = {}  # In-memory store of projects

        # Initialize from environment if available
        from app.core.config import get_settings

        settings = get_settings()
        self.projects_directory = settings.PROJECT_DIR

        # Create projects directory if it doesn't exist
        os.makedirs(self.projects_directory, exist_ok=True)

        # Load existing projects
        self._load_projects()

        logger.info("ProjectHandler initialized")

    def _load_projects(self):
        """Load existing projects from the projects directory."""
        try:
            for project_dir_name in os.listdir(self.projects_directory):
                project_path = os.path.join(self.projects_directory, project_dir_name)

                # Skip non-directories
                if not os.path.isdir(project_path):
                    continue

                # Check for project config file
                config_path = os.path.join(project_path, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            project_data = json.load(f)
                            # Basic validation
                            if "id" in project_data:
                                self.projects[project_data["id"]] = project_data
                            else:
                                logger.warning(
                                    f"Skipping config file without id in {project_path}"
                                )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Skipping invalid JSON config file: {config_path}"
                        )
                    except Exception as e_inner:
                        logger.warning(
                            f"Error loading project config {config_path}: {e_inner}"
                        )

            logger.info(f"Loaded {len(self.projects)} existing projects")
        except Exception as e:
            logger.error(
                f"Error listing projects directory {self.projects_directory}: {str(e)}"
            )

    def create_project(
        self, name: str, description: str = "", metadata: dict = None
    ) -> dict:
        """
        Create a new project.

        Args:
            name: Project name
            description: Project description
            metadata: Additional project metadata

        Returns:
            Project data dictionary
        """
        project_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()

        project_data = {
            "id": project_id,
            "name": name,
            "description": description,
            "created_at": created_at,
            "updated_at": created_at,
            "metadata": metadata or {},
            "conversations": [],
            "documents": [],
        }

        # Create project directory
        project_dir = os.path.join(self.projects_directory, project_id)
        os.makedirs(project_dir, exist_ok=True)

        # Create subdirectories
        os.makedirs(os.path.join(project_dir, "documents"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "conversations"), exist_ok=True)

        # Save project config
        try:
            with open(os.path.join(project_dir, "config.json"), "w") as f:
                json.dump(project_data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save project config for {project_id}: {e}")
            # Decide how to handle - raise error? Return None?
            raise  # Re-raise for now

        # Add to in-memory store
        self.projects[project_id] = project_data

        logger.info(f"Created new project: {name} (ID: {project_id})")
        return project_data

    def get_project(self, project_id: str) -> Optional[dict]:
        """
        Get project data by ID.

        Args:
            project_id: The project ID

        Returns:
            Project data dictionary or None if not found
        """
        return self.projects.get(project_id)

    def list_projects(self) -> List[dict]:
        """
        List all projects.

        Returns:
            List of project data dictionaries
        """
        # Maybe sort by name or date?
        return sorted(list(self.projects.values()), key=lambda p: p.get("name", ""))

    def add_conversation_to_project(
        self, project_id: str, conversation_id: str
    ) -> bool:
        """
        Associate a conversation with a project (primarily for metadata).
        The actual association happens when ChatService creates the chat with a project_id.

        Args:
            project_id: The project ID
            conversation_id: The conversation ID

        Returns:
            True if project exists, False otherwise (association is handled by ChatService)
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project not found: {project_id}")
            return False

        # Optional: Update project metadata if needed (e.g., last activity)
        project["updated_at"] = datetime.now().isoformat()
        try:
            project_dir = os.path.join(self.projects_directory, project_id)
            with open(os.path.join(project_dir, "config.json"), "w") as f:
                json.dump(project, f, indent=2)
        except IOError as e:
            logger.warning(
                f"Failed to update project config timestamp for {project_id}: {e}"
            )

        logger.info(
            f"Ensured conversation {conversation_id} is associated with project {project_id} (via ChatService). Updated project timestamp."
        )
        return True

    def get_project_conversations(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Get all conversation summaries associated with a project using ChatService.

        Args:
            project_id: The project ID

        Returns:
            List of conversation summary dictionaries from ChatService
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project not found: {project_id}")
            return []

        try:
            # Use ChatService to get the list of chats for the project
            # list_chats_for_project returns List[Dict[str, Any]] containing metadata
            chats = self.chat_service.list_chats_for_project(project_id)
            logger.info(
                f"Retrieved {len(chats)} chats for project {project_id} via ChatService."
            )
            return chats
        except Exception as e:
            logger.error(
                f"Error retrieving chats for project {project_id} from ChatService: {e}"
            )
            return []

    def add_document_to_project(
        self, project_id: str, document_id: str, metadata: dict = None
    ) -> bool:
        """
        Add a document to a project.

        Args:
            project_id: The project ID
            document_id: The document ID
            metadata: Additional document metadata

        Returns:
            True if successful, False otherwise
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project not found: {project_id}")
            return False

        # Create document entry
        doc_entry = {
            "id": document_id,
            "added_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Update project data
        doc_ids = [doc["id"] for doc in project["documents"]]
        if document_id not in doc_ids:
            project["documents"].append(doc_entry)
            project["updated_at"] = datetime.now().isoformat()

            # Save project config
            project_dir = os.path.join(self.projects_directory, project_id)
            with open(os.path.join(project_dir, "config.json"), "w") as f:
                json.dump(project, f, indent=2)

        logger.info(f"Added document {document_id} to project {project_id}")
        return True

    def get_project_summary(self, project_id: str) -> Optional[dict]:
        """
        Get a summary of project data including conversations and documents.

        Args:
            project_id: The project ID

        Returns:
            Project summary dictionary or None if not found
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project not found: {project_id}")
            return None

        # Get conversation summaries from ChatService
        conversation_summaries = self.get_project_conversations(project_id)
        # get_project_conversations already returns List[Dict], which is suitable

        # Combine project data with conversation summaries
        summary = project.copy()
        summary["conversations"] = (
            conversation_summaries  # Replace list of IDs with list of summaries
        )
        # Document list is already part of the project data

        return summary

    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project and its associated data.

        Args:
            project_id: The project ID

        Returns:
            True if successful, False otherwise
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project not found for deletion: {project_id}")
            return False

        # TODO: Implement deletion of associated conversations via ChatService
        # Need a way to list conversations for a project and delete them
        try:
            conversation_summaries = self.get_project_conversations(project_id)
            for chat_summary in conversation_summaries:
                chat_id = chat_summary.get("chat_id")
                if chat_id:
                    # Call ChatService to delete the chat
                    # Assuming ChatService has an async delete_chat method
                    # This might need to be run in an event loop if called from sync code
                    import asyncio

                    try:
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(self.chat_service.delete_chat(chat_id))
                        logger.info(
                            f"Deleted associated chat {chat_id} for project {project_id}"
                        )
                    except RuntimeError:  # No event loop running
                        asyncio.run(self.chat_service.delete_chat(chat_id))
                        logger.info(
                            f"Deleted associated chat {chat_id} for project {project_id} (new loop)"
                        )
                    except Exception as chat_del_e:
                        logger.warning(
                            f"Failed to delete chat {chat_id} for project {project_id}: {chat_del_e}"
                        )
        except Exception as e:
            logger.error(
                f"Error listing conversations for deletion for project {project_id}: {e}"
            )

        # TODO: Implement deletion of associated documents from DocumentService?
        # This is complex - documents might be shared or global.
        # Maybe just remove the project association from document metadata?
        # Or just delete the project config/folder and leave documents in vector store?

        # Delete project directory
        try:
            project_dir = os.path.join(self.projects_directory, project_id)
            if os.path.exists(project_dir):
                import shutil

                shutil.rmtree(project_dir)
                logger.info(f"Deleted project directory: {project_dir}")
        except Exception as e:
            logger.error(f"Error deleting project directory {project_dir}: {str(e)}")
            # Continue deletion even if directory removal fails?

        # Remove from in-memory store
        if project_id in self.projects:
            del self.projects[project_id]

        logger.info(f"Deleted project: {project_id}")
        return True
