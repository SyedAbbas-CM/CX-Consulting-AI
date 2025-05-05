import os
import json
import uuid
from datetime import datetime
from typing import List, Optional

class ProjectHandler:
    """
    Handle project-related operations including creation, retrieval, and management
    of project-specific data and conversations.
    """
    
    def __init__(self, memory_manager=None, document_service=None):
        """
        Initialize the project handler.
        
        Args:
            memory_manager: Optional memory manager for conversation tracking
            document_service: Optional document service for project-related documents
        """
        self.memory_manager = memory_manager
        self.document_service = document_service
        self.projects = {}  # In-memory store of projects
        
        # Initialize from environment if available
        self.projects_directory = os.getenv("PROJECTS_DIRECTORY", "./projects")
        
        # Create projects directory if it doesn't exist
        os.makedirs(self.projects_directory, exist_ok=True)
        
        # Load existing projects
        self._load_projects()
        
        logger.info("ProjectHandler initialized")
    
    def _load_projects(self):
        """Load existing projects from the projects directory."""
        try:
            for project_dir in os.listdir(self.projects_directory):
                project_path = os.path.join(self.projects_directory, project_dir)
                
                # Skip non-directories
                if not os.path.isdir(project_path):
                    continue
                
                # Check for project config file
                config_path = os.path.join(project_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        project_data = json.load(f)
                        self.projects[project_data["id"]] = project_data
            
            logger.info(f"Loaded {len(self.projects)} existing projects")
        except Exception as e:
            logger.error(f"Error loading projects: {str(e)}")
    
    def create_project(self, name: str, description: str = "", metadata: dict = None) -> dict:
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
            "documents": []
        }
        
        # Create project directory
        project_dir = os.path.join(self.projects_directory, project_id)
        os.makedirs(project_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(project_dir, "documents"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "conversations"), exist_ok=True)
        
        # Save project config
        with open(os.path.join(project_dir, "config.json"), 'w') as f:
            json.dump(project_data, f, indent=2)
        
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
        return list(self.projects.values())
    
    def add_conversation_to_project(self, project_id: str, conversation_id: str) -> bool:
        """
        Add a conversation to a project.
        
        Args:
            project_id: The project ID
            conversation_id: The conversation ID
            
        Returns:
            True if successful, False otherwise
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project not found: {project_id}")
            return False
        
        # Update project data
        if conversation_id not in project["conversations"]:
            project["conversations"].append(conversation_id)
            project["updated_at"] = datetime.now().isoformat()
            
            # Save project config
            project_dir = os.path.join(self.projects_directory, project_id)
            with open(os.path.join(project_dir, "config.json"), 'w') as f:
                json.dump(project, f, indent=2)
        
        # Update memory manager if available
        if self.memory_manager:
            self.memory_manager.set_conversation_project(conversation_id, project_id)
        
        logger.info(f"Added conversation {conversation_id} to project {project_id}")
        return True
    
    def get_project_conversations(self, project_id: str) -> List[str]:
        """
        Get all conversation IDs associated with a project.
        
        Args:
            project_id: The project ID
            
        Returns:
            List of conversation IDs
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project not found: {project_id}")
            return []
        
        # If we have a memory manager, use it to get conversations
        if self.memory_manager:
            memory_conversations = self.memory_manager.get_project_conversations(project_id)
            # Merge with project data
            all_conversations = list(set(project["conversations"] + memory_conversations))
            return all_conversations
        
        return project["conversations"]
    
    def add_document_to_project(self, project_id: str, document_id: str, metadata: dict = None) -> bool:
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
            "metadata": metadata or {}
        }
        
        # Update project data
        doc_ids = [doc["id"] for doc in project["documents"]]
        if document_id not in doc_ids:
            project["documents"].append(doc_entry)
            project["updated_at"] = datetime.now().isoformat()
            
            # Save project config
            project_dir = os.path.join(self.projects_directory, project_id)
            with open(os.path.join(project_dir, "config.json"), 'w') as f:
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
        
        # Get conversation data if memory manager is available
        conversation_data = []
        if self.memory_manager:
            conversation_ids = self.get_project_conversations(project_id)
            for conv_id in conversation_ids:
                messages = self.memory_manager.get_conversation(conv_id)
                if messages:
                    first_message = next((msg for msg in messages if msg["role"] == "user"), None)
                    conversation_data.append({
                        "id": conv_id,
                        "message_count": len(messages),
                        "first_message": first_message["content"] if first_message else "",
                        "last_updated": messages[-1]["timestamp"] if "timestamp" in messages[-1] else ""
                    })
        
        # Get document data if document service is available
        document_data = []
        if self.document_service:
            for doc in project["documents"]:
                doc_info = self.document_service.get_document_info(doc["id"])
                if doc_info:
                    document_data.append({
                        "id": doc["id"],
                        "filename": doc_info.get("filename", ""),
                        "added_at": doc["added_at"],
                        "metadata": doc["metadata"]
                    })
        
        summary = {
            "id": project["id"],
            "name": project["name"],
            "description": project["description"],
            "created_at": project["created_at"],
            "updated_at": project["updated_at"],
            "conversation_count": len(conversation_data),
            "conversations": conversation_data,
            "document_count": len(document_data),
            "documents": document_data
        }
        
        return summary 