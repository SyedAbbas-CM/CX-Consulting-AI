from typing import Dict, List, Tuple, Optional, Any
import os
import time
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger("cx_consulting_ai.project_manager")

class ProjectManager:
    """Manager for projects and project documents."""
    
    def __init__(self, storage_type: str = None, project_dir: str = None):
        """
        Initialize the project manager.
        
        Args:
            storage_type: Type of storage (file or redis)
            project_dir: Directory to store project files
        """
        self.storage_type = storage_type or os.getenv("PROJECT_STORAGE_TYPE", "file")
        self.project_dir = project_dir or os.getenv("PROJECT_DIR", "app/data/projects")
        
        # Create project directory if it doesn't exist
        if self.storage_type == "file":
            os.makedirs(self.project_dir, exist_ok=True)
            
            # Initialize projects index if it doesn't exist
            self.projects_index_path = os.path.join(self.project_dir, "projects_index.json")
            if not os.path.exists(self.projects_index_path):
                with open(self.projects_index_path, "w") as f:
                    json.dump({"projects": {}}, f)
            
            # Load projects index
            with open(self.projects_index_path, "r") as f:
                self.projects_index = json.load(f)
        
        elif self.storage_type == "redis":
            # Initialize Redis connection
            try:
                import redis
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self.redis = redis.from_url(redis_url)
                logger.info(f"Connected to Redis at {redis_url}")
            except ImportError:
                logger.warning("Redis package not installed, falling back to file storage")
                self.storage_type = "file"
                os.makedirs(self.project_dir, exist_ok=True)
                
                # Initialize projects index
                self.projects_index_path = os.path.join(self.project_dir, "projects_index.json")
                if not os.path.exists(self.projects_index_path):
                    with open(self.projects_index_path, "w") as f:
                        json.dump({"projects": {}}, f)
                
                # Load projects index
                with open(self.projects_index_path, "r") as f:
                    self.projects_index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                logger.warning("Falling back to file storage")
                self.storage_type = "file"
                os.makedirs(self.project_dir, exist_ok=True)
                
                # Initialize projects index
                self.projects_index_path = os.path.join(self.project_dir, "projects_index.json")
                if not os.path.exists(self.projects_index_path):
                    with open(self.projects_index_path, "w") as f:
                        json.dump({"projects": {}}, f)
                
                # Load projects index
                with open(self.projects_index_path, "r") as f:
                    self.projects_index = json.load(f)
        
        logger.info(f"Project manager initialized with storage_type={self.storage_type}")
    
    def create_project(
        self, 
        name: str,
        client_name: str,
        industry: str,
        description: str,
        owner_id: str,
        shared_with: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new project.
        
        Args:
            name: Project name
            client_name: Client name
            industry: Client industry
            description: Project description
            owner_id: User ID of the project owner
            shared_with: Optional list of user IDs to share the project with
            metadata: Optional metadata
            
        Returns:
            Project ID
        """
        project_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        project = {
            "id": project_id,
            "name": name,
            "client_name": client_name,
            "industry": industry,
            "description": description,
            "created_at": now,
            "updated_at": now,
            "owner_id": owner_id,
            "shared_with": shared_with or [],
            "conversation_ids": [],
            "document_ids": [],
            "metadata": metadata or {}
        }
        
        if self.storage_type == "file":
            # Save project to file
            project_path = os.path.join(self.project_dir, f"{project_id}.json")
            with open(project_path, "w") as f:
                json.dump(project, f, indent=2)
            
            # Update projects index
            self.projects_index["projects"][project_id] = {
                "id": project_id,
                "name": name,
                "client_name": client_name,
                "owner_id": owner_id,
                "updated_at": now
            }
            
            # Save projects index
            with open(self.projects_index_path, "w") as f:
                json.dump(self.projects_index, f, indent=2)
        
        elif self.storage_type == "redis":
            # Save project to Redis
            self.redis.set(f"project:{project_id}", json.dumps(project))
            
            # Add to project index
            self.redis.hset(
                "projects:index",
                project_id,
                json.dumps({
                    "id": project_id,
                    "name": name,
                    "client_name": client_name,
                    "owner_id": owner_id,
                    "updated_at": now
                })
            )
        
        logger.info(f"Created project {project_id}: {name} (owner: {owner_id})")
        return project_id
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a project by ID.
        
        Args:
            project_id: Project ID
            
        Returns:
            Project data or None if not found
        """
        if self.storage_type == "file":
            project_path = os.path.join(self.project_dir, f"{project_id}.json")
            if not os.path.exists(project_path):
                logger.warning(f"Project {project_id} not found")
                return None
            
            with open(project_path, "r") as f:
                return json.load(f)
        
        elif self.storage_type == "redis":
            project_json = self.redis.get(f"project:{project_id}")
            if not project_json:
                logger.warning(f"Project {project_id} not found")
                return None
            
            return json.loads(project_json)
    
    def update_project(self, project_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a project.
        
        Args:
            project_id: Project ID
            updates: Fields to update
            
        Returns:
            True if successful, False otherwise
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project {project_id} not found for update")
            return False
        
        # Update fields
        for key, value in updates.items():
            if key in project and key not in ["id", "created_at"]:
                project[key] = value
        
        # Update timestamp
        project["updated_at"] = datetime.now().isoformat()
        
        if self.storage_type == "file":
            # Save project to file
            project_path = os.path.join(self.project_dir, f"{project_id}.json")
            with open(project_path, "w") as f:
                json.dump(project, f, indent=2)
            
            # Update projects index
            self.projects_index["projects"][project_id]["name"] = project["name"]
            self.projects_index["projects"][project_id]["client_name"] = project["client_name"]
            self.projects_index["projects"][project_id]["updated_at"] = project["updated_at"]
            
            # Save projects index
            with open(self.projects_index_path, "w") as f:
                json.dump(self.projects_index, f, indent=2)
        
        elif self.storage_type == "redis":
            # Save project to Redis
            self.redis.set(f"project:{project_id}", json.dumps(project))
            
            # Update project index
            self.redis.hset(
                "projects:index",
                project_id,
                json.dumps({
                    "id": project_id,
                    "name": project["name"],
                    "client_name": project["client_name"],
                    "updated_at": project["updated_at"]
                })
            )
        
        logger.info(f"Updated project {project_id}")
        return True
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            True if successful, False otherwise
        """
        if self.storage_type == "file":
            project_path = os.path.join(self.project_dir, f"{project_id}.json")
            if not os.path.exists(project_path):
                logger.warning(f"Project {project_id} not found for deletion")
                return False
            
            # Delete project file
            os.remove(project_path)
            
            # Delete any document files
            doc_dir = os.path.join(self.project_dir, project_id)
            if os.path.exists(doc_dir):
                for file in os.listdir(doc_dir):
                    os.remove(os.path.join(doc_dir, file))
                os.rmdir(doc_dir)
            
            # Update projects index
            if project_id in self.projects_index["projects"]:
                del self.projects_index["projects"][project_id]
                
                # Save projects index
                with open(self.projects_index_path, "w") as f:
                    json.dump(self.projects_index, f, indent=2)
        
        elif self.storage_type == "redis":
            # Check if project exists
            if not self.redis.exists(f"project:{project_id}"):
                logger.warning(f"Project {project_id} not found for deletion")
                return False
            
            # Get project data to find document IDs
            project_json = self.redis.get(f"project:{project_id}")
            project = json.loads(project_json)
            
            # Delete all project documents
            for doc_id in project.get("document_ids", []):
                self.redis.delete(f"document:{doc_id}")
            
            # Delete project
            self.redis.delete(f"project:{project_id}")
            
            # Remove from project index
            self.redis.hdel("projects:index", project_id)
        
        logger.info(f"Deleted project {project_id}")
        return True
    
    def list_projects(self, limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """
        List all projects.
        
        Args:
            limit: Maximum number of projects to return
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of projects, total count)
        """
        if self.storage_type == "file":
            # Get project IDs from index
            project_ids = list(self.projects_index["projects"].keys())
            
            # Sort by updated_at (most recent first)
            project_ids.sort(
                key=lambda pid: self.projects_index["projects"][pid].get("updated_at", ""),
                reverse=True
            )
            
            # Apply pagination
            paginated_ids = project_ids[offset:offset+limit]
            
            # Load full project data
            projects = []
            for pid in paginated_ids:
                project_path = os.path.join(self.project_dir, f"{pid}.json")
                if os.path.exists(project_path):
                    with open(project_path, "r") as f:
                        projects.append(json.load(f))
            
            return projects, len(project_ids)
        
        elif self.storage_type == "redis":
            # Get all projects from index
            project_index = self.redis.hgetall("projects:index")
            
            # Convert from bytes
            projects_info = [json.loads(info) for info in project_index.values()]
            
            # Sort by updated_at (most recent first)
            projects_info.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
            
            # Apply pagination
            paginated_info = projects_info[offset:offset+limit]
            
            # Load full project data
            projects = []
            for info in paginated_info:
                project_json = self.redis.get(f"project:{info['id']}")
                if project_json:
                    projects.append(json.loads(project_json))
            
            return projects, len(projects_info)
    
    def add_conversation_to_project(self, project_id: str, conversation_id: str) -> bool:
        """
        Add a conversation to a project.
        
        Args:
            project_id: Project ID
            conversation_id: Conversation ID
            
        Returns:
            True if successful, False otherwise
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project {project_id} not found")
            return False
        
        # Add conversation ID if not already in the list
        if conversation_id not in project["conversation_ids"]:
            project["conversation_ids"].append(conversation_id)
            
            # Update project
            return self.update_project(project_id, {"conversation_ids": project["conversation_ids"]})
        
        return True
    
    def remove_conversation_from_project(self, project_id: str, conversation_id: str) -> bool:
        """
        Remove a conversation from a project.
        
        Args:
            project_id: Project ID
            conversation_id: Conversation ID
            
        Returns:
            True if successful, False otherwise
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project {project_id} not found")
            return False
        
        # Remove conversation ID if in the list
        if conversation_id in project["conversation_ids"]:
            project["conversation_ids"].remove(conversation_id)
            
            # Update project
            return self.update_project(project_id, {"conversation_ids": project["conversation_ids"]})
        
        return True
    
    def create_document(
        self,
        project_id: str,
        title: str,
        content: str,
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new project document.
        
        Args:
            project_id: Project ID
            title: Document title
            content: Document content
            document_type: Type of document
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project {project_id} not found")
            raise ValueError(f"Project {project_id} not found")
        
        document_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        document = {
            "id": document_id,
            "project_id": project_id,
            "title": title,
            "content": content,
            "document_type": document_type,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {}
        }
        
        if self.storage_type == "file":
            # Create document directory if it doesn't exist
            doc_dir = os.path.join(self.project_dir, project_id)
            os.makedirs(doc_dir, exist_ok=True)
            
            # Save document to file
            doc_path = os.path.join(doc_dir, f"{document_id}.json")
            with open(doc_path, "w") as f:
                json.dump(document, f, indent=2)
            
            # Update project with document ID
            project["document_ids"].append(document_id)
            self.update_project(project_id, {"document_ids": project["document_ids"]})
        
        elif self.storage_type == "redis":
            # Save document to Redis
            self.redis.set(f"document:{document_id}", json.dumps(document))
            
            # Update project with document ID
            project["document_ids"].append(document_id)
            self.update_project(project_id, {"document_ids": project["document_ids"]})
        
        logger.info(f"Created document {document_id} for project {project_id}")
        return document_id
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        if self.storage_type == "file":
            # Search in all project directories
            for project_id in self.projects_index["projects"]:
                doc_path = os.path.join(self.project_dir, project_id, f"{document_id}.json")
                if os.path.exists(doc_path):
                    with open(doc_path, "r") as f:
                        return json.load(f)
            
            logger.warning(f"Document {document_id} not found")
            return None
        
        elif self.storage_type == "redis":
            document_json = self.redis.get(f"document:{document_id}")
            if not document_json:
                logger.warning(f"Document {document_id} not found")
                return None
            
            return json.loads(document_json)
    
    def get_project_documents(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Get all documents for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of documents
        """
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project {project_id} not found")
            return []
        
        documents = []
        
        if self.storage_type == "file":
            doc_dir = os.path.join(self.project_dir, project_id)
            if not os.path.exists(doc_dir):
                return []
            
            for filename in os.listdir(doc_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(doc_dir, filename), "r") as f:
                        documents.append(json.load(f))
        
        elif self.storage_type == "redis":
            for doc_id in project["document_ids"]:
                document_json = self.redis.get(f"document:{doc_id}")
                if document_json:
                    documents.append(json.loads(document_json))
        
        # Sort by updated_at (most recent first)
        documents.sort(key=lambda d: d.get("updated_at", ""), reverse=True)
        
        return documents
    
    def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a document.
        
        Args:
            document_id: Document ID
            updates: Fields to update
            
        Returns:
            True if successful, False otherwise
        """
        document = self.get_document(document_id)
        if not document:
            logger.warning(f"Document {document_id} not found")
            return False
        
        # Update fields
        for key, value in updates.items():
            if key in document and key not in ["id", "project_id", "created_at"]:
                document[key] = value
        
        # Update timestamp
        document["updated_at"] = datetime.now().isoformat()
        
        if self.storage_type == "file":
            # Save document to file
            doc_path = os.path.join(self.project_dir, document["project_id"], f"{document_id}.json")
            with open(doc_path, "w") as f:
                json.dump(document, f, indent=2)
        
        elif self.storage_type == "redis":
            # Save document to Redis
            self.redis.set(f"document:{document_id}", json.dumps(document))
        
        logger.info(f"Updated document {document_id}")
        return True
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        document = self.get_document(document_id)
        if not document:
            logger.warning(f"Document {document_id} not found")
            return False
        
        project_id = document["project_id"]
        
        if self.storage_type == "file":
            # Delete document file
            doc_path = os.path.join(self.project_dir, project_id, f"{document_id}.json")
            if os.path.exists(doc_path):
                os.remove(doc_path)
            
            # Update project
            project = self.get_project(project_id)
            if project and document_id in project["document_ids"]:
                project["document_ids"].remove(document_id)
                self.update_project(project_id, {"document_ids": project["document_ids"]})
        
        elif self.storage_type == "redis":
            # Delete document
            self.redis.delete(f"document:{document_id}")
            
            # Update project
            project = self.get_project(project_id)
            if project and document_id in project["document_ids"]:
                project["document_ids"].remove(document_id)
                self.update_project(project_id, {"document_ids": project["document_ids"]})
        
        logger.info(f"Deleted document {document_id}")
        return True
    
    def get_user_projects(self, user_id: str, include_shared: bool = True) -> List[Dict[str, Any]]:
        """
        Get all projects for a user.
        
        Args:
            user_id: User ID
            include_shared: Whether to include projects shared with the user
            
        Returns:
            List of projects
        """
        projects = []
        
        if self.storage_type == "file":
            for project_id in self.projects_index["projects"]:
                try:
                    project = self.get_project(project_id)
                    if project:
                        # Add project if user is owner
                        if project.get("owner_id") == user_id:
                            projects.append(project)
                        # Add project if shared with user and include_shared is True
                        elif include_shared and user_id in project.get("shared_with", []):
                            projects.append(project)
                except Exception as e:
                    logger.error(f"Error getting project {project_id}: {str(e)}")
        
        elif self.storage_type == "redis":
            # Get all project IDs
            project_ids = self.redis.hkeys("projects:index")
            
            for project_id in project_ids:
                try:
                    project_id = project_id.decode("utf-8") if isinstance(project_id, bytes) else project_id
                    project = self.get_project(project_id)
                    if project:
                        # Add project if user is owner
                        if project.get("owner_id") == user_id:
                            projects.append(project)
                        # Add project if shared with user and include_shared is True
                        elif include_shared and user_id in project.get("shared_with", []):
                            projects.append(project)
                except Exception as e:
                    logger.error(f"Error getting project {project_id}: {str(e)}")
        
        # Sort by updated_at (newest first)
        projects.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
        
        return projects
    
    def share_project(self, project_id: str, user_ids: List[str]) -> bool:
        """
        Share a project with other users.
        
        Args:
            project_id: Project ID
            user_ids: List of user IDs to share the project with
            
        Returns:
            True if successful, False otherwise
        """
        # Get project
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project {project_id} not found for sharing")
            return False
        
        # Update shared_with list
        shared_with = set(project.get("shared_with", []))
        shared_with.update(user_ids)
        shared_with = list(shared_with)
        
        # Update project
        update_result = self.update_project(project_id, {"shared_with": shared_with})
        if not update_result:
            logger.error(f"Failed to update project {project_id} for sharing")
            return False
        
        logger.info(f"Shared project {project_id} with users: {', '.join(user_ids)}")
        return True
    
    def unshare_project(self, project_id: str, user_ids: List[str]) -> bool:
        """
        Remove users from a shared project.
        
        Args:
            project_id: Project ID
            user_ids: List of user IDs to remove from the project
            
        Returns:
            True if successful, False otherwise
        """
        # Get project
        project = self.get_project(project_id)
        if not project:
            logger.warning(f"Project {project_id} not found for unsharing")
            return False
        
        # Update shared_with list
        shared_with = set(project.get("shared_with", []))
        shared_with = shared_with - set(user_ids)
        shared_with = list(shared_with)
        
        # Update project
        update_result = self.update_project(project_id, {"shared_with": shared_with})
        if not update_result:
            logger.error(f"Failed to update project {project_id} for unsharing")
            return False
        
        logger.info(f"Removed users from project {project_id}: {', '.join(user_ids)}")
        return True
    
    def can_access_project(self, project_id: str, user_id: str) -> bool:
        """
        Check if a user can access a project.
        
        Args:
            project_id: Project ID
            user_id: User ID
            
        Returns:
            True if the user can access the project, False otherwise
        """
        # Get project
        project = self.get_project(project_id)
        if not project:
            return False
        
        # Check if user is owner
        if project.get("owner_id") == user_id:
            return True
        
        # Check if project is shared with user
        if user_id in project.get("shared_with", []):
            return True
        
        return False 