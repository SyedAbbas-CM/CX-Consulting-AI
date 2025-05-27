import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

from app.services.document_service import DocumentService

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger("cx_consulting_ai.project_manager")


class ProjectManager:
    """Manager for projects and project documents."""

    def __init__(
        self,
        storage_type: Optional[str] = None,
        storage_path: Optional[Union[str, Path]] = None,
        document_service: Optional[DocumentService] = None,
    ):
        """
        Initialize the project manager.

        Args:
            storage_type: Type of storage (file or redis)
            storage_path: Directory to store project files
            document_service: Instance of DocumentService for re-indexing
        """
        self.storage_type = storage_type or os.getenv("PROJECT_STORAGE_TYPE", "file")
        self.document_service = document_service

        # Use storage_path, default to env var or "app/data/projects"
        _raw_path = storage_path or os.getenv("PROJECT_DIR", "app/data/projects")
        self.project_dir_path: Path = Path(_raw_path)

        # Create project directory if it doesn't exist
        if self.storage_type == "file":
            self.project_dir_path.mkdir(
                parents=True, exist_ok=True
            )  # Use Path object's mkdir

            self.projects_index_file_path = (
                self.project_dir_path / "projects_index.json"
            )
            self._load_projects_index()  # Call the new load/rebuild method

        elif self.storage_type == "redis":
            # Initialize Redis connection
            try:
                import redis

                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self.redis = redis.from_url(redis_url)
                logger.info(f"Connected to Redis at {redis_url}")
            except ImportError:
                logger.warning(
                    "Redis package not installed, falling back to file storage"
                )
                self.storage_type = "file"
                self.project_dir_path.mkdir(parents=True, exist_ok=True)

                # Initialize projects index
                self.projects_index_file_path = (
                    self.project_dir_path / "projects_index.json"
                )
                if not self.projects_index_file_path.exists():
                    with open(self.projects_index_file_path, "w") as f:
                        json.dump({"projects": {}}, f)

                # Load projects index
                with open(self.projects_index_file_path, "r") as f:
                    self.projects_index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                logger.warning("Falling back to file storage")
                self.storage_type = "file"
                self.project_dir_path.mkdir(parents=True, exist_ok=True)

                # Initialize projects index
                self.projects_index_file_path = (
                    self.project_dir_path / "projects_index.json"
                )
                if not self.projects_index_file_path.exists():
                    with open(self.projects_index_file_path, "w") as f:
                        json.dump({"projects": {}}, f)

                # Load projects index
                with open(self.projects_index_file_path, "r") as f:
                    self.projects_index = json.load(f)

        logger.info(
            f"Project manager initialized with storage_type={self.storage_type}"
        )

    def _load_projects_index(self):
        """Load the projects index from file, or rebuild it if necessary."""
        if self.storage_type != "file":
            self.projects_index = {
                "projects": {}
            }  # Should not happen if logic is correct
            return

        rebuilt_index = False
        if self.projects_index_file_path.exists():
            try:
                with open(self.projects_index_file_path, "r") as f:
                    self.projects_index = json.load(f)
                if not self.projects_index.get("projects"):
                    logger.warning(
                        "Projects index is empty or malformed. Attempting rebuild."
                    )
                    self.projects_index = {"projects": {}}  # Reset before rebuild
                    rebuilt_index = self._rebuild_projects_index()
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to decode projects index file {self.projects_index_file_path}. Attempting rebuild.",
                    exc_info=True,
                )
                self.projects_index = {"projects": {}}
                rebuilt_index = self._rebuild_projects_index()
        else:
            logger.info(
                f"Projects index file {self.projects_index_file_path} not found. Attempting rebuild."
            )
            self.projects_index = {"projects": {}}
            rebuilt_index = self._rebuild_projects_index()

        if not rebuilt_index and not self.projects_index.get("projects"):
            # If not rebuilt and still empty, ensure it's initialized correctly
            logger.info(
                "Initializing empty projects index file as no projects found or rebuilt."
            )
            self.projects_index = {"projects": {}}
            self._save_projects_index()  # Save the empty structure

    def _rebuild_projects_index(self) -> bool:
        """Rebuild the projects index by scanning individual project files."""
        logger.info(
            f"Attempting to rebuild projects index from directory: {self.project_dir_path}"
        )
        found_projects_count = 0
        rebuilt_projects = {}
        for item in self.project_dir_path.iterdir():
            if (
                item.is_file()
                and item.name.endswith(".json")
                and item.name != "projects_index.json"
            ):
                try:
                    project_id = item.stem  # Get filename without .json
                    with open(item, "r") as f:
                        project_data = json.load(f)

                    # Validate essential fields for the index
                    if all(
                        k in project_data
                        for k in ["id", "name", "client_name", "owner_id", "updated_at"]
                    ):
                        if project_data["id"] == project_id:
                            rebuilt_projects[project_id] = {
                                "id": project_data["id"],
                                "name": project_data["name"],
                                "client_name": project_data["client_name"],
                                "owner_id": project_data["owner_id"],
                                "updated_at": project_data["updated_at"],
                            }
                            found_projects_count += 1
                        else:
                            logger.warning(
                                f"Project ID mismatch in file {item.name}: expected {project_id}, got {project_data.get('id')}"
                            )
                    else:
                        logger.warning(
                            f"Skipping project file {item.name} during rebuild due to missing essential keys for index."
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to decode project file {item.name} during rebuild.",
                        exc_info=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing project file {item.name} during rebuild: {e}",
                        exc_info=True,
                    )

        if found_projects_count > 0:
            self.projects_index["projects"] = rebuilt_projects
            logger.info(
                f"Successfully rebuilt projects index with {found_projects_count} projects."
            )
            self._save_projects_index()
            return True
        else:
            logger.info("No valid project files found to rebuild index.")
            return False

    def _save_projects_index(self):
        """Atomically save the current projects index to file."""
        if self.storage_type != "file":
            return

        try:
            # Create a temporary file in the same directory to ensure `os.replace` is atomic
            with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                dir=self.project_dir_path,
                prefix="projects_index_",
                suffix=".json.tmp",
            ) as tmp_file:
                json.dump(self.projects_index, tmp_file, indent=2)
                tmp_file_path = tmp_file.name  # Get the path of the temporary file

            # Atomically replace the old index file with the new one
            os.replace(tmp_file_path, self.projects_index_file_path)
            logger.debug(
                f"Atomically saved projects index to {self.projects_index_file_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save projects index: {e}", exc_info=True)
            # If temp file path was set and still exists, try to clean it up
            if "tmp_file_path" in locals() and Path(tmp_file_path).exists():
                try:
                    os.unlink(tmp_file_path)
                except Exception as unlink_e:
                    logger.error(
                        f"Failed to cleanup temporary index file {tmp_file_path}: {unlink_e}"
                    )

    def create_project(
        self,
        name: str,
        client_name: str,
        industry: str,
        description: str,
        owner_id: str,
        shared_with: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
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
            "metadata": metadata or {},
        }

        if self.storage_type == "file":
            # Save project to file
            project_file_path = self.project_dir_path / f"{project_id}.json"
            with open(project_file_path, "w") as f:
                json.dump(project, f, indent=2)

            # Update projects index
            self.projects_index["projects"][project_id] = {
                "id": project_id,
                "name": name,
                "client_name": client_name,
                "owner_id": owner_id,
                "updated_at": now,
            }

            # Save projects index
            self._save_projects_index()

        elif self.storage_type == "redis":
            # Save project to Redis
            self.redis.set(f"project:{project_id}", json.dumps(project))

            # Add to project index
            self.redis.hset(
                "projects:index",
                project_id,
                json.dumps(
                    {
                        "id": project_id,
                        "name": name,
                        "client_name": client_name,
                        "owner_id": owner_id,
                        "updated_at": now,
                    }
                ),
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
            project_file_path = self.project_dir_path / f"{project_id}.json"
            if not project_file_path.exists():
                logger.warning(f"Project {project_id} not found")
                return None

            with open(project_file_path, "r") as f:
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
            project_file_path = self.project_dir_path / f"{project_id}.json"
            with open(project_file_path, "w") as f:
                json.dump(project, f, indent=2)

            # Update projects index
            self.projects_index["projects"][project_id]["name"] = project["name"]
            self.projects_index["projects"][project_id]["client_name"] = project[
                "client_name"
            ]
            self.projects_index["projects"][project_id]["updated_at"] = project[
                "updated_at"
            ]

            # Save projects index
            self._save_projects_index()

        elif self.storage_type == "redis":
            # Save project to Redis
            self.redis.set(f"project:{project_id}", json.dumps(project))

            # Update project index
            self.redis.hset(
                "projects:index",
                project_id,
                json.dumps(
                    {
                        "id": project_id,
                        "name": project["name"],
                        "client_name": project["client_name"],
                        "updated_at": project["updated_at"],
                    }
                ),
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
            project_file_path = self.project_dir_path / f"{project_id}.json"
            if not project_file_path.exists():
                logger.warning(f"Project {project_id} not found for deletion")
                return False

            # Delete project file
            project_file_path.unlink()

            # Delete any document files
            doc_dir = self.project_dir_path / project_id
            if doc_dir.exists():
                for file in doc_dir.glob("*"):
                    file.unlink()
                doc_dir.rmdir()

            # Update projects index
            if project_id in self.projects_index["projects"]:
                del self.projects_index["projects"][project_id]

                # Save projects index
                self._save_projects_index()

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

    def list_projects(
        self, limit: int = 50, offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
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
                key=lambda pid: self.projects_index["projects"][pid].get(
                    "updated_at", ""
                ),
                reverse=True,
            )

            # Apply pagination
            paginated_ids = project_ids[offset : offset + limit]

            # Load full project data
            projects = []
            for pid in paginated_ids:
                project_file_path = self.project_dir_path / f"{pid}.json"
                if project_file_path.exists():
                    with open(project_file_path, "r") as f:
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
            paginated_info = projects_info[offset : offset + limit]

            # Load full project data
            projects = []
            for info in paginated_info:
                project_json = self.redis.get(f"project:{info['id']}")
                if project_json:
                    projects.append(json.loads(project_json))

            return projects, len(projects_info)

    def add_conversation_to_project(
        self, project_id: str, conversation_id: str
    ) -> bool:
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
            return self.update_project(
                project_id, {"conversation_ids": project["conversation_ids"]}
            )

        return True

    def remove_conversation_from_project(
        self, project_id: str, conversation_id: str
    ) -> bool:
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
            return self.update_project(
                project_id, {"conversation_ids": project["conversation_ids"]}
            )

        return True

    def create_document(
        self,
        project_id: str,
        title: str,
        content: str,
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> str:
        """
        Create a new document within a project.
        If document_id is provided, it will be used; otherwise, a new one is generated.
        """
        project = self.get_project(project_id)
        if not project:
            logger.error(f"Project {project_id} not found, cannot create document")
            raise ValueError(f"Project {project_id} not found")

        # Use provided document_id or generate a new one
        doc_id = document_id if document_id else str(uuid.uuid4())

        now = datetime.now().isoformat()

        # Create document object
        doc = {
            "id": doc_id,
            "project_id": project_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "title": title,
            "content": content,
            "document_type": document_type,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
        }

        # Add chunk_count and token_count (can be updated later if needed)
        doc["metadata"]["chunk_count"] = 0
        doc["metadata"]["token_count"] = 0

        if self.storage_type == "file":
            # Save document to file
            project_docs_dir = self.project_dir_path / project_id
            project_docs_dir.mkdir(parents=True, exist_ok=True)
            doc_file_path = project_docs_dir / f"{doc_id}.json"
            with open(doc_file_path, "w") as f:
                json.dump(doc, f, indent=2)

            # Update project's document list
            project_doc_ids = project.get("document_ids", [])
            if doc_id not in project_doc_ids:
                project_doc_ids.append(doc_id)
                self.update_project(project_id, {"document_ids": project_doc_ids})
            else:
                # This case should ideally not happen if project_id is validated before calling
                logger.error(
                    f"Document {doc_id} already exists in project {project_id}"
                )

        elif self.storage_type == "redis":
            # Save document to Redis
            self.redis.set(f"document:{doc_id}", json.dumps(doc))

            # Update project's document list
            project_doc_ids = project.get("document_ids", [])
            if doc_id not in project_doc_ids:
                project_doc_ids.append(doc_id)
                self.update_project(project_id, {"document_ids": project_doc_ids})
            else:
                logger.error(
                    f"Document {doc_id} already exists in project {project_id} in Redis"
                )

        logger.info(
            f"Created document {doc_id} in project {project_id} (user: {user_id})"
        )
        return doc_id

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
                doc_file_path = (
                    self.project_dir_path / project_id / f"{document_id}.json"
                )
                if doc_file_path.exists():
                    with open(doc_file_path, "r") as f:
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
            doc_dir = self.project_dir_path / project_id
            if not doc_dir.exists():
                return []

            for doc_file_path in doc_dir.glob("*.json"):
                documents.append(json.load(doc_file_path.open()))

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
            doc_file_path = (
                self.project_dir_path / document["project_id"] / f"{document_id}.json"
            )
            with open(doc_file_path, "w") as f:
                json.dump(document, f, indent=2)

        elif self.storage_type == "redis":
            # Save document to Redis
            self.redis.set(f"document:{document_id}", json.dumps(document))

        logger.info(f"Updated document {document_id}")

        # Re-indexing logic has been removed from ProjectManager.update_document.
        # The calling service should explicitly trigger re-indexing if content has changed.
        # logger.warning(f"ProjectManager.update_document: Re-indexing from here is currently disabled/problematic due to async/sync mismatch. Re-index explicitly if needed.")

        return True  # Return True as the document itself was updated

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
            doc_file_path = self.project_dir_path / project_id / f"{document_id}.json"
            if doc_file_path.exists():
                doc_file_path.unlink()

            # Update project
            project = self.get_project(project_id)
            if project and document_id in project["document_ids"]:
                project["document_ids"].remove(document_id)
                self.update_project(
                    project_id, {"document_ids": project["document_ids"]}
                )

        elif self.storage_type == "redis":
            # Delete document
            self.redis.delete(f"document:{document_id}")

            # Update project
            project = self.get_project(project_id)
            if project and document_id in project["document_ids"]:
                project["document_ids"].remove(document_id)
                self.update_project(
                    project_id, {"document_ids": project["document_ids"]}
                )

        logger.info(f"Deleted document {document_id}")
        return True

    def get_user_projects(
        self, user_id: str, include_shared: bool = True
    ) -> List[Dict[str, Any]]:
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
                        elif include_shared and user_id in project.get(
                            "shared_with", []
                        ):
                            projects.append(project)
                except Exception as e:
                    logger.error(f"Error getting project {project_id}: {str(e)}")

        elif self.storage_type == "redis":
            # Get all project IDs
            project_ids = self.redis.hkeys("projects:index")

            for project_id in project_ids:
                try:
                    project_id = (
                        project_id.decode("utf-8")
                        if isinstance(project_id, bytes)
                        else project_id
                    )
                    project = self.get_project(project_id)
                    if project:
                        # Add project if user is owner
                        if project.get("owner_id") == user_id:
                            projects.append(project)
                        # Add project if shared with user and include_shared is True
                        elif include_shared and user_id in project.get(
                            "shared_with", []
                        ):
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

    def save_revision(
        self,
        document_id: str,
        revision_id: str,
        new_content: str,
        author: str,  # e.g., "model" or user_id
        refinement_prompt: Optional[str] = None,  # Optional: log the prompt used
    ) -> Optional[Dict[str, Any]]:
        """
        Save a new revision of a document.
        Updates content, logs revision metadata, and triggers re-indexing.

        Args:
            document_id: The ID of the document to update.
            revision_id: A unique ID for this revision.
            new_content: The new content for the document.
            author: Identifier for the author of this revision.
            refinement_prompt: The prompt used for this refinement, if applicable.

        Returns:
            The updated document dictionary if successful, None otherwise.
        """
        document = self.get_document(document_id)
        if not document:
            logger.warning(
                f"Document {document_id} not found for saving revision {revision_id}"
            )
            return None

        now_iso = datetime.now().isoformat()

        # Update main content
        document["content"] = new_content
        document["updated_at"] = now_iso

        # Initialize or append to revisions list in metadata
        if "revisions" not in document["metadata"]:
            document["metadata"]["revisions"] = []

        revision_entry: Dict[str, Any] = {
            "revision_id": revision_id,
            "author": author,
            "timestamp": now_iso,
            "content_hash": str(
                uuid.uuid5(uuid.NAMESPACE_DNS, new_content)
            ),  # Simple hash for content tracking
        }
        if refinement_prompt:
            revision_entry["refinement_prompt"] = refinement_prompt

        document["metadata"]["revisions"].append(revision_entry)
        # Optionally, keep only a certain number of recent revisions to prevent metadata bloat
        # max_revisions = 10
        # document["metadata"]["revisions"] = document["metadata"]["revisions"][-max_revisions:]

        if self.storage_type == "file":
            project_docs_dir = self.project_dir_path / document["project_id"]
            # Ensure directory exists (should normally, but defensive check)
            project_docs_dir.mkdir(parents=True, exist_ok=True)
            doc_file_path = project_docs_dir / f"{document_id}.json"
            try:
                with open(doc_file_path, "w") as f:
                    json.dump(document, f, indent=2)
            except IOError as e:
                logger.error(
                    f"Failed to write revision for document {document_id} to file {doc_file_path}: {e}"
                )
                return None

        elif self.storage_type == "redis":
            try:
                self.redis.set(f"document:{document_id}", json.dumps(document))
            except Exception as e:
                logger.error(
                    f"Failed to save revision for document {document_id} to Redis: {e}"
                )
                return None

        logger.info(
            f"Saved revision {revision_id} by {author} for document {document_id}"
        )

        # Re-indexing logic has been removed from ProjectManager.update_document.
        # The calling service should explicitly trigger re-indexing if content has changed.
        # logger.warning(f"ProjectManager.update_document: Re-indexing from here is currently disabled/problematic due to async/sync mismatch. Re-index explicitly if needed.")

        return document

    def get_document_content_by_revision(
        self,
        document_id: str,
        revision_id: str,
    ) -> Optional[str]:
        """
        Get the content of a document at a specific revision.

        Args:
            document_id: The ID of the document.
            revision_id: The ID of the revision.

        Returns:
            The content of the document at the specified revision, or None if not found.
        """
        document = self.get_document(document_id)
        if not document:
            logger.warning(f"Document {document_id} not found")
            return None

        # Find the revision with the specified revision_id
        for revision in document.get("revisions", []):
            if revision["revision_id"] == revision_id:
                return revision["content"]

        logger.warning(f"Revision {revision_id} not found in document {document_id}")
        return None

    async def get_latest_document_summary(
        self, project_id: str, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves a summary of the most recently created document for a given project.
        Optionally filters by user_id if provided.
        The summary includes title, id, creation timestamp, and metadata like page_count,
        chunk_count, and extracted_title if available.
        """
        logger.debug(
            f"Attempting to get latest document summary for project '{project_id}', user '{user_id}'."
        )
        project_data = await self._load_project_data(project_id)
        if (
            not project_data
            or "documents" not in project_data
            or not project_data["documents"]
        ):
            logger.warning(
                f"No documents found or project data is invalid for project {project_id}."
            )
            return None

        documents = project_data["documents"]

        # Filter by user_id if provided
        if user_id:
            user_documents = [
                doc
                for doc in documents
                if doc.get("metadata", {}).get("uploader_user_id") == user_id
                or doc.get("user_id") == user_id
            ]
            if not user_documents:
                logger.info(
                    f"No documents found for user '{user_id}' in project '{project_id}'."
                )
                # Fallback: if user has no docs, maybe show latest project doc? Or return None?
                # For now, returning None if user_id is specified and no docs match.
                return None
            documents_to_search = user_documents
        else:
            documents_to_search = documents

        if (
            not documents_to_search
        ):  # Should not happen if user_id was None and project had docs, but as a safeguard
            logger.warning(
                f"No documents available to search for latest in project {project_id} after filtering."
            )
            return None

        # Sort documents by 'created_at' timestamp in descending order.
        # Handle cases where 'created_at' might be missing or not a float.
        def get_sort_key(doc):
            created_at = doc.get("created_at")
            if isinstance(created_at, (int, float)):
                return float(created_at)
            # Attempt to parse if it's a string representation of a float/int
            if isinstance(created_at, str):
                try:
                    return float(created_at)
                except ValueError:
                    pass  # Not a float string
            return -1  # Default to a low value for docs without a valid timestamp

        sorted_documents = sorted(documents_to_search, key=get_sort_key, reverse=True)

        if not sorted_documents:
            logger.info(
                f"No sortable documents found for project '{project_id}' (user: '{user_id}')."
            )
            return None

        latest_doc = sorted_documents[0]

        # Construct a summary. Ensure all keys are safely accessed.
        summary = {
            "id": latest_doc.get("id"),
            "title": latest_doc.get("title", "Untitled Document"),
            "created_at": latest_doc.get("created_at"),  # This is the timestamp
            "metadata": latest_doc.get("metadata", {}),
            # page_count and chunk_count might be in top-level metadata or nested further
            # The route handler will try to extract these robustly.
        }

        # Log the summary being returned
        logger.info(
            f"Latest document summary for project '{project_id}' (user: '{user_id}'): {summary.get('title')} (ID: {summary.get('id')})"
        )
        return summary
