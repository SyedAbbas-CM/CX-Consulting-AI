import asyncio
import json
import logging
import os
import tarfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.llm_service import LLMService  # Import LLMService
from app.template_wrappers.prompt_template import (  # Import PromptTemplateManager
    PromptTemplateManager,
)

logger = logging.getLogger(__name__)

# Define the base directory for projects relative to the app or project root
# Assuming project root contains 'data' folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECTS_BASE_DIR = PROJECT_ROOT / "data" / "projects"
PROJECTS_BASE_DIR = Path(os.getenv("PROJECTS_DATA_DIR", DEFAULT_PROJECTS_BASE_DIR))


class ProjectMemoryService:
    """
    Manages project-specific long-term memory, including:
    - Narrative summaries (memory.md) using LLM for condensation.
    - Branch snapshots (.tar.gz)
    - (Potentially) Raw chat logs (chat_log.jsonl) - though ChatService handles this mostly
    - (Potentially) Generated deliverables
    """

    def __init__(
        self,
        llm_service: LLMService,
        prompt_manager: PromptTemplateManager,
        base_dir: str = str(PROJECTS_BASE_DIR),
    ):
        """
        Initializes the ProjectMemoryService.

        Args:
            llm_service: An instance of LLMService for summary condensation.
            prompt_manager: An instance of PromptTemplateManager to load prompts.
            base_dir: The root directory where all project data is stored.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.llm_service = llm_service
        self.prompt_manager = prompt_manager
        self.project_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        logger.info(
            f"ProjectMemoryService initialized with LLMService and PromptManager. "
            f"Base directory: {self.base_dir}"
        )

    def _get_project_dir(self, project_id: str) -> Path:
        """Returns the specific directory for a given project ID."""
        return self.base_dir / project_id

    def _ensure_project_dirs(self, project_id: str) -> None:
        """Creates the necessary subdirectories for a project if they don't exist."""
        project_dir = self._get_project_dir(project_id)
        (project_dir / "deliverables").mkdir(parents=True, exist_ok=True)
        (project_dir / "branch-snapshots").mkdir(parents=True, exist_ok=True)

    async def get_narrative_summary(self, project_id: str) -> str:
        """
        Reads the narrative summary (memory.md) for a project.
        """
        project_dir = self._get_project_dir(project_id)
        summary_file = project_dir / "memory.md"

        if not summary_file.exists():
            logger.warning(
                f"Narrative summary file not found for project {project_id}. Returning empty."
            )
            return ""

        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary_content = f.read()
            logger.debug(f"Read narrative summary for project {project_id}")
            return summary_content
        except IOError as e:
            logger.error(
                f"Error reading summary file for project {project_id}: {e}",
                exc_info=True,
            )
            return ""

    async def update_narrative_summary(
        self, project_id: str, section_name: str, new_section_content: str
    ) -> bool:
        """
        Updates the narrative summary (memory.md) by condensing new section content
        with the existing summary using an LLM.

        Args:
            project_id: The ID of the project.
            section_name: The name of the recently completed section.
            new_section_content: The full content of the recently completed deliverable section.

        Returns:
            True if successful, False otherwise.
        """
        self._ensure_project_dirs(project_id)
        project_dir = self._get_project_dir(project_id)
        summary_file = project_dir / "memory.md"

        existing_summary = await self.get_narrative_summary(project_id)

        try:
            prompt_template = self.prompt_manager.get_template(
                "condense_narrative_prompt"
            )
            prompt_context = {
                "existing_summary": existing_summary,
                "section_name": section_name,
                "new_section_content": new_section_content,
            }
            prompt = prompt_template.render(prompt_context)

            logger.info(
                f"Condensing narrative summary for project {project_id} after section '{section_name}'"
            )
            # Consider adding max_tokens for summary generation, e.g., 500-1000 tokens
            condensed_summary = await self.llm_service.generate(prompt, max_tokens=1024)

            if condensed_summary is None:
                logger.warning(
                    f"LLM returned None for narrative summary condensation for project {project_id}. Previous summary preserved if exists."
                )
                return False

            await asyncio.to_thread(
                summary_file.write_text, condensed_summary.strip(), encoding="utf-8"
            )
            logger.info(
                f"Successfully updated narrative summary for project {project_id} using LLM condensation."
            )
            return True

        except Exception as e:
            logger.error(
                f"Error updating narrative summary for project {project_id}: {e}",
                exc_info=True,
            )
            return False

    async def append_chat_interaction(
        self, project_id: str, user_query: str, llm_response: str
    ) -> bool:
        """
        Appends a user query and LLM response to the project's memory.md file
        under a '## Chat History' section.

        Args:
            project_id: The ID of the project.
            user_query: The user's query.
            llm_response: The LLM's response.

        Returns:
            True if successful, False otherwise.
        """
        project_lock = self.project_locks[project_id]
        async with project_lock:
            self._ensure_project_dirs(project_id)
            project_dir = self._get_project_dir(project_id)
            summary_file = project_dir / "memory.md"

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            interaction_entry = (
                f"\n### Interaction at {timestamp}\n"  # Use H3 for individual interactions
                f"**User:** {user_query}\n"
                f"**AI:** {llm_response}\n"
            )

            try:
                content = ""
                if summary_file.exists():
                    content = await asyncio.to_thread(
                        summary_file.read_text, encoding="utf-8"
                    )

                chat_history_heading = "## Chat History"
                if chat_history_heading not in content:
                    content += f"\n\n{chat_history_heading}\n"

                content += interaction_entry

                await asyncio.to_thread(
                    summary_file.write_text, content, encoding="utf-8"
                )

                logger.info(
                    f"Appended chat interaction to memory.md for project {project_id}"
                )
                return True
            except IOError as e:
                logger.error(
                    f"Error appending chat interaction to memory.md for project {project_id}: {e}",
                    exc_info=True,
                )
                return False
            except Exception as e:
                logger.error(
                    f"Unexpected error appending chat interaction for project {project_id}: {e}",
                    exc_info=True,
                )
                return False

    async def create_branch_snapshot(
        self, project_id: str, branch_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Creates a snapshot (.tar.gz) of the current project state (memory, deliverables).
        """
        project_dir = self._get_project_dir(project_id)
        if not project_dir.exists():
            logger.error(f"Project directory not found for snapshot: {project_id}")
            return None

        self._ensure_project_dirs(project_id)
        snapshot_dir = project_dir / "branch-snapshots"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"snapshot_{branch_name or 'main'}_{timestamp}.tar.gz"
        snapshot_filepath = snapshot_dir / snapshot_filename

        logger.info(
            f"Creating snapshot for project {project_id} at {snapshot_filepath}"
        )

        try:
            with tarfile.open(snapshot_filepath, "w:gz") as tar:
                memory_file = project_dir / "memory.md"
                if memory_file.exists():
                    tar.add(memory_file, arcname="memory.md")

                deliverables_dir = project_dir / "deliverables"
                if deliverables_dir.exists() and any(deliverables_dir.iterdir()):
                    tar.add(deliverables_dir, arcname="deliverables")

            logger.info(f"Snapshot created successfully: {snapshot_filepath}")
            return str(snapshot_filepath)

        except Exception as e:
            logger.error(
                f"Error creating snapshot for project {project_id}: {e}", exc_info=True
            )
            if snapshot_filepath.exists():
                try:
                    os.remove(snapshot_filepath)
                except OSError:
                    pass
            return None

    # Note: Restoring from snapshot is more complex as it involves potentially:
    # 1. Creating a *new* project ID (usually handled by API/ProjectManager).
    # 2. Extracting the snapshot into the *new* project directory.
    # 3. Handling vector store aliasing/copying based on snapshot contents.
    # This service primarily handles *creating* the snapshot file.
    # The API route /projects/{id}/fork would orchestrate:
    # - Calling ProjectManager to create a new project record.
    # - Calling this service's create_branch_snapshot on the *source* project.
    # - Copying/extracting the snapshot into the *new* project's directory.


# Dependency Injection Helper
# Needs to be updated in dependencies.py to inject LLMService and PromptTemplateManager
def get_project_memory_service(
    llm_service: LLMService,  # Add LLMService dependency
    prompt_manager: PromptTemplateManager,  # Add PromptTemplateManager dependency
):
    return ProjectMemoryService(llm_service=llm_service, prompt_manager=prompt_manager)
