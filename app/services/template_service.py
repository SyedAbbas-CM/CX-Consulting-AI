import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# from app.template_wrappers.prompt_template import PromptTemplateManager # Reuse or adapt
# from app.services.file_converters import markdown_to_pptx, markdown_to_docx # Placeholder for future
import jinja2  # Assuming Jinja2 is used

logger = logging.getLogger(__name__)

# Assuming templates are stored relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATES_DIR = PROJECT_ROOT / "data" / "templates"


class TemplateService:
    """
    Manages loading, processing, and potentially converting deliverable templates.
    Focuses on Jinja2 for Markdown initially.
    """

    def __init__(self, templates_dir: str = str(DEFAULT_TEMPLATES_DIR)):
        """
        Initializes the TemplateService.

        Args:
            templates_dir: Directory where deliverable templates (.md, .pptx, .docx) are stored.
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=jinja2.select_autoescape(),  # Basic autoescaping
        )
        logger.info(
            f"TemplateService initialized. Loading templates from: {self.templates_dir}"
        )

    def list_available_templates(
        self, file_extensions: List[str] = [".md", ".pptx", ".docx"]
    ) -> Dict[str, Path]:
        """
        Lists available template files in the templates directory.

        Args:
            file_extensions: List of file extensions to look for.

        Returns:
            A dictionary mapping template names (filenames without extension) to their file paths.
        """
        available = {}
        for ext in file_extensions:
            for template_path in self.templates_dir.glob(f"*{ext}"):
                template_name = template_path.stem  # Name without extension
                available[template_name] = template_path
        logger.debug(f"Found available templates: {list(available.keys())}")
        return available

    def load_markdown_template(self, template_name: str) -> Optional[jinja2.Template]:
        """
        Loads a Jinja2 template from a .md file.

        Args:
            template_name: The name of the template (without .md extension).

        Returns:
            A compiled Jinja2 template object, or None if not found.
        """
        template_filename = f"{template_name}.md"
        try:
            template = self.jinja_env.get_template(template_filename)
            logger.info(f"Loaded Markdown template: {template_filename}")
            return template
        except jinja2.exceptions.TemplateNotFound:
            logger.error(
                f"Markdown template not found: {template_filename} in {self.templates_dir}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error loading template {template_filename}: {e}", exc_info=True
            )
            return None

    def render_markdown_template(
        self, template_name: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Renders a loaded Markdown template with the given context.

        Args:
            template_name: The name of the template (without .md extension).
            context: A dictionary containing variables for the template.

        Returns:
            The rendered Markdown content as a string, or None on error.
        """
        template = self.load_markdown_template(template_name)
        if not template:
            return None

        try:
            rendered_content = template.render(context)
            logger.info(f"Rendered markdown template: {template_name}.md")
            return rendered_content
        except Exception as e:
            logger.error(
                f"Error rendering template {template_name}.md: {e}", exc_info=True
            )
            return None

    # --- File Conversion (Implementation Pending) ---


# End of TemplateService class

# Removed Dependency Injection Helper function get_template_service
# Dependency injection is handled in app/api/dependencies.py
