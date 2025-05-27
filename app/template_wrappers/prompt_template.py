import json
import logging
import os
import re
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Template class for structured prompting."""

    def __init__(self, template_string: str, template_name: str = None):
        """
        Initialize a prompt template.

        Args:
            template_string: The template string with placeholders
            template_name: Optional name for the template
        """
        self.template = template_string
        self.name = template_name

    def format(self, **kwargs) -> str:
        """
        Format the template with the given arguments, stripping comment lines.

        Args:
            **kwargs: Key-value pairs for template placeholders

        Returns:
            The formatted template
        """
        # Add current date if not provided
        if "current_date" not in kwargs:
            kwargs["current_date"] = datetime.now().strftime("%Y-%m-%d")

        # --- FIX: Remove comment lines ---
        # 1. remove multi-line Jinja comments {# ... #}
        # Make sure to handle potential None from self.template
        tpl_str = self.template if self.template is not None else ""
        # clean_template = re.sub(r"\\{#.*?#\\}", "", tpl_str, flags=re.DOTALL) # TEMP COMMENT OUT

        # 2. drop single-line python-style comments starting with `#`
        # clean_template = "\\n".join(
        #     line for line in clean_template.splitlines() if not line.strip().startswith("#")
        # )
        # FIX for comment stripper regex (Checklist Item 4-A)
        # clean_template = "\\n".join(
        #     line for line in clean_template.splitlines()
        #     if not re.match(r"^\\s*#.*", line)
        # ) # TEMP COMMENT OUT
        clean_template = tpl_str  # TEMP: Use tpl_str directly, bypassing cleaning

        try:
            return clean_template.format(**kwargs)
        except KeyError as e:
            logger.error(
                f"KeyError formatting template '{self.name or 'Unnamed'}': Missing key {e}"
            )
            logger.error(f"Available keys: {list(kwargs.keys())}")
            logger.error(f"Cleaned template snippet: {clean_template[:500]}...")
            # Re-raise the error after logging details
            raise e
        except Exception as format_err:
            logger.error(
                f"Unexpected error formatting template '{self.name or 'Unnamed'}': {format_err}",
                exc_info=True,
            )
            raise format_err

    @classmethod
    def from_file(cls, file_path: str) -> "PromptTemplate":
        """
        Load a template from a file.

        Args:
            file_path: Path to the template file

        Returns:
            A prompt template instance
        """
        with open(file_path, "r", encoding="utf-8") as f:
            template_string = f.read()

        # Use the filename as the template name
        template_name = Path(file_path).stem

        return cls(template_string, template_name)

    @classmethod
    def from_json(cls, json_path: str, key: str = "template") -> "PromptTemplate":
        """
        Load a template from a JSON file.

        Args:
            json_path: Path to the JSON file
            key: The key in the JSON that contains the template string

        Returns:
            A prompt template instance
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        template_string = data[key]
        template_name = data.get("name", Path(json_path).stem)

        return cls(template_string, template_name)


class PromptTemplateManager:
    """Manager for handling multiple prompt templates."""

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the template manager.

        Args:
            templates_dir: Directory containing template files.
                           If None, defaults to "app/data/templates".
        """
        self.templates: Dict[str, PromptTemplate] = {}
        self.supported_ext = {".txt", ".md", ".jinja"}

        effective_templates_dir_str: str
        if templates_dir is not None:
            effective_templates_dir_str = templates_dir
        else:
            # Default path based on observed logs and common practice.
            # Consider making this configurable via settings in a future revision.
            effective_templates_dir_str = "app/data/templates"
            logger.info(
                f"templates_dir not provided to PromptTemplateManager, "
                f"defaulting to '{effective_templates_dir_str}'"
            )

        # Ensure self.template_dir is always a Path object
        self.template_dir: Path = Path(effective_templates_dir_str).resolve()

        # Load base templates
        self._load_base_templates()

        # Load templates from the effective directory if it exists
        if self.template_dir.exists():
            self.load_templates_from_directory(str(self.template_dir))
        else:
            logger.warning(
                f"Template directory '{self.template_dir}' does not exist. "
                "No custom templates will be loaded from this path."
            )

    def _load_base_templates(self):
        """Load built-in base templates."""
        # Define base system prompt
        system_prompt = """You are a powerful AI assistant specializing in Customer Experience (CX) consulting, powered by Google Gemma. You operate within the CX Consulting AI application.

You are assisting a USER with CX-related tasks. These tasks may involve analyzing client information, generating consulting deliverables, understanding CX concepts, or retrieving information from the knowledge base.

<context_usage>
Each time the USER sends a message, you will use the following sources of information to generate your response:
1.  **User Query:** The specific question or instruction from the USER.
2.  **Knowledge Base Context:** Relevant snippets retrieved from uploaded documents (e.g., case studies, best practice guides, project documents). This context will be provided with source identifiers.
3.  **Conversation History:** The ongoing dialogue with the USER, providing short-term memory.
4.  **Project Context (if applicable):** Details about the specific client project the USER is working on.
</context_usage>

<capabilities>
Your main capabilities include:
- Answering questions about CX concepts, methodologies, and best practices.
- Retrieving and synthesizing information from the knowledge base relevant to the user's query.
- Generating CX deliverables such as Proposals, ROI Analyses, and Customer Journey Maps based on user input and knowledge base context.
- Maintaining context within a conversation.
- Citing sources for information retrieved from the knowledge base.
</capabilities>

<persona>
- Maintain a professional, knowledgeable, and helpful tone suitable for a consultant.
- Be proactive in providing relevant information but concise in your answers.
- If the answer requires information not present in the provided context or conversation history, clearly state that the information is unavailable rather than speculating.
- When generating deliverables, follow the specified structures and formats.
</persona>

<citations>
When using information directly from the provided knowledge base context, you MUST cite the source ID. The context will contain `<source id=DOCUMENT_ID>` tags. Cite the relevant `DOCUMENT_ID` after the information derived from it. For example: "Customer satisfaction increased by 10% after implementing the changes (Source: doc-abc-123)." If multiple sources support a statement, cite them all. Use information from the knowledge base context as the primary source of truth whenever available.
</citations>

Current date: {current_date}
"""
        self.add_template("system", system_prompt)

        # Define RAG prompt - updated to work with Gemma's chat template
        rag_prompt = """Role: You are a CX (Customer Experience) consultant assistant.

Task: Answer the user's query. Use the provided Context Information as your primary source. Also, refer to the Previous Conversation for short-term memory and to understand follow-up questions or references to earlier parts of the dialogue.

Instructions:
1.  Analyze the User Query.
2.  Carefully review the Context Information provided below (if any).
3.  Review the Previous Conversation provided below (if any).
4.  Synthesize the relevant information from the Context AND the Previous Conversation to construct your answer.
5.  If citing from Context Information, use the format (Source: DOC_ID) as specified in your system instructions.
6.  If the context and conversation history do not contain the answer, state that clearly.
7.  Maintain a professional and helpful tone.

Context Information (Knowledge Base):
---------------------
{context}
---------------------

Previous Conversation:
---------------------
{conversation_history}
---------------------

User Query:
{query}

Answer:
"""
        self.add_template("rag", rag_prompt)

    def add_template(self, name: str, template_string: str) -> None:
        """
        Add a template to the manager.

        Args:
            name: Name of the template
            template_string: Template string
        """
        self.templates[name] = PromptTemplate(template_string, name)

    def get_template(self, name: str) -> PromptTemplate:
        """
        Get a template by name.

        Args:
            name: Name of the template

        Returns:
            The prompt template
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")

        return self.templates[name]

    def get_raw_template(self, name: str) -> str:
        """
        Get the raw (unrendered) template string by name.

        Args:
            name: Name of the template (usually filename stem)

        Returns:
            The raw template string.
        """
        # This is a simplified approach; a more robust way might involve
        # storing original file paths or ensuring names directly map to discoverable files.

        # First, check if a template object exists with this name to validate.
        if name not in self.templates:
            # Attempt to find and load it if it's a known deliverable pattern but not yet loaded
            # This could happen if AgentRunner asks for a deliverable template not pre-loaded
            # (e.g., if templates are discovered dynamically rather than all at startup)
            # For now, we'll assume it must be in self.templates if it's valid.
            # A more robust solution might try to load it here.
            potential_path = self.template_dir / f"{name}.txt"  # or .j2, etc.
            if potential_path.exists():
                logger.info(
                    f"Dynamically loading template '{name}' from {potential_path}"
                )
                try:
                    template_content = potential_path.read_text()
                    self.templates[name] = self.jinja_env.from_string(template_content)
                except Exception as e:
                    logger.error(f"Failed to dynamically load template '{name}': {e}")
                    raise ValueError(
                        f"Template '{name}' not found or failed to load."
                    ) from e
            else:
                raise ValueError(
                    f"Raw template '{name}' not found and could not be dynamically loaded."
                )

        # Assuming the template name is the stem and it's a .txt file
        # This is brittle; consider storing full paths or a mapping.
        template_file_path = self.template_dir / f"{name}.txt"
        if not template_file_path.exists():
            # Try with .j2 extension if .txt not found
            template_file_path = self.template_dir / f"{name}.j2"
            if not template_file_path.exists():
                raise ValueError(
                    f"Raw template file for '{name}' not found at expected paths: {self.template_dir / f'{name}.txt'} or {self.template_dir / f'{name}.j2'}"
                )

        try:
            return template_file_path.read_text()
        except Exception as e:
            logger.error(
                f"Error reading raw template '{name}' from {template_file_path}: {e}"
            )
            raise

    # ------------------------------------------------------------------
    # Public helper – render a template with either jinja2 or langchain
    # ------------------------------------------------------------------
    def render(self, template_name: str, **kwargs) -> str:
        """
        Renders *any* registered template.
        • If it's a jinja2.Template   → use .render(**kwargs)
        • If it's a langchain Prompt  → use .format(**kwargs)
        """
        tmpl = self.get_template(template_name)

        if hasattr(tmpl, "render"):  # jinja2
            return tmpl.render(**kwargs)

        if hasattr(tmpl, "format"):  # langchain PromptTemplate
            # This is our local PromptTemplate, not Langchain's directly
            # It already handles kwargs correctly in its format method
            return tmpl.format(**kwargs)

        raise TypeError(
            f"Unknown template type: {type(tmpl).__name__} – "
            f"expected a local PromptTemplate or a Jinja2 template, but got {type(tmpl)}"
        )

    def get_available_templates(self) -> List[str]:
        """
        Get a list of available templates.

        Returns:
            A list of template names
        """
        return list(self.templates.keys())

    def load_templates_from_directory(self, directory: str) -> None:
        """
        Load all templates from a directory.

        Args:
            directory: Directory containing template files
        """
        logger.info(f"Loading custom templates from directory: {directory}")
        found_count = 0
        # Load templates with supported extensions
        for ext in self.supported_ext:
            for file_path in Path(directory).glob(f"*{ext}"):
                # FIX: Forbid .jinja files for now (Checklist Item 4-B)
                if file_path.suffix == ".jinja":
                    logger.warning(
                        f"Skipping loading of Jinja template '{file_path.name}' as Jinja rendering is not currently supported."
                    )
                    continue

                try:
                    template = PromptTemplate.from_file(str(file_path))
                    self.templates[template.name] = template
                    logger.debug(
                        f"Loaded template '{template.name}' from {file_path.name}"
                    )
                    found_count += 1
                except Exception as e:
                    logger.error(
                        f"Error loading template from {file_path}: {e}", exc_info=True
                    )

        # Original code for JSON (kept for compatibility if needed, but O3 fix focuses on text)
        for file_path in Path(directory).glob("*.json"):
            try:
                template = PromptTemplate.from_json(str(file_path))
                self.templates[template.name] = template
            except (json.JSONDecodeError, KeyError):
                print(f"Error loading template from {file_path}")

    def save_template(self, name: str, directory: str) -> None:
        """
        Save a template to a file.

        Args:
            name: Name of the template
            directory: Directory to save the template
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")

        template = self.templates[name]

        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Save template to file
        file_path = os.path.join(directory, f"{name}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(template.template)
