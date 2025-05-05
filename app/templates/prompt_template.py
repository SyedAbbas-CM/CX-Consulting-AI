from typing import Dict, Any, List, Optional
import string
from datetime import datetime
import os
import json
from pathlib import Path

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
        Format the template with the given arguments.
        
        Args:
            **kwargs: Key-value pairs for template placeholders
            
        Returns:
            The formatted template
        """
        # Add current date if not provided
        if 'current_date' not in kwargs:
            kwargs['current_date'] = datetime.now().strftime("%Y-%m-%d")
            
        return self.template.format(**kwargs)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'PromptTemplate':
        """
        Load a template from a file.
        
        Args:
            file_path: Path to the template file
            
        Returns:
            A prompt template instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            template_string = f.read()
        
        # Use the filename as the template name
        template_name = Path(file_path).stem
        
        return cls(template_string, template_name)
    
    @classmethod
    def from_json(cls, json_path: str, key: str = 'template') -> 'PromptTemplate':
        """
        Load a template from a JSON file.
        
        Args:
            json_path: Path to the JSON file
            key: The key in the JSON that contains the template string
            
        Returns:
            A prompt template instance
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        template_string = data[key]
        template_name = data.get('name', Path(json_path).stem)
        
        return cls(template_string, template_name)


class PromptTemplateManager:
    """Manager for handling multiple prompt templates."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Load base templates
        self._load_base_templates()
        
        # Load templates from directory if provided
        if templates_dir:
            self.load_templates_from_directory(templates_dir)
    
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

Task: Answer the user's query based *primarily* on the provided Context Information. Use the Previous Conversation for short-term memory only if relevant.

Instructions:
1.  Analyze the User Query.
2.  Carefully review the Context Information provided below.
3.  Synthesize the relevant information from the Context to construct your answer.
4.  Cite sources for information taken from the Context using the format (Source: DOC_ID) as specified in your system instructions.
5.  If the context does not contain the answer, state that clearly.
6.  Maintain a professional and helpful tone.

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
        
        # Define proposal template - updated to work with Gemma's chat template
        proposal_prompt = """Task: Create a professional CX consulting proposal using Markdown formatting.

Instructions:
1.  Analyze the provided Client Information and Project Requirements to understand the client's needs.
2.  Use the Context Information (retrieved from our knowledge base) to inform the technical approach, methodology, relevant experience, and team details. Cite sources using (Source: DOC_ID) when referencing specific context.
3.  Generate a comprehensive proposal with the following sections, clearly marked using Markdown headings:
    *   ## 1. Executive Summary
        (Provide a brief overview of the client's challenge and the proposed solution's key benefits.)
    *   ## 2. Client Situation & Challenges
        (Detail the client's current situation, pain points, and objectives based on the Client Information and Requirements.)
    *   ## 3. Recommended Approach & Methodology
        (Describe the proposed solution, steps, and methodology. Draw heavily from relevant Context Information.)
    *   ## 4. Proposed Project Plan
        (Outline key phases, activities, and a high-level timeline.)
    *   ## 5. Expected Outcomes & Deliverables
        (List the specific results and tangible outputs the client will receive.)
    *   ## 6. Team & Expertise
        (Highlight relevant team experience and skills, referencing Context Information for case studies or qualifications.)
    *   ## 7. Investment & Timeline
        (Provide estimated pricing/budget and a more detailed timeline if possible.)
4.  Maintain a professional and persuasive tone.

Inputs:

Client Information:
{client_info}

Project Requirements:
{requirements}

Context Information (Knowledge Base):
---------------------
{context}
---------------------

Proposal Output (Use Markdown):
"""
        self.add_template("proposal", proposal_prompt)
        
        # Define ROI analysis template - updated to work with Gemma's chat template
        roi_prompt = """Task: Create a comprehensive business case analysis using Markdown formatting.

Instructions:
1.  Analyze the provided Client Information and Project Details to understand the current situation, costs, proposed solution, and potential benefits.
2.  Use the Context Information (retrieved from our knowledge base, e.g., benchmarks, case studies) to quantify costs, benefits, and support calculations where possible. Cite sources using (Source: DOC_ID) when referencing specific context.
3.  Generate a detailed business case with the following sections, clearly marked using Markdown headings:
    *   ## 1. Executive Summary
        (Briefly summarize the problem, solution, key financial metrics like ROI/Payback, and recommendation.)
    *   ## 2. Current State Assessment
        (Describe the client's current situation and processes related to the project.)
    *   ## 3. Cost of Current Issues
        (Quantify the negative impact or cost of the current problems, using data from Project Details or Context.)
    *   ## 4. Proposed Solution
        (Clearly describe the solution being proposed.)
    *   ## 5. Implementation Costs
        (Estimate the costs associated with implementing the solution - software, hardware, training, personnel. Use Context for benchmarks if needed.)
    *   ## 6. Expected Benefits
        (Quantify the expected positive outcomes - cost savings, revenue increase, efficiency gains. Clearly state assumptions and reference Context.)
    *   ## 7. ROI Calculation
        (Calculate the Return on Investment over a specific period, showing the formula: ROI = (Net Benefits / Total Costs) * 100%. Net Benefits = Total Benefits - Total Costs.)
    *   ## 8. Payback Period
        (Calculate the time it takes for the accumulated benefits to equal the initial investment.)
    *   ## 9. Sensitivity Analysis
        (Briefly discuss how changes in key assumptions might affect the outcome.)
    *   ## 10. Recommendations
        (Provide a clear recommendation based on the analysis.)
4.  Maintain a professional, analytical, and data-driven tone.

Inputs:

Client Information:
{client_info}

Project Details:
{project_details}

Context Information (Knowledge Base):
---------------------
{context}
---------------------

ROI Analysis Output (Use Markdown):
"""
        self.add_template("roi", roi_prompt)
    
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
    
    def load_templates_from_directory(self, directory: str) -> None:
        """
        Load all templates from a directory.
        
        Args:
            directory: Directory containing template files
        """
        # Load text templates
        for file_path in Path(directory).glob("*.txt"):
            template = PromptTemplate.from_file(str(file_path))
            self.templates[template.name] = template
        
        # Load JSON templates
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
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(template.template) 