"""
Deliverable Templates for CX Consulting AI

This file contains template definitions for various CX consulting deliverables.
"""
from typing import Dict, Any, List

class DeliverableTemplates:
    """Manager for CX consulting deliverable templates."""
    
    def __init__(self):
        """Initialize the deliverable templates."""
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all deliverable templates."""
        # Strategic Documents
        self.templates["cx_strategy"] = self._create_cx_strategy_template()
        self.templates["cx_maturity_assessment"] = self._create_cx_maturity_assessment_template()
        self.templates["competitive_analysis"] = self._create_competitive_analysis_template()
        self.templates["voc_program"] = self._create_voc_program_template()
        self.templates["gap_analysis"] = self._create_gap_analysis_template()
        
        # Client Proposals
        self.templates["consulting_proposal"] = self._create_consulting_proposal_template()
        self.templates["project_scope"] = self._create_project_scope_template()
        self.templates["approach_methodology"] = self._create_approach_methodology_template()
        
        # Financial Analyses
        self.templates["roi_model"] = self._create_roi_model_template()
        self.templates["business_case"] = self._create_business_case_template()
        self.templates["value_realization"] = self._create_value_realization_template()
        
        # Experience Design Documents
        self.templates["journey_map"] = self._create_journey_map_template()
        self.templates["service_blueprint"] = self._create_service_blueprint_template()
        self.templates["experience_design"] = self._create_experience_design_template()
        
        # Research Materials
        self.templates["discussion_guide"] = self._create_discussion_guide_template()
        self.templates["research_summary"] = self._create_research_summary_template()
        self.templates["persona"] = self._create_persona_template()
        
        # Implementation Support
        self.templates["project_plan"] = self._create_project_plan_template()
        self.templates["executive_presentation"] = self._create_executive_presentation_template()
        self.templates["change_management"] = self._create_change_management_template()
    
    def get_template(self, template_name: str) -> str:
        """
        Get a specific deliverable template.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            Template string
        """
        return self.templates.get(template_name, "Template not found.")
    
    def get_template_list(self) -> List[str]:
        """
        Get a list of all available templates.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    # Strategic Documents Templates
    
    def _create_cx_strategy_template(self) -> str:
        """Create CX Strategy template."""
        return """
# Customer Experience Strategy

## Executive Summary
{executive_summary}

## Current State Assessment
{current_state}

## Voice of Customer Insights
{voc_insights}

## CX Vision and Principles
{vision_principles}

## Strategic Objectives
{strategic_objectives}

## CX Initiatives Roadmap
{initiatives_roadmap}

## Success Metrics and KPIs
{success_metrics}

## Governance and Operating Model
{governance_model}

## Implementation Timeline
{implementation_timeline}

## Investment and ROI
{investment_roi}

## Conclusion
{conclusion}
"""
    
    def _create_cx_maturity_assessment_template(self) -> str:
        """Create CX Maturity Assessment template."""
        return """
# CX Maturity Assessment

## Assessment Overview
{assessment_overview}

## Methodology
{methodology}

## Current Maturity Rating
{current_maturity}

## Maturity By Dimension

### Strategy & Leadership
{strategy_leadership}

### Customer Understanding
{customer_understanding}

### Design & Delivery
{design_delivery}

### Measurement
{measurement}

### Culture
{culture}

### Technology & Data
{technology_data}

## Maturity Heatmap
{maturity_heatmap}

## Recommendations & Next Steps
{recommendations}

## Implementation Roadmap
{implementation_roadmap}
"""
    
    def _create_competitive_analysis_template(self) -> str:
        """Create Competitive Analysis template."""
        return """
# CX Competitive Analysis Report

## Executive Summary
{executive_summary}

## Methodology
{methodology}

## Competitive Landscape Overview
{landscape_overview}

## Competitor Profiles
{competitor_profiles}

## Experience Benchmarking
{experience_benchmarking}

## Strengths and Weaknesses Analysis
{strengths_weaknesses}

## Customer Sentiment Analysis
{sentiment_analysis}

## Differentiators and Opportunities
{differentiators_opportunities}

## Recommendations
{recommendations}

## Appendix: Data Sources
{data_sources}
"""
    
    def _create_voc_program_template(self) -> str:
        """Create Voice of Customer Program template."""
        return """
# Voice of Customer Program Design

## Program Objectives
{program_objectives}

## Customer Listening Framework
{listening_framework}

## Research Methods
{research_methods}

## Feedback Collection Touchpoints
{feedback_touchpoints}

## Metrics and KPIs
{metrics_kpis}

## Analysis Approach
{analysis_approach}

## Closed-Loop Process
{closed_loop_process}

## Governance and Roles
{governance_roles}

## Technology Requirements
{technology_requirements}

## Implementation Roadmap
{implementation_roadmap}

## Budget and Resources
{budget_resources}
"""
    
    def _create_gap_analysis_template(self) -> str:
        """Create CX Gap Analysis template."""
        return """
# CX Gap Analysis Report

## Executive Summary
{executive_summary}

## Assessment Methodology
{assessment_methodology}

## Current Experience Assessment
{current_experience}

## Desired Experience Vision
{desired_experience}

## Gap Identification
{gap_identification}

## Root Cause Analysis
{root_cause_analysis}

## Impact Assessment
{impact_assessment}

## Recommendations
{recommendations}

## Implementation Plan
{implementation_plan}

## Success Measures
{success_measures}
"""
    
    # Client Proposal Templates
    
    def _create_consulting_proposal_template(self) -> str:
        """Create Consulting Proposal template."""
        return """
# CX Consulting Proposal

## Executive Summary
{executive_summary}

## Client Situation & Challenges
{client_situation}

## Project Objectives
{project_objectives}

## Approach & Methodology
{approach_methodology}

## Project Phases and Activities
{project_phases}

## Deliverables
{deliverables}

## Timeline
{timeline}

## Team & Expertise
{team_expertise}

## Investment
{investment}

## Terms & Conditions
{terms_conditions}

## Next Steps
{next_steps}
"""
    
    def _create_project_scope_template(self) -> str:
        """Create Project Scope template."""
        return """
# Project Scope Document

## Project Overview
{project_overview}

## Objectives
{objectives}

## Scope of Work
{scope_of_work}

## In Scope
{in_scope}

## Out of Scope
{out_of_scope}

## Deliverables
{deliverables}

## Timeline
{timeline}

## Roles & Responsibilities
{roles_responsibilities}

## Dependencies & Assumptions
{dependencies_assumptions}

## Change Management Process
{change_management}

## Approval
{approval}
"""
    
    def _create_approach_methodology_template(self) -> str:
        """Create Approach & Methodology template."""
        return """
# Approach & Methodology

## Project Philosophy
{project_philosophy}

## Methodological Framework
{methodological_framework}

## Phase 1: Discovery & Assessment
{discovery_assessment}

## Phase 2: Design & Development
{design_development}

## Phase 3: Implementation
{implementation}

## Phase 4: Measurement & Refinement
{measurement_refinement}

## Tools & Techniques
{tools_techniques}

## Quality Assurance
{quality_assurance}

## Stakeholder Engagement
{stakeholder_engagement}

## Risk Management
{risk_management}
"""
    
    # Financial Analysis Templates
    
    def _create_roi_model_template(self) -> str:
        """Create ROI Model template."""
        return """
# CX ROI Model

## Executive Summary
{executive_summary}

## Current State Analysis
{current_state}

## Investment Requirements
{investment_requirements}

## One-Time Costs
{one_time_costs}

## Ongoing Costs
{ongoing_costs}

## Revenue Impact
{revenue_impact}

## Cost Savings
{cost_savings}

## Customer Lifetime Value Impact
{cltv_impact}

## ROI Calculation
{roi_calculation}

## Payback Period
{payback_period}

## Sensitivity Analysis
{sensitivity_analysis}

## Implementation Timeline
{implementation_timeline}

## Measurement Plan
{measurement_plan}

## Conclusion
{conclusion}
"""
    
    def _create_business_case_template(self) -> str:
        """Create Business Case template."""
        return """
# CX Business Case

## Executive Summary
{executive_summary}

## Strategic Context
{strategic_context}

## Problem Statement
{problem_statement}

## Current State Assessment
{current_state}

## Proposed Solution
{proposed_solution}

## Benefits Analysis
{benefits_analysis}

## Financial Analysis
{financial_analysis}

## Risk Assessment
{risk_assessment}

## Implementation Approach
{implementation_approach}

## Success Criteria
{success_criteria}

## Governance
{governance}

## Recommendations
{recommendations}
"""
    
    def _create_value_realization_template(self) -> str:
        """Create Value Realization template."""
        return """
# Value Realization Plan

## Executive Summary
{executive_summary}

## Value Drivers
{value_drivers}

## Success Metrics
{success_metrics}

## Measurement Approach
{measurement_approach}

## Baseline Establishment
{baseline_establishment}

## Target Setting
{target_setting}

## Value Tracking Process
{tracking_process}

## Reporting Framework
{reporting_framework}

## Governance Model
{governance_model}

## Value Realization Timeline
{realization_timeline}

## Stakeholder Communication
{stakeholder_communication}

## Continuous Improvement
{continuous_improvement}
"""
    
    # Experience Design Templates
    
    def _create_journey_map_template(self) -> str:
        """Create Journey Map template."""
        return """
# Customer Journey Map

## Persona
{persona}

## Scenario
{scenario}

## Journey Overview
(Provide a brief narrative summary of the customer's journey through this scenario.)

## Journey Stages & Details (Use Markdown Table Format)

| Stage | Goals | Actions | Touchpoints/Channels | Thoughts | Feelings | Pain Points | Opportunities |
|---|---|---|---|---|---|---|---|
| Stage 1 Name | Goal(s) for this stage | Key actions the customer takes | Channels/touchpoints used | What the customer is thinking | Dominant emotions (e.g., frustrated, curious, satisfied) | Specific issues encountered | Ideas for improvement |
| Stage 2 Name | ... | ... | ... | ... | ... | ... | ... |
| ... (Add more stages as needed) ... | ... | ... | ... | ... | ... | ... | ... |

## Supporting Context (From Knowledge Base)
(Briefly summarize any key insights from the provided context that informed the map, citing sources (Source: DOC_ID) if applicable.)
---------------------
{context}
---------------------

## Key Moments of Truth
(Identify critical points in the journey that significantly impact the customer's perception.)

## Metrics & KPIs
(Suggest relevant metrics to measure the success of this journey, e.g., task completion rate, CSAT per stage.)

## Recommendations
(List actionable recommendations based on the identified pain points and opportunities.)
"""
    
    def _create_service_blueprint_template(self) -> str:
        """Create Service Blueprint template."""
        return """
# Service Blueprint

## Service Overview
{service_overview}

## Customer Journey Stages
{journey_stages}

## Customer Actions
{customer_actions}

## Frontstage Actions
{frontstage_actions}

## Backstage Actions
{backstage_actions}

## Support Processes
{support_processes}

## Physical Evidence
{physical_evidence}

## Systems & Technology
{systems_technology}

## Pain Points & Friction
{pain_points}

## Optimization Opportunities
{optimization_opportunities}

## Implementation Considerations
{implementation_considerations}

## Success Metrics
{success_metrics}
"""
    
    def _create_experience_design_template(self) -> str:
        """Create Experience Design Recommendations template."""
        return """
# Experience Design Recommendations

## Executive Summary
{executive_summary}

## Design Principles
{design_principles}

## Target Experience Vision
{experience_vision}

## Current Pain Points
{current_pain_points}

## Design Recommendations
{design_recommendations}

## Experience Prototypes
{experience_prototypes}

## Implementation Considerations
{implementation_considerations}

## Success Metrics
{success_metrics}

## Roadmap
{roadmap}

## Conclusion
{conclusion}
"""
    
    # Research Materials Templates
    
    def _create_discussion_guide_template(self) -> str:
        """Create Discussion Guide template."""
        return """
# Interview Discussion Guide

## Research Objectives
{research_objectives}

## Participant Profile
{participant_profile}

## Introduction Script
{introduction_script}

## Warm-Up Questions
{warmup_questions}

## Core Questions
{core_questions}

## Experience Mapping
{experience_mapping}

## Probing Questions
{probing_questions}

## Wrap-Up
{wrap_up}

## Moderator Notes
{moderator_notes}

## Materials Needed
{materials_needed}

## Analysis Plan
{analysis_plan}
"""
    
    def _create_research_summary_template(self) -> str:
        """Create Research Summary template."""
        return """
# Research Summary Report

## Executive Summary
{executive_summary}

## Research Objectives
{research_objectives}

## Methodology
{methodology}

## Participant Demographics
{participant_demographics}

## Key Findings
{key_findings}

## Detailed Insights
{detailed_insights}

## Customer Quotes
{customer_quotes}

## Patterns & Themes
{patterns_themes}

## Recommendations
{recommendations}

## Next Steps
{next_steps}

## Appendix: Research Materials
{research_materials}
"""
    
    def _create_persona_template(self) -> str:
        """Create Persona template."""
        return """
# Customer Persona Document

## Persona Overview
{persona_overview}

## Demographic Profile
{demographic_profile}

## Psychographic Profile
{psychographic_profile}

## Goals & Motivations
{goals_motivations}

## Pain Points & Frustrations
{pain_points}

## Decision-Making Factors
{decision_factors}

## Technology Usage
{technology_usage}

## Communication Preferences
{communication_preferences}

## Typical Day
{typical_day}

## Quotes
{quotes}

## Scenarios
{scenarios}

## Design Implications
{design_implications}
"""
    
    # Implementation Support Templates
    
    def _create_project_plan_template(self) -> str:
        """Create Project Plan template."""
        return """
# Project Plan

## Project Overview
{project_overview}

## Objectives
{objectives}

## Scope
{scope}

## Team Structure
{team_structure}

## Roles & Responsibilities
{roles_responsibilities}

## Timeline & Milestones
{timeline_milestones}

## Workstreams
{workstreams}

## Deliverables
{deliverables}

## Interdependencies
{interdependencies}

## Risk Management
{risk_management}

## Change Management
{change_management}

## Communication Plan
{communication_plan}

## Budget
{budget}

## Approval Process
{approval_process}
"""
    
    def _create_executive_presentation_template(self) -> str:
        """Create Executive Presentation template."""
        return """
# Executive Presentation

## Background & Context
{background_context}

## Objectives
{objectives}

## Current State Assessment
{current_state}

## Key Insights
{key_insights}

## Recommendations
{recommendations}

## Implementation Approach
{implementation_approach}

## Benefits & ROI
{benefits_roi}

## Timeline
{timeline}

## Resource Requirements
{resource_requirements}

## Next Steps
{next_steps}

## Q&A
{q_and_a}
"""
    
    def _create_change_management_template(self) -> str:
        """Create Change Management template."""
        return """
# Change Management Approach

## Change Overview
{change_overview}

## Stakeholder Assessment
{stakeholder_assessment}

## Impact Analysis
{impact_analysis}

## Change Vision & Strategy
{change_vision}

## Leadership Alignment
{leadership_alignment}

## Communication Plan
{communication_plan}

## Training & Enablement
{training_enablement}

## Resistance Management
{resistance_management}

## Measurement & Reinforcement
{measurement_reinforcement}

## Change Network
{change_network}

## Sustainability Plan
{sustainability_plan}

## Timeline
{timeline}
""" 