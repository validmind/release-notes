# Static configuration
RELEASES_DIR = "releases"
REPOS = ["backend", "frontend", "agents", "installation", "documentation", "validmind-library"]

# Label hierarchy for organizing release notes
label_hierarchy = ["highlight", "enhancement", "breaking-change", "deprecation", "bug", "documentation"]

# Labels that should exclude PRs from release notes
EXCLUDED_LABELS = ["internal", "auto-merge"]

label_to_category = {
    "highlight": "## Release highlights",
    "enhancement": "## Enhancements",
    "breaking-change": "## Breaking changes",
    "deprecation": "## Deprecations",
    "bug": "## Bug fixes",
    "documentation": "## Documentation"
}

categories = { 
    "highlight": [],
    "enhancement": [],
    "breaking-change": [],
    "deprecation": [],
    "bug": [],
    "documentation": []
}

# --- Section definitions ---
INCLUDED_SECTIONS = [
    "Release notes",
]

EXCLUDED_SECTIONS = [
    "What and why?",
    "How to test",
    "What needs special review?"
    "Dependencies, breaking changes, and deployment notes",
    "Breaking changes",
    "Dependencies",
    "Deployment",
    "Checklist",
    "Review",
    "Testing",
    "Internal"
    "Screenshots"
]

# --- LLM settings ---
# Model settings
MODEL_CLASSIFYING = "gpt-4o-mini"  # For quick section classification
MODEL_EDITING = "gpt-4o"           # For content editing (single-pass and default multi-pass)
MODEL_VALIDATION = "gpt-4o"        # For edit validation
MODEL_PROOFREADING = "gpt-4o-mini" # For proofreading tasks

# Multi-pass editing model settings
MODEL_PASS_1 = "gpt-4o"       # Pass 1: Clean and flatten
MODEL_PASS_2 = "gpt-4o"       # Pass 2: Deduplicate
MODEL_PASS_3 = "gpt-4o"       # Pass 3: Streamline and proofread

# Editing temperature settings
BASE_TEMPERATURE = 0.3          # Slightly higher for better creativity in editing
MAX_TEMPERATURE = 0.7           # Lower max to maintain consistency
TEMPERATURE_INCREMENT = 0.1     # Larger increments for faster adaptation
TEMPERATURE_FORMAT_ADJUST = 0.1
TEMPERATURE_MEANING_ADJUST = -0.05

# Validation temperature settings
VALIDATION_BASE_TEMPERATURE = 0.2       # Starting temperature for validation
VALIDATION_TEMPERATURE_INCREMENT = 0.15 # How much to increase per attempt
VALIDATION_MAX_TEMPERATURE = 0.5        # Maximum validation temperature

# Model parameters
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0

def get_model_api_params(model, max_tokens_value, temperature=BASE_TEMPERATURE):
    """Get model-specific API parameters for OpenAI calls.
    
    Args:
        model (str): Model name (e.g., 'o3', 'o3-mini', 'gpt-4o')
        max_tokens_value (int): Maximum tokens value
        temperature (float): Temperature value
        
    Returns:
        dict: API parameters suitable for client.chat.completions.create()
    """
    api_params = {"model": model}
    
    # Handle token parameter based on model
    if model in ["o3", "o3-mini"]:
        api_params["max_completion_tokens"] = max_tokens_value
        # o3 models only support default temperature (1), not custom values
        # Don't set temperature, frequency_penalty, or presence_penalty
    else:
        api_params["max_tokens"] = max_tokens_value
        api_params["temperature"] = temperature
        api_params["frequency_penalty"] = FREQUENCY_PENALTY
        api_params["presence_penalty"] = PRESENCE_PENALTY
    
    return api_params

# Token limits
MAX_TOKENS_CLASSIFICATION = 10
MAX_TOKENS_VALIDATION = 200
MAX_TOKENS_EDITING = 4096
MAX_TOKENS_PROOFREADING = 200

# Retry and delay settings
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_DELAY = 0.1
DEFAULT_MAX_DELAY = 1
MIN_SLEEP_TIME = 0.1
JITTER_RANGE = (0, 0.1)
JITTER_RANGE_WIDE = (-0.2, 0.2)

# Content limits
SUMMARY_CHAR_LIMIT = 250
PROOFREAD_MAX_TRIES = 5

# --- Editing prompts ---
EDIT_TITLE_PROMPT = (
    "Edit the following PR title for release notes:\n"
    "- Keep it under 80 characters, in a single line\n"
    "- Remove ticket numbers, branch names, prefixes, and trailing periods\n"
    "- Enclose technical terms (e.g., file names, words_with_underscores) in backticks\n"
    "- Use sentence-style capitalization (only the first word and proper nouns)\n"
    "- Make it clear and concise for end users\n"
    "- Do not include PR body or summary content\n"
    "- Do not add extra context or explanatory text\n\n"
    "{title}\n"
    "{body}"
)

# --- Content editing instructions ---
EDIT_CONTENT_SYSTEM = (
     "You are a professional release notes editor. Your task is to clean up, deduplicate and streamline content.\n"
    "The result should be clear and concise for an external audience without access to the codebase, making the changes easy to understand."
)

CORE_EDITING_PRINCIPLES = (
    "- Keep original meaning and technical accuracy\n"
    "- Use simple, clear language addressed to 'you' not 'users'\n"
    "- Use sentence-style capitalization, uppercase acronyms\n"
    "- Use sentence-style capitalization (only the first word and proper nouns)\n"
    "- Uppercase acronyms (e.g., 'LLM', 'API') and spell proper names correctly\n"
    "- Enclose technical terms in backticks\n"
    "- Follow Quarto formatting with proper spacing\n"
    "- Ensure content starts with text, not images\n"
    "- Keep code formatting (backticks) intact\n"
    "- Replace 'this PR' with 'this update'\n"
    "- Don't alter comment tags (<!-- ... -->)\n"
)

EDIT_CONTENT_PROMPT = (
    "When editing content:\n"
    f"{CORE_EDITING_PRINCIPLES}\n"
)

# --- Content quality assessment instructions for Pass 0 ---
PASS_0_ASSESSMENT_SYSTEM = (
    "You are a content quality assessor for release notes. Your task is to analyze the initial content quality "
    "and generate tailored editing instructions for subsequent processing passes."
)

PASS_0_ASSESSMENT_PROMPT = (
    "Analyze the following release notes content and generate tailored editing instructions for each pass. "
    "Only include editing steps that are actually needed based on the specific content issues you identify.\n\n"
    
    "ANALYSIS AREAS:\n"
    "1. CONTENT STRUCTURE: Redundant headings, poor organization, missing user benefit statements\n"
    "2. DUPLICATION: Overlap between PR summary and external notes, repeated features/changes\n"
    "3. CLARITY & STYLE: Verbose language, technical jargon, formatting issues\n"
    "4. COMPLETENESS: Missing context, unclear explanations, formatting problems\n\n"
    
    "INSTRUCTION GUIDELINES:\n"
    "- Only include steps that address actual issues found in this specific content\n"
    "- Be specific about what to change, not generic advice\n"
    "- If no issues exist in a category, say 'No changes needed' for that pass\n"
    "- Focus on the most impactful changes first\n"
    "- Provide actionable, specific instructions rather than general principles\n\n"
    
    "Content to analyze:\n"
    "PR Summary: {pr_summary}\n\n"
    "External Notes: {external_notes}\n\n"
    
    "Generate tailored, content-specific editing instructions that address the actual issues present in this content. "
    "Your instructions should work together with these core editing principles:\n\n"
    f"CORE EDITING PRINCIPLES:\n{CORE_EDITING_PRINCIPLES}\n\n"
    "Focus your tailored instructions on the specific content issues you identify beyond these core principles.\n\n"
    "ASSESSMENT:\n"
    "[Brief assessment of main issues found]\n\n"
    "PASS_1_INSTRUCTIONS:\n"
    "[Content-specific cleanup and structure instructions - address actual structural issues in this content]\n\n"
    "PASS_2_INSTRUCTIONS:\n"
    "[Content-specific deduplication instructions - only include if actual duplication exists]\n\n"
    "PASS_3_INSTRUCTIONS:\n"
    "[Content-specific streamlining instructions - address actual clarity/flow issues in this content]"
)

# --- Instruction validation prompt ---
INSTRUCTION_VALIDATION_SYSTEM = "You are a quality checker for editing instructions. Assess if the generated instructions are specific and actionable."

INSTRUCTION_VALIDATION_PROMPT = (
    "Review these generated editing instructions and determine if they are specific and actionable:\n\n"
    "{instructions}\n\n"
    "VALIDATION CRITERIA:\n"
    "1. Are instructions specific to the actual content issues (not generic advice)?\n"
    "2. Do they provide concrete, actionable steps?\n"
    "3. Do they avoid unnecessary work by only addressing real issues?\n"
    "4. Are they clear about what specifically needs to be changed?\n"
    "5. Do they include 'No changes needed' for passes where no issues exist?\n\n"
    "PASS if instructions are tailored, specific, and actionable.\n"
    "FAIL if instructions are generic, vague, or include unnecessary steps.\n\n"
    "Respond with exactly 'PASS' or 'FAIL: [specific reason]'."
)

# --- Content validation instructions ---
VALIDATION_SYSTEM = "You are a quality checker for release notes. Be lenient and only fail content with serious issues."

VALIDATION_PROMPT = (
    "Check this edited content for CRITICAL ISSUES ONLY. Most content should PASS.\n"
    "\n"
    "FAIL ONLY if you find:\n"
    "1. Exact duplicate sentences (word-for-word identical).\n"
    "2. Internal development sections like '# PR Summary', '## Checklist', '## Testing'.\n"
    "3. Content starts with ![image] or <img tag instead of text.\n"
    "4. Completely fabricated information not in the original.\n"
    "\n"
    "ALWAYS PASS if:\n"
    "- Content starts with text (even if it has images later).\n"
    "- No internal development sections.\n"
    "- No exact word-for-word duplicate sentences.\n"
    "- Content is professional and reasonable.\n"
    "- Similar sentences with different wording (this is normal editing).\n"
    "- Temporal language like 'now', 'this release', 'you can now'.\n"
    "- Technical terms, bullet points, formatting improvements.\n"
    "- Reasonable length changes or rewording for clarity.\n"
    "\n"
    "Be VERY LENIENT. Only fail for the 4 critical issues above.\n"
    "\n"
    "Respond with exactly 'PASS' or 'FAIL: [brief reason]'."
)

VALIDATION_CRITERIA = {
    'title': [
        "Is clear and professional",
        "Does not contain obvious errors"
    ],
    'notes': [
        "Starts with text, not images",
        "Does not contain internal sections",
        "Does not have identical duplicate sentences"
        "Is not substantially longer than the original"
    ]
}

# --- Section classification prompt ---
SECTION_CLASSIFICATION_SYSTEM = "You are a release notes classifier. Your job is to determine if a section should be included in public release notes."

SECTION_CLASSIFICATION_PROMPT = (
    "Classify if this section should be included in public release notes.\n"
    "Section title: {title}\n"
    "Section content: {content}\n\n"
    "Rules:\n"
    "1. Include sections that describe user-facing changes, features, or improvements.\n"
    "2. Include sections about dependencies, breaking changes, or upgrade notes.\n"
    "3. Include sections with screenshots or media.\n"
    "4. Exclude internal notes, checklists, deployment steps, or review points.\n"
    "5. Exclude sections about testing, QA, or development processes.\n"
    "6. Exclude sections marked as internal or for team use only.\n\n"
    "Respond with only 'INCLUDE' or 'EXCLUDE'."
)

# --- Content extraction prompt ---
CONTENT_EXTRACTION_SYSTEM = "You are a release notes content extractor. Your job is to extract only the user-facing content from PR descriptions."

CONTENT_EXTRACTION_PROMPT = (
    "Extract only the content that should be included in public release notes from this PR body.\n\n"
    "INCLUDE these sections if present:\n"
    "{included_sections}\n\n"
    "EXCLUDE these sections if present:\n"
    "{excluded_sections}\n\n"
    "Additional rules:\n"
    "1. Always extract any mentions of breaking changes, even if they appear in excluded sections.\n"
    "2. Include sections with screenshots or media that relate to user-facing changes.\n"
    "3. Preserve the original markdown formatting and structure.\n"
    "4. Remove the '## Release notes' heading\n"
    "5. If no relevant content is found, respond with 'NO_CONTENT'.\n\n"
    "PR Body:\n{pr_body}\n\n"
    "Extract and return only the relevant content, maintaining original formatting:"
)

# --- PR summary extraction prompt ---
PR_SUMMARY_EXTRACTION_SYSTEM = "You are a PR summary extractor. Your job is to find and extract PR summaries from GitHub comments."

PR_SUMMARY_EXTRACTION_PROMPT = (
    "Find the PR summary in these GitHub comments and extract ONLY the first paragraph.\n\n"
    "Steps:\n"
    "1. Find a comment with '# PR Summary' heading\n"
    "2. Extract ONLY the first paragraph after this heading\n"
    "3. Stop at the first line break, bullet point, numbered list, or colon\n\n"
    "Rules:\n"
    "- Extract ONLY the first paragraph - nothing else\n"
    "- Stop immediately at bullet points (•, -, *), numbers (1., 2.), or colons (:)\n"
    "- Remove the '# PR Summary' heading\n"
    "- Replace 'this PR' with 'this update'\n"
    "- If no summary found, respond with 'NO_SUMMARY'\n\n"
    "Comments:\n{comments}\n\n"
    "Extract ONLY the first paragraph:"
)

# --- Merge PR classification prompt ---
MERGE_PR_CLASSIFICATION_SYSTEM = "You are a PR classifier. Your job is to determine if a PR represents an automatic merge or contains actual changes."

# Keywords that indicate a PR is an automatic merge
MERGE_KEYWORDS = [
    "merge main into staging", "merge staging into prod", "main branch", "branch merged", "merged into", "merge main", "merge staging",
    "merge branch", "merge pull request", "merge into", "release", "deploy", "sync", "auto-merge",
    "merge production", "merge develop", "merge hotfix", "merge feature", "merge bugfix",
    "merge master", "merge to", "merge from", "merge changes", "merge update", "merge upstream",
    "merge down", "merge up", "merge test", "merge qa", "merge rc", "merge candidate",
    "sync main", "sync develop", "sync branch", "sync changes", "sync upstream",
    "sync staging", "sync production", "sync with main", "sync main into", "sync staging into",
    "sync staging branch", "synchronize", "automatically merge", "automatically sync",
    "deploy to", "deploy prod", "deploy production", "deploy staging", "deploy develop",
    "release candidate", "release hotfix", "release patch", "release update",
    "prod:", "staging:", "production:", "hotfix:", "bump version", "version bump"
]

MERGE_PR_CLASSIFICATION_PROMPT = (
    "Analyze this PR title and determine if it represents an automatic merge PR.\n"
    "A merge PR typically:\n"
    "- Merges branches or changes between environments\n"
    "- Syncs code between branches\n"
    "- Deploys code to different environments\n"
    "- Updates release candidates or hotfixes\n"
    f"- Contains keywords like (case-insensitive): {', '.join(MERGE_KEYWORDS)}\n\n"
    "PR Title: {title}\n\n"
    "Respond with only 'MERGE' if this is an automatic merge PR, or 'NOT_MERGE' if it contains actual changes."
)

# --- Summary proofreading prompt for generate_highlevel_summary output ---
MODEL_PROOFREADING_SYSTEM = "You are a professional technical writer."

PROOFREAD_SUMMARY_PROMPT = (
    "Proofread and streamline the following release summary for clarity and natural flow. "
    "Keep it concise, user-facing, and ensure it starts with 'This release includes'. "
    "Do not add or remove features, just improve the language and flow. "
    "Return only the improved summary.\n\n"
    "Summary:\n{summary}"
)

import subprocess
import json
import re
import shutil
import numpy as np
import datetime
import openai
from dotenv import dotenv_values
import os
from collections import defaultdict
from IPython import get_ipython
from collections import Counter
import sys
import requests
import argparse
import concurrent.futures
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin
from playwright.sync_api import sync_playwright
import base64
import imghdr
from difflib import SequenceMatcher

ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def unified_openai_call(messages, api_params, debug=False, function_name="OpenAI", 
                        max_retries=DEFAULT_MAX_RETRIES, success_condition=None, 
                        failure_callback=None, retry_callback=None, return_full_response=False):
    """
    Unified OpenAI API call with exponential backoff, jitter, and flexible success/failure handling.
    
    Args:
        messages: The messages to send to the API
        api_params: The API parameters (model, max_tokens, temperature, etc.)
        debug: Whether to print debug information
        function_name: Name of the calling function for error messages
        max_retries: Maximum number of retry attempts
        success_condition: Optional function to validate if response is successful (response) -> bool
        failure_callback: Optional function called on each failure (attempt, error, response) -> bool (continue retrying)
        retry_callback: Optional function called before each retry (attempt, last_response) -> modified_messages or None
        return_full_response: If True, return full response object; if False, return content string
        
    Returns:
        The API response (full object or content string), or None if all retries failed
    """
    delay = DEFAULT_INITIAL_DELAY
    last_response = None
    
    for attempt in range(max_retries):
        try:
            # Allow retry callback to modify messages
            current_messages = messages
            if retry_callback and attempt > 0:
                modified_messages = retry_callback(attempt, last_response)
                if modified_messages is not None:
                    current_messages = modified_messages
            
            response = client.chat.completions.create(
                messages=current_messages,
                **api_params
            )
            
            last_response = response
            
            # Check custom success condition if provided
            if success_condition:
                if success_condition(response):
                    return response if return_full_response else response.choices[0].message.content.strip()
                else:
                    # Custom validation failed, treat as failure
                    if failure_callback:
                        should_continue = failure_callback(attempt, "Custom validation failed", response)
                        if not should_continue:
                            return response if return_full_response else response.choices[0].message.content.strip()
                    
                    if attempt < max_retries - 1:
                        if debug:
                            print(f"DEBUG: [{function_name}] Custom validation failed on attempt {attempt + 1}, retrying...")
                    continue  # Retry
            else:
                # No custom validation, return successful API response
                return response if return_full_response else response.choices[0].message.content.strip()
                
        except Exception as e:
            if debug:
                print(f"DEBUG: [{function_name}] Attempt {attempt + 1} failed: {e}")
            
            # Call failure callback
            if failure_callback:
                should_continue = failure_callback(attempt, str(e), None)
                if not should_continue:
                    return None
            
            if attempt < max_retries - 1:
                # Add jitter to prevent thundering herd
                jitter = random.uniform(*JITTER_RANGE) * delay
                sleep_time = min(DEFAULT_MAX_DELAY, delay + jitter)
                
                if debug:
                    print(f"DEBUG: [{function_name}] Retrying in {sleep_time:.1f} seconds...")
                
                time.sleep(sleep_time)
                delay = min(DEFAULT_MAX_DELAY, delay * 2)  # Exponential backoff
            else:
                if debug:
                    print(f"DEBUG: [{function_name}] All {max_retries} attempts failed")
    
    return None

# Backward compatibility alias
def robust_openai_call(messages, api_params, debug=False, function_name="OpenAI", max_retries=DEFAULT_MAX_RETRIES):
    """Legacy function - use unified_openai_call instead."""
    return unified_openai_call(messages, api_params, debug, function_name, max_retries)

def classify_section(section_title, section_content, debug=False):
    """Use OpenAI to classify whether a section should be included in release notes.
    
    Args:
        section_title (str): The section title
        section_content (str): The section content
        debug (bool): Whether to show debug output
        
    Returns:
        bool: True if section should be included, False if excluded
    """
    if debug:
        print(f"\nDEBUG: [classify_section] Classifying section:")
        print(f"DEBUG: [classify_section] Title: {section_title}")
        print(f"DEBUG: [classify_section] Content: {section_content[:200]}...")  # Show first 200 chars of content
    
    prompt = SECTION_CLASSIFICATION_PROMPT.format(
        title=section_title,
        content=section_content
    )
        
    api_params = get_model_api_params(MODEL_CLASSIFYING, MAX_TOKENS_CLASSIFICATION)
    messages = [
        {
            "role": "system",
            "content": SECTION_CLASSIFICATION_SYSTEM
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    result = robust_openai_call(messages, api_params, debug, "classify_section")
    
    if result:
        result_upper = result.upper()
        if debug:
            print(f"DEBUG: [classify_section] OpenAI response: {result_upper}")
        return result_upper == 'INCLUDE'
    else:
        if debug:
            print("DEBUG: [classify_section] All OpenAI attempts failed, defaulting to exclude section")
        # Default to excluding section if OpenAI fails
        return False

def extract_relevant_content(pr_body, debug=False):
    """Use OpenAI to extract only relevant content from PR body for release notes.
    
    Args:
        pr_body (str): The full PR body content
        debug (bool): Whether to show debug output
        
    Returns:
        str: Extracted relevant content, or None if no relevant content found
    """
    if not pr_body or not pr_body.strip():
        return None
        
    if debug:
        print(f"\nDEBUG: [extract_relevant_content] Extracting relevant content from PR body")
        print(f"DEBUG: [extract_relevant_content] PR body length: {len(pr_body)}")
    
    # Format the sections lists for the prompt
    included_sections_text = "\n".join(f"- {section}" for section in INCLUDED_SECTIONS)
    excluded_sections_text = "\n".join(f"- {section}" for section in EXCLUDED_SECTIONS)
    
    prompt = CONTENT_EXTRACTION_PROMPT.format(
        pr_body=pr_body,
        included_sections=included_sections_text,
        excluded_sections=excluded_sections_text
    )
        
    api_params = get_model_api_params(MODEL_CLASSIFYING, MAX_TOKENS_EDITING)  # Use higher token limit for extraction
    messages = [
        {
            "role": "system",
            "content": CONTENT_EXTRACTION_SYSTEM
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    result = robust_openai_call(messages, api_params, debug, "extract_relevant_content")
    
    if result:
        if debug:
            print(f"DEBUG: [extract_relevant_content] OpenAI response length: {len(result)}")
            print(f"DEBUG: [extract_relevant_content] OpenAI response (first 200 chars): {result[:200]}...")
        
        # Check if LLM found no relevant content
        if result.upper() == 'NO_CONTENT':
            if debug:
                print(f"DEBUG: [extract_relevant_content] LLM found no relevant content")
            return None
            
        return result
    else:
        if debug:
            print("DEBUG: [extract_relevant_content] All OpenAI attempts failed, returning None")
        # Return None if extraction fails
        return None

def extract_pr_summary(comments, debug=False):
    """Use OpenAI to extract PR summary from GitHub comments.
    
    Args:
        comments (list): List of GitHub comment objects
        debug (bool): Whether to show debug output
        
    Returns:
        str: Extracted PR summary, or None if no summary found
    """
    if not comments:
        return None
        
    if debug:
        print(f"\nDEBUG: [extract_pr_summary] Extracting PR summary from {len(comments)} comments")
    
    # Combine all comments into a single text for LLM processing
    comments_text = ""
    for i, comment in enumerate(comments):
        body = comment.get('body', '')
        if body.strip():
            comments_text += f"\n--- Comment {i+1} ---\n{body}\n"
    
    if not comments_text.strip():
        if debug:
            print("DEBUG: [extract_pr_summary] No comment content found")
        return None
    
    prompt = PR_SUMMARY_EXTRACTION_PROMPT.format(comments=comments_text)
        
    api_params = get_model_api_params(MODEL_CLASSIFYING, MAX_TOKENS_EDITING)  # Use higher token limit for extraction
    messages = [
        {
            "role": "system",
            "content": PR_SUMMARY_EXTRACTION_SYSTEM
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    result = robust_openai_call(messages, api_params, debug, "extract_pr_summary")
    
    if result:
        if debug:
            print(f"DEBUG: [extract_pr_summary] OpenAI response length: {len(result)}")
            print(f"DEBUG: [extract_pr_summary] OpenAI response (first 200 chars): {result[:200]}...")
        
        # Check if LLM found no summary
        if result.upper() == 'NO_SUMMARY':
            if debug:
                print(f"DEBUG: [extract_pr_summary] LLM found no PR summary")
            return None
            
        return result
    else:
        if debug:
            print("DEBUG: [extract_pr_summary] All OpenAI attempts failed, returning None")
        # Return None if extraction fails
        return None

def generate_validation_comment(commit, debug=False):
    """Generate HTML comment with validation summary information.
    
    Args:
        commit (dict): Commit/PR data with validation information
        debug (bool): Whether to include debug information
        
    Returns:
        str: HTML comment with validation summary (only if debug=True)
    """
    # Only output validation summary if debug is enabled
    if not debug:
        return ""
        
    # Handle both single validation_summary and multiple validation_summaries
    validation_summaries = []
    
    if commit.get('validation_summaries'):
        validation_summaries = commit['validation_summaries']
    elif commit.get('validation_summary'):
        validation_summaries = [commit['validation_summary']]
    
    if not validation_summaries:
        return ""
    
    all_comment_lines = []
    
    for i, validation_info in enumerate(validation_summaries):
        if i == 0:
            header = "\n\n<!--- VALIDATION SUMMARY"
        else:
            header = f"\nVALIDATION SUMMARY {i+1}"
            
        comment_lines = [
            header,
            f"Content Type: {validation_info.get('content_type', 'unknown')}",
            f"Validation Status: {'CHECK' if validation_info.get('validation_failed') else 'PASSED'}",
            f"Attempts: {validation_info.get('attempts', 'unknown')}",
            f"Validation Temperature: {validation_info.get('validation_temperature', 'unknown')}",

            f"Last Validation: {validation_info.get('last_validation_time', 'unknown')}"
        ]
        
        if validation_info.get('validation_result'):
            result = validation_info['validation_result']
            # Truncate very long results
            if len(result) > 300:
                result = result[:300] + "..."
            comment_lines.append(f"Result: {result}")
        
        if validation_info.get('failure_patterns'):
            patterns = validation_info['failure_patterns']
            comment_lines.append(f"Failure Patterns: {patterns}")
        
        if validation_info.get('reedit_available'):
            comment_lines.append("Reedit Available: Yes")
            if validation_info.get('reedit_message'):
                # Truncate long messages
                msg = validation_info['reedit_message']
                if len(msg) > 200:
                    msg = msg[:200] + "..."
                comment_lines.append(f"Reedit Message: {msg}")
        
        all_comment_lines.extend(comment_lines)
    
    all_comment_lines.append("--->\n")
    return "\n".join(all_comment_lines)

class PR:
    def __init__(self, repo_name=None, pr_number=None, title=None, body=None, url=None, labels=None, debug=False):
        self.repo_name = repo_name
        self.pr_number = pr_number
        self.url = url
        self.data_json = None
        self.debug = debug
        
        self.title = title
        self.cleaned_title = None
        self.pr_body = None
        self.labels = labels if labels is not None else []

        self.generated_lines = None
        self.edited_text = None
        
        self.pr_interpreted_summary = None
        self.pr_details = None # final form
        self.validated = False  # Track if any content was validated
        self.last_validation_result = None  # Add this attribute to store the result

    def load_data_json(self):
        """Loads the JSON data from a PR to self.data_json, sets to None if any labels are 'internal'

        Modifies:
            self.data_json
        """
        print(f"Storing data from PR #{self.pr_number} of {self.repo_name} ...\n")
        cmd = ['gh', 'pr', 'view', self.pr_number, '--json', 'title,body,url,labels', '--repo', self.repo_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout.strip()

        output_clean = ansi_escape.sub('', output)

        try:
            self.data_json = json.loads(output_clean)
        except json.JSONDecodeError:
            print(f"ERROR: Unable to parse PR data for PR number {self.pr_number} in repository {self.repo_name}")
            return None
        
        if any(label['name'] == 'internal' for label in self.data_json['labels']):
            self.data_json = None  # Ignore PRs with the 'internal' label
        
    def extract_external_release_notes(self):
        """Turns the JSON body into lines (str) that are ready for ChatGPT
        
        Modifies: 
            self.pr_body - Contains JSON body
            self.generated_lines - Converted string ready for ChatGPT
        """
        self.pr_body = self.data_json['body']
        
        if self.debug:
            print("\nDEBUG: [extract_external_release_notes] Extracting external release notes")
            print(f"DEBUG: [extract_external_release_notes] PR body length: {len(self.pr_body)}")
        
        # Extract only the sections we want to keep
        sections = []
        
        # Split the body into sections with more flexible matching
        section_pattern = r"##\s*([^\n]+)\s*(.+?)(?=^##\s*|\Z)"
        matches = re.finditer(section_pattern, self.pr_body, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            section_title = match.group(1).strip()
            section_content = match.group(2).strip()
            
            if self.debug:
                print(f"\nDEBUG: [extract_external_release_notes] Found section: {section_title}")
                print(f"DEBUG: [extract_external_release_notes] Section content (first 200 chars): {section_content[:200]}")
            
            # Use OpenAI to classify the section
            if classify_section(section_title, section_content, self.debug):
                if self.debug:
                    print(f"DEBUG: [extract_external_release_notes] Including section: {section_title}")
                sections.append(f"## {section_title}\n{section_content}")
            else:
                if self.debug:
                    print(f"DEBUG: [extract_external_release_notes] Excluding section: {section_title}")
        
        # If we found any sections, combine them
        if sections:
            if self.debug:
                print(f"\nDEBUG: [extract_external_release_notes] Found {len(sections)} sections to include")
            self.generated_lines = "\n\n".join(sections)
            return True
            
        if self.debug:
            print("\nDEBUG: [extract_external_release_notes] No sections found, trying fallback patterns")
            
        # Fallback to old format with more robust matching
        old_format_patterns = [
            r"##\s*External\s+Release\s+Notes\s*(.+)",
            r"##\s*Release\s+Notes\s*(.+)",
            r"##\s*Notes\s*(.+)",
            r"##\s*What's\s+New\s*(.+)",
            r"##\s*Changes\s*(.+)",
            r"##\s*Overview\s*(.+)"
        ]
        
        for pattern in old_format_patterns:
            match = re.search(pattern, self.pr_body, re.DOTALL | re.IGNORECASE)
            if match:
                if self.debug:
                    print(f"DEBUG: Found match with pattern: {pattern}")
                extracted_text = match.group(1).strip()
                self.generated_lines = '\n'.join(''.join(['#', line]) if line.lstrip().startswith('###') else line for line in extracted_text.split('\n'))
                return True
                
        if self.debug:
            print("DEBUG: [extract_external_release_notes] No sections found in fallback patterns")
        return None
        


    def _assess_content_quality(self, external_notes, pr_summary, debug=False):
        """Pass 0: Assess content quality and generate tailored editing instructions."""
        print(f"Analyzing content quality for PR #{self.pr_number} in {self.repo_name} ...")
        if debug:
            print(f"DEBUG: [_assess_content_quality] PR #{self.pr_number} in {self.repo_name} - Analyzing content quality")
        
        max_attempts = DEFAULT_MAX_RETRIES
        for attempt in range(max_attempts):
            try:
                # Adjust temperature for retries to get more specific instructions
                temperature = BASE_TEMPERATURE + (attempt * TEMPERATURE_INCREMENT)
                
                prompt = PASS_0_ASSESSMENT_PROMPT.format(
                    pr_summary=pr_summary or "None",
                    external_notes=external_notes or "None"
                )
                
                # Add retry guidance for subsequent attempts
                if attempt > 0:
                    prompt += f"\n\nPREVIOUS ATTEMPT FEEDBACK: Instructions were too generic. Be more specific about actual issues in this content and avoid generic editing advice."
                
                api_params = get_model_api_params(MODEL_EDITING, MAX_TOKENS_EDITING, temperature)
                messages = [
                    {"role": "system", "content": PASS_0_ASSESSMENT_SYSTEM},
                    {"role": "user", "content": prompt}
                ]
                
                result = unified_openai_call(
                    messages=messages,
                    api_params=api_params,
                    debug=debug,
                    function_name="_assess_content_quality"
                )
                
                if result:
                    if debug:
                        print(f"DEBUG: [_assess_content_quality] Attempt {attempt + 1} result length: {len(result)}")
                    
                    # Parse the response to extract instructions for each pass
                    instructions = self._parse_assessment_result(result, debug)
                    
                    # Validate the generated instructions
                    if instructions and self._validate_instructions(instructions, debug):
                        if debug:
                            print(f"DEBUG: [_assess_content_quality] Generated tailored instructions successfully on attempt {attempt + 1}")
                        
                        # Track successful Pass 0 assessment
                        if not hasattr(self, 'validation_summaries'):
                            self.validation_summaries = []
                        
                        assessment_summary = {
                            'content_type': 'pass_0_assessment',
                            'validation_failed': False,
                            'attempts': attempt + 1,
                            'validation_result': "PASS: Tailored instructions generated and validated",
                            'failure_patterns': {},
                            'validation_temperature': temperature,
                            'last_validation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        self.validation_summaries.append(assessment_summary)
                        
                        return instructions
                    else:
                        if debug:
                            print(f"DEBUG: [_assess_content_quality] Attempt {attempt + 1} validation failed")
                        if attempt < max_attempts - 1:
                            print(f"Instructions not specific enough, retrying... (attempt {attempt + 2}/{max_attempts})")
                else:
                    if debug:
                        print(f"DEBUG: [_assess_content_quality] Attempt {attempt + 1} returned None")
                    if attempt < max_attempts - 1:
                        print(f"Error generating instructions, retrying... (attempt {attempt + 2}/{max_attempts})")
                        
            except Exception as e:
                if debug:
                    print(f"DEBUG: [_assess_content_quality] Attempt {attempt + 1} error: {e}")
                if attempt < max_attempts - 1:
                    print(f"Error generating instructions, retrying... (attempt {attempt + 2}/{max_attempts})")
        
        # If all attempts failed, this is a critical error since we need tailored instructions
        print(f"ERROR: Failed to generate tailored editing instructions for PR #{self.pr_number} after {max_attempts} attempts")
        
        # Track failed Pass 0 assessment
        if not hasattr(self, 'validation_summaries'):
            self.validation_summaries = []
        
        assessment_summary = {
            'content_type': 'pass_0_assessment',
            'validation_failed': True,
            'attempts': max_attempts,
            'validation_result': "FAIL: Could not generate specific enough instructions",
            'failure_patterns': {'generic_instructions': max_attempts},
            'validation_temperature': BASE_TEMPERATURE + ((max_attempts - 1) * TEMPERATURE_INCREMENT),
            'last_validation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.validation_summaries.append(assessment_summary)
        
        # Return None - the calling code will need to handle this error case
        return None
    
    def _parse_assessment_result(self, result, debug=False):
        """Parse the assessment result to extract individual pass instructions."""
        try:
            instructions = {}
            
            # Extract each section using regex patterns
            pass_1_match = re.search(r'PASS_1_INSTRUCTIONS:\s*\n(.*?)(?=\n\nPASS_2_INSTRUCTIONS:|\Z)', result, re.DOTALL)
            pass_2_match = re.search(r'PASS_2_INSTRUCTIONS:\s*\n(.*?)(?=\n\nPASS_3_INSTRUCTIONS:|\Z)', result, re.DOTALL)
            pass_3_match = re.search(r'PASS_3_INSTRUCTIONS:\s*\n(.*?)(?=\n\n|\Z)', result, re.DOTALL)
            
            if pass_1_match:
                instructions['pass_1'] = pass_1_match.group(1).strip()
            if pass_2_match:
                instructions['pass_2'] = pass_2_match.group(1).strip()
            if pass_3_match:
                instructions['pass_3'] = pass_3_match.group(1).strip()
            
            if debug:
                print(f"DEBUG: [_parse_assessment_result] Extracted {len(instructions)} instruction sets")
            
            # Return None if we didn't get all three instruction sets
            if len(instructions) != 3:
                if debug:
                    print(f"DEBUG: [_parse_assessment_result] Missing instruction sets, got {list(instructions.keys())}")
                return None
                
            return instructions
            
        except Exception as e:
            if debug:
                print(f"DEBUG: [_parse_assessment_result] Error parsing assessment result: {e}")
            return None
    
    def _validate_instructions(self, instructions, debug=False):
        """Validate that the generated instructions are meaningful and specific."""
        try:
            # Combine all instructions for validation
            all_instructions = "\n\n".join([
                f"Pass 1: {instructions['pass_1']}",
                f"Pass 2: {instructions['pass_2']}",
                f"Pass 3: {instructions['pass_3']}"
            ])
            
            prompt = INSTRUCTION_VALIDATION_PROMPT.format(instructions=all_instructions)
            
            api_params = get_model_api_params(MODEL_VALIDATION, MAX_TOKENS_VALIDATION, VALIDATION_BASE_TEMPERATURE)
            messages = [
                {"role": "system", "content": INSTRUCTION_VALIDATION_SYSTEM},
                {"role": "user", "content": prompt}
            ]
            
            result = unified_openai_call(
                messages=messages,
                api_params=api_params,
                debug=debug,
                function_name="_validate_instructions"
            )
            
            if result is None:
                if debug:
                    print(f"DEBUG: [_validate_instructions] All OpenAI attempts failed")
                return False
            
            if debug:
                print(f"DEBUG: [_validate_instructions] Validation result: {result}")
            
            return result.upper().startswith('PASS')
            
        except Exception as e:
            if debug:
                print(f"DEBUG: [_validate_instructions] Error validating instructions: {e}")
            return False

    def edit_content(self, content_type, content, editing_instructions, edit=False, skip_passes=None):
        """Unified function to edit PR content (summaries, titles, or release notes) with three-pass editing for notes/summary."""
        if skip_passes is None:
            skip_passes = set()
        
        # Initialize validation summaries list if it doesn't exist
        if not hasattr(self, 'validation_summaries'):
            self.validation_summaries = []
        # Use multi-pass for 'notes', single pass for 'title'
        if content_type in ("notes",) and edit:
            # Pass 0: Assess content quality and generate tailored instructions
            tailored_instructions = None
            if 0 not in skip_passes:
                # For content quality assessment, we need both external_notes and pr_summary
                # The content parameter should be external_notes when content_type is 'notes'
                external_notes = content  # This is the external_notes passed to edit_content
                pr_summary = getattr(self, 'pr_interpreted_summary', None)
                tailored_instructions = self._assess_content_quality(external_notes, pr_summary, self.debug)
            
            # Tailored instructions are required - if Pass 0 failed, we can't proceed
            if not tailored_instructions:
                print(f"ERROR: Cannot proceed with editing PR #{self.pr_number} - Pass 0 failed to generate tailored instructions")
                # Set the content as-is without editing
                if content_type == 'notes':
                    self.edited_text = content
                return
            
            print(f"Using tailored editing instructions for PR #{self.pr_number}")
            pass_1_instructions = tailored_instructions['pass_1']
            pass_2_instructions = tailored_instructions['pass_2']
            pass_3_instructions = tailored_instructions['pass_3']
            
            # Pass 1: Group and Flatten
            if 1 not in skip_passes:
                grouped = self._edit_pass(
                    content_type,
                    content,
                    pass_1_instructions,
                    attr_name="grouped_text",
                    edit=edit
                )
            else:
                grouped = content
                self.grouped_text = grouped
            # Pass 2: Deduplicate
            if 2 not in skip_passes:
                deduped = self._edit_pass(
                    content_type,
                    grouped,
                    pass_2_instructions,
                    attr_name="deduplicated_text",
                    edit=edit
                )
            else:
                deduped = grouped
                self.deduplicated_text = deduped
            # Pass 3: Streamline and Summarise
            if 3 not in skip_passes:
                final = self._edit_pass(
                    content_type,
                    deduped,
                    pass_3_instructions,
                    attr_name="edited_text",
                    edit=edit
                )
            else:
                final = deduped
                self.edited_text = final
            # Set output based on content type
            if content_type == 'notes':
                self.edited_text = final
            return
        # Single pass for titles or fallback for other types
        if not edit:
            if content_type == 'title':
                self.cleaned_title = content.rstrip('.')
            elif content_type == 'notes':
                self.edited_text = content
            return
        if self.debug:
            print(f"DEBUG: [edit_content] PR #{self.pr_number} in {self.repo_name} - Editing {content_type}")
            print(f"DEBUG: [edit_content] Original content (first 100 chars): {repr(content[:100]) if content else None}")
            print(f"DEBUG: [edit_content] Editing instructions (first 100 chars): {repr(editing_instructions[:100]) if editing_instructions else None}")
        print(f"Editing {content_type} for PR #{self.pr_number} in {self.repo_name} ...")
        
        try:            
            # Combine the specific editing instructions with the general content instructions
            full_instructions = f"{editing_instructions}\n\n{EDIT_CONTENT_PROMPT}\n\nIMPORTANT: Maintain the scope of this specific PR (#{self.pr_number}). Do not merge content from other PRs or add information not present in the original content."
            
            # Initialize variables for retry loop
            max_attempts = DEFAULT_MAX_RETRIES
            initial_delay = DEFAULT_INITIAL_DELAY
            delay = initial_delay
            max_delay = DEFAULT_MAX_DELAY
            last_validation_result = None
            failure_patterns = {}  # Track patterns in failures
            content_for_reedit = None
            
            for attempt in range(max_attempts):
                # Calculate temperature based on attempt number and failure patterns
                base_temp = 0.2
                if 'formatting' in failure_patterns:
                    base_temp += 0.1  # Increase temperature for formatting issues
                if 'meaning' in failure_patterns:
                    base_temp -= 0.05  # Decrease temperature for meaning preservation issues
                temperature = min(0.7, base_temp + (attempt * 0.05))  # Gradual increase with cap
                
                # Add feedback from previous attempt if available
                if last_validation_result and attempt > 0:
                    # Analyze failure pattern
                    if 'formatting' in last_validation_result.lower():
                        failure_patterns['formatting'] = failure_patterns.get('formatting', 0) + 1
                    if 'meaning' in last_validation_result.lower():
                        failure_patterns['meaning'] = failure_patterns.get('meaning', 0) + 1
                    
                    # Add specific guidance based on failure patterns
                    guidance = []
                    if failure_patterns.get('formatting', 0) > 1:
                        guidance.append("Pay special attention to markdown formatting and structure.")
                    if failure_patterns.get('meaning', 0) > 1:
                        guidance.append("Focus on preserving the exact meaning and technical details.")
                    
                    full_instructions += f"\n\nPrevious attempt failed: {last_validation_result}"
                    if guidance:
                        full_instructions += f"\nPlease address these specific issues:\n" + "\n".join(f"- {g}" for g in guidance)
                
                # Add deduplication instruction to the editing prompt
                dedup_instruction = ("If content from the PR body and PR summary is duplicated or very similar, "
                                    "keep only the most clear and concise version and remove the duplicate.")
                full_instructions_with_dedup = f"{full_instructions}\n\n{dedup_instruction}"

                # Use content_for_reedit['edited'] if available from previous validation failure
                if attempt > 0 and content_for_reedit and 'edited' in content_for_reedit:
                    content_to_edit = content_for_reedit['edited']
                else:
                    content_to_edit = content

                # Make API call with model-specific parameters
                api_params = get_model_api_params(MODEL_EDITING, MAX_TOKENS_EDITING, temperature)
                
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": EDIT_CONTENT_SYSTEM
                        },
                        {
                            "role": "user",
                            "content": f"Instructions:\n{full_instructions_with_dedup}\n\nContent to edit:\n{content_to_edit}"
                        }
                    ],
                    **api_params
                )
                
                current_edit = response.choices[0].message.content.strip()
                if self.debug:
                    print(f"DEBUG: [edit_content] Attempt {attempt + 1} content (first 100 chars): {repr(current_edit[:100]) if current_edit else None}")
                
                # Validate current attempt
                is_valid, validation_result, content_for_reedit = self.validate_edit(content_type, content, current_edit, edit, attempt + 1)
                if self.debug:
                    print(f"DEBUG: [edit_content] Attempt {attempt + 1} validation result: {is_valid}")
                
                if is_valid:
                    edited_content = current_edit
                    self.last_validation_result = validation_result
                    
                    # Add successful validation summary
                    validation_summary = {
                        'content_type': content_type,
                        'validation_failed': False,
                        'attempts': attempt + 1,
                        'validation_result': validation_result,
                        'failure_patterns': failure_patterns,
                        'validation_temperature': min(VALIDATION_MAX_TEMPERATURE, VALIDATION_BASE_TEMPERATURE + attempt * VALIDATION_TEMPERATURE_INCREMENT),

                        'last_validation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.validation_summaries.append(validation_summary)
                    self.validation_summary = validation_summary
                    break
                
                last_validation_result = validation_result
                
                # If not the last attempt, wait with exponential backoff
                if attempt < max_attempts - 1:
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = min(max_delay, delay + jitter)
                    print(f"Validation failed, waiting {sleep_time:.1f} seconds before retry...")
                    time.sleep(sleep_time)
                    delay = min(max_delay, delay * 2)  # Exponential backoff with cap
                else:
                    # Instead, build comprehensive validation summary for HTML comments
                    validation_summary = {
                        'content_type': content_type,
                        'validation_failed': True,
                        'attempts': max_attempts,
                        'validation_result': validation_result,
                        'failure_patterns': failure_patterns,
                        'validation_temperature': min(VALIDATION_MAX_TEMPERATURE, VALIDATION_BASE_TEMPERATURE + (max_attempts - 1) * VALIDATION_TEMPERATURE_INCREMENT),

                        'last_validation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    if content_for_reedit:
                        validation_summary['reedit_available'] = True
                        validation_summary['reedit_message'] = content_for_reedit.get('validation_message', '')
                    
                    # Add to the list of validation summaries
                    self.validation_summaries.append(validation_summary)
                    # Also set as single validation_summary for backward compatibility
                    self.validation_summary = validation_summary
                    
                    print(f"WARN: All {max_attempts} content edit attempts failed for {content_type} in PR #{self.pr_number}")
                    if self.debug:
                        print(f"Validation result: {validation_result}")
                    print(f"Failure patterns: {failure_patterns}")
                    if content_for_reedit:
                        print(f"Content available for reedit with validation message: {content_for_reedit['validation_message']}")
                    # Do NOT revert to the original content; keep the last attempted edit
                    edited_content = current_edit
                    self.last_validation_result = validation_result
            
            # Set the content based on type
            if content_type == 'title':
                if not is_valid and attempt == max_attempts - 1:
                    orig = content.rstrip('.')
                    new = edited_content.rstrip('.')
                    ratio = SequenceMatcher(None, orig.lower(), new.lower()).ratio()
                    comment = "<!--- CHECK: Content may be substantially different from original --->\n"
                    if orig.lower() not in new.lower() and ratio < 0.7:
                        comment += f"<!--- CHECK: Title may be substantially different from original --->\n<!--- ORIGINAL: {orig} --->\n"
                    self.cleaned_title_comment = comment
                    self.cleaned_title = new
                else:
                    self.cleaned_title_comment = ""
                    self.cleaned_title = edited_content.rstrip('.')
                if self.debug:
                    print(f"DEBUG: [edit_content] Set cleaned_title: {self.cleaned_title}")

            elif content_type == 'notes':
                if not is_valid and attempt == max_attempts - 1:
                    comment = "<!--- CHECK: Content may be substantially different from original --->\n"
                    self.content_warning_comment = comment
                    self.edited_text = edited_content
                else:
                    self.content_warning_comment = ""
                    self.edited_text = edited_content
                if self.debug:
                    print(f"DEBUG: [edit_content] Set edited_text")
            
            if is_valid:
                self.validated = True
                
        except Exception as e:
            print(f"\nFailed to edit {content_type} with OpenAI: {str(e)}")
            if self.debug:
                print(f"\n{content}\n")
            # Use original content if editing fails
            if content_type == 'title':
                self.cleaned_title = content.rstrip('.')
            elif content_type == 'notes':
                self.edited_text = content

    def _edit_pass(self, content_type, content, pass_instructions, attr_name, edit=True):
        """Helper for a single editing pass with retry/validation, storing output in self.attr_name."""
        if self.debug:
            print(f"DEBUG: [edit_content] {attr_name} pass for PR #{self.pr_number} in {self.repo_name}")
        
        # Select model based on the editing pass
        if attr_name == "grouped_text":
            model = MODEL_PASS_1
            pass_name = "Pass 1 (Group and Flatten)"
        elif attr_name == "deduplicated_text":
            model = MODEL_PASS_2
            pass_name = "Pass 2 (Deduplicate)"
        elif attr_name == "edited_text":
            model = MODEL_PASS_3
            pass_name = "Pass 3 (Streamline)"
        else:
            model = MODEL_EDITING  # Fallback to default
            pass_name = "Single Pass"
        
        if self.debug:
            print(f"DEBUG: [_edit_pass] Using model {model} for {pass_name}")
        
        base_instructions = f"{pass_instructions}\n\n{EDIT_CONTENT_PROMPT}\n\nIMPORTANT: Maintain the scope of this specific PR (#{self.pr_number}). Do not merge content from other PRs or add information not present in the original content."
        max_attempts = DEFAULT_MAX_RETRIES
        last_validation_result = None
        failure_patterns = {}
        content_for_reedit = None
        
        for attempt in range(max_attempts):
            # Build dynamic instructions based on previous failures
            full_instructions = base_instructions
            if last_validation_result and attempt > 0:
                if 'formatting' in last_validation_result.lower():
                    failure_patterns['formatting'] = failure_patterns.get('formatting', 0) + 1
                if 'meaning' in last_validation_result.lower():
                    failure_patterns['meaning'] = failure_patterns.get('meaning', 0) + 1
                guidance = []
                if failure_patterns.get('formatting', 0) > 1:
                    guidance.append("Pay special attention to markdown formatting and structure.")
                if failure_patterns.get('meaning', 0) > 1:
                    guidance.append("Focus on preserving the exact meaning and technical details.")
                full_instructions += f"\n\nPrevious attempt failed: {last_validation_result}"
                if guidance:
                    full_instructions += f"\nPlease address these specific issues:\n" + "\n".join(f"- {g}" for g in guidance)
            
            # Determine content to edit
            if attempt > 0 and content_for_reedit and 'edited' in content_for_reedit:
                content_to_edit = content_for_reedit['edited']
            else:
                content_to_edit = content
            
            # Calculate adaptive temperature
            base_temp = BASE_TEMPERATURE
            if 'formatting' in failure_patterns:
                base_temp += TEMPERATURE_INCREMENT
            if 'meaning' in failure_patterns:
                base_temp -= (TEMPERATURE_INCREMENT / 2)
            temperature = min(0.7, base_temp + (attempt * (TEMPERATURE_INCREMENT / 2)))
            
            # Use appropriate parameters based on model
            api_params = get_model_api_params(model, MAX_TOKENS_EDITING, temperature)
            messages = [
                {"role": "system", "content": EDIT_CONTENT_SYSTEM},
                {"role": "user", "content": f"Instructions:\n{full_instructions}\n\nContent to edit:\n{content_to_edit}"}
            ]
            
            current_edit = unified_openai_call(
                messages=messages,
                api_params=api_params,
                debug=self.debug,
                function_name=f"_edit_pass_{attr_name}"
            )
            
            if current_edit is None:
                if self.debug:
                    print(f"DEBUG: [edit_content] {attr_name} pass attempt {attempt + 1} returned None")
                continue
            
            if self.debug:
                print(f"DEBUG: [edit_content] {attr_name} pass attempt {attempt + 1} content (first 100 chars): {repr(current_edit[:100])}")
            
            is_valid, validation_result, content_for_reedit = self.validate_edit(content_type, content, current_edit, edit)
            if self.debug:
                print(f"DEBUG: [edit_content] {attr_name} pass attempt {attempt + 1} validation result: {is_valid}")
            
            if is_valid:
                # Add successful validation summary for multi-pass editing
                validation_summary = {
                    'content_type': f"{content_type} ({attr_name})",
                    'validation_failed': False,
                    'attempts': attempt + 1,
                    'validation_result': validation_result,
                    'failure_patterns': failure_patterns,
                    'validation_temperature': min(VALIDATION_MAX_TEMPERATURE, VALIDATION_BASE_TEMPERATURE + attempt * VALIDATION_TEMPERATURE_INCREMENT),
                    'last_validation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                if not hasattr(self, 'validation_summaries'):
                    self.validation_summaries = []
                self.validation_summaries.append(validation_summary)
                
                setattr(self, attr_name, current_edit)
                return current_edit
            
            last_validation_result = validation_result
            
            # Handle critical failures with enhanced instructions
            critical_failures = ['duplicate content', 'begin with text summary', 'same feature', 'same functionality']
            is_critical_failure = any(failure in validation_result.lower() for failure in critical_failures)
            
            if is_critical_failure and attempt < max_attempts - 1:
                print(f"CRITICAL validation failure detected: {validation_result}")
                # Add critical failure guidance to base instructions for next attempt
                if 'duplicate' in validation_result.lower():
                    base_instructions += f"\n\nCRITICAL: Previous attempt failed due to duplicate content. You MUST consolidate all information about the same feature into a single paragraph. Do not repeat any concepts or explanations."
                elif 'text summary' in validation_result.lower():
                    base_instructions += f"\n\nCRITICAL: Previous attempt failed because content started with an image. You MUST begin with explanatory text before any images."
            
            if attempt < max_attempts - 1:
                print(f"Validation failed, retrying {attr_name} pass... (attempt {attempt + 2}/{max_attempts})")
        
        # All attempts failed - record failure and return last attempt
        validation_summary = {
            'content_type': f"{content_type} ({attr_name})",
            'validation_failed': True,
            'attempts': max_attempts,
            'validation_result': validation_result,
            'failure_patterns': failure_patterns,
            'validation_temperature': min(VALIDATION_MAX_TEMPERATURE, VALIDATION_BASE_TEMPERATURE + (max_attempts - 1) * VALIDATION_TEMPERATURE_INCREMENT),
            'last_validation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if content_for_reedit:
            validation_summary['reedit_available'] = True
            validation_summary['reedit_message'] = content_for_reedit.get('validation_message', '')
        
        if not hasattr(self, 'validation_summaries'):
            self.validation_summaries = []
        self.validation_summaries.append(validation_summary)
        
        print(f"WARN: All {max_attempts} content edit attempts failed for {attr_name} pass in PR #{self.pr_number}")
        if self.debug:
            print(f"Validation result: {validation_result}")
        print(f"Failure patterns: {failure_patterns}")
        if content_for_reedit:
            print(f"Content available for reedit with validation message: {content_for_reedit['validation_message']}")
        
        # Keep the last attempted edit
        setattr(self, attr_name, current_edit)
        return current_edit

    def validate_edit(self, content_type, original_content, edited_content, edit=False, attempt_number=1):
        """Uses LLM to validate edits by checking for common issues.
        
        Args:
            content_type (str): Type of content that was edited
            original_content (str): The original content before editing
            edited_content (str): The edited content to validate
            edit (bool): Whether to perform validation
            attempt_number (int): Which editing attempt this is (1-based, for adaptive temperature)
            
        Returns:
            tuple: (bool, str, dict): (is_valid, validation_message, content_for_reedit)
                - is_valid: True if the edit passes validation, False otherwise
                - validation_message: Detailed validation result message
                - content_for_reedit: Dict with content to reedit if validation fails
        """
        # Always perform basic validation
        validation_info = {
            'content_type': content_type,
            'original_length': len(original_content) if original_content else 0,
            'edited_length': len(edited_content) if edited_content else 0,
            'validation_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if not edit:
            print(f"Validation skipped for {content_type} edit in PR #{self.pr_number} (edit disabled)")
            return True, "PASS: No validation performed (edit disabled)", None

        # Cache key for validation results
        cache_key = f"{content_type}:{hash(original_content)}:{hash(edited_content)}"
        if hasattr(self, '_validation_cache') and cache_key in self._validation_cache:
            cached_result = self._validation_cache[cache_key]
            print(f"Using cached validation result for {content_type} edit in PR #{self.pr_number}: {cached_result[1]}")
            return cached_result

        # Print validation start message
        print(f"Validating {content_type} edit for PR #{self.pr_number} in {self.repo_name}...")

        if self.debug:
            print(f"DEBUG: [validate_edit] PR #{self.pr_number} - Validating {content_type}")
            print(f"DEBUG: [validate_edit] Original (first 100 chars): {repr(original_content[:100]) if original_content else None}")
            print(f"DEBUG: [validate_edit] Edited (first 100 chars): {repr(edited_content[:100]) if edited_content else None}")



        # Build detailed validation prompt
        validation_prompt = VALIDATION_PROMPT.format(content_type=content_type)
        if content_type in VALIDATION_CRITERIA:
            validation_prompt += "\n\nSpecific criteria to check:\n"
            for criterion in VALIDATION_CRITERIA[content_type]:
                validation_prompt += f"- {criterion}\n"

        # Calculate adaptive validation temperature based on editing attempt
        validation_temperature = min(
            VALIDATION_MAX_TEMPERATURE,
            VALIDATION_BASE_TEMPERATURE + (attempt_number - 1) * VALIDATION_TEMPERATURE_INCREMENT
        )
        
        # Use appropriate parameters based on model with adaptive temperature
        api_params = get_model_api_params(MODEL_VALIDATION, MAX_TOKENS_VALIDATION, validation_temperature)
        
        # Truncate content if too long to avoid token limits
        max_content_length = 8000  # Higher limit for gpt-4o
        truncated_original = original_content[:max_content_length] if original_content else ""
        truncated_edited = edited_content[:max_content_length] if edited_content else ""
        
        if len(original_content) > max_content_length or len(edited_content) > max_content_length:
            truncated_original += "... [truncated]"
            truncated_edited += "... [truncated]"
        
        messages = [
            {
                "role": "system",
                "content": VALIDATION_SYSTEM
            },
            {
                "role": "user",
                "content": f"Original content: {truncated_original}\n\nEdited content: {truncated_edited}"
            }
        ]
        
        result = unified_openai_call(
            messages=messages,
            api_params=api_params,
            debug=self.debug,
            function_name="validate_edit"
        )
        
        if result is None:
            print(f"\nAll validation attempts failed for {content_type} edit in PR #{self.pr_number}")
            return True, f"FAIL: All OpenAI validation attempts failed", None
        
        if self.debug:
            print(f"DEBUG: [validate_edit] Validation LLM response: {result}")
        
        # Handle empty or invalid responses
        if not result:
            result = "PASS"  # Default to pass if no response
            if self.debug:
                print(f"DEBUG: [validate_edit] Empty validation response, defaulting to PASS")
        
        # Clean up the result and handle edge cases
        result = result.strip()
        if not result or len(result) < 3:
            result = "PASS"  # Default to pass for very short responses
            if self.debug:
                print(f"DEBUG: [validate_edit] Very short validation response, defaulting to PASS")
        
        # Add validation result to info
        validation_info['validation_result'] = result
        validation_info['attempt_number'] = 1  # Unified call handles retries internally
        
        # Determine if validation passed - be more lenient with parsing
        result_upper = result.upper()
        is_valid = (result_upper.startswith('PASS') or 
                   (not result_upper.startswith('FAIL') and 
                    'PASS' in result_upper) or
                   (not result_upper.startswith('FAIL') and 
                    'FAIL' not in result_upper))
        
        # Print validation result with appropriate detail level
        if is_valid:
            if self.debug:
                print(f"Validation result for {content_type} edit in PR #{self.pr_number}: {result}")
            else:
                print(f"Validation PASS for {content_type} edit in PR #{self.pr_number}")
        else:
            print(f"Validation FAIL for {content_type} edit in PR #{self.pr_number}: {result}")
        
        # Prepare content for reedit if validation fails
        content_for_reedit = None
        if not is_valid:
            content_for_reedit = {
                'original': original_content,
                'edited': edited_content,
                'validation_message': result,
                'content_type': content_type
            }
        
        # Cache the result
        if not hasattr(self, '_validation_cache'):
            self._validation_cache = {}
        self._validation_cache[cache_key] = (is_valid, result, content_for_reedit)
        
        return is_valid, result, content_for_reedit





class ReleaseURL:
    def __init__(self, url):
        self.url = url
        self.repo_name = None # backend
        self.tag_name = None # v.2.0.23
        self.data_json = None
        self.prs = []

    def extract_repo_name(self):
        """Extracts and returns the repository name from the URL."""
        match = re.search(r"github\.com/(.+)/releases/tag/", self.url)
        if not match:
            print(f"ERROR: Invalid URL format '{self.url}'")
            return None
        return match.group(1)

    def set_repo_and_tag_name(self):
        """Sets the repo name (documentation/backend/...) and the release tag from the GitHub URL.

        Modifies:
            self.repo_name
            self.tag_name
        """
        match = re.search(r"github\.com/(.+)/releases/tag/(.+)$", self.url)
        if not match:
            print(f"ERROR: Invalid URL format '{self.url}'")
            return

        self.repo_name, self.tag_name = match.groups()

    def extract_prs(self):
        """Extracts PRs from the release URL.

        Modifies:
            self.prs
            self.data_json
        """
        print(f"Extracting PRs from {self.url} ...\n")
        cmd_release = ['gh', 'api', f'repos/{self.repo_name}/releases/tags/{self.tag_name}']
        result_release = subprocess.run(cmd_release, capture_output=True, text=True)
        output_release = result_release.stdout.strip()

        output_release_clean = ansi_escape.sub('', output_release) # to clean up notebook output

        try:
            self.data_json = json.loads(output_release_clean)
        except json.JSONDecodeError:
            print(f"ERROR: Unable to parse release data for URL '{self.url}'")      
        
        if 'body' in self.data_json:
            body = self.data_json['body']
            pr_numbers = re.findall(r"https://github.com/.+/pull/(\d+)", body)

            for pr_number in pr_numbers: # initialize PR objects using pr_numbers and add to list of PRs
                curr_PR = PR(self.repo_name, pr_number)
                self.prs.append(curr_PR)
                print(f"PR #{pr_number} retrieved.\n")

        else:
            print(f"ERROR: No body found in release data for URL '{self.url}'")

    def populate_pr_data(self):
        """Helper method. Calls JSON loader on each PR in a URL.
        """
        for pr in self.prs:
            pr.load_data_json()

def get_env_location():
    """
    Gets the location of the .env file, using the default if it exists.

    Returns:
        str: The path to the .env file.
    """
    # Default location of the .env file
    default_env_location = "./.env"
    
    # If default location exists, use it without any output
    if os.path.exists(default_env_location):
        return default_env_location
        
    # Only prompt and print if default location doesn't exist
    print(f"WARNING: .env file not found at {default_env_location}")
    env_location = input(
        f"Enter the location of your .env file (leave empty for default [{default_env_location}]): "
    ) or default_env_location

    print(f"Using .env file location: {env_location}\n")
    return env_location

def setup_openai_api(env_location):
    """
    Loads .env file from the specified location and retrieves the OpenAI API key.
    Also checks environment variables for the API key.

    Args:
        env_location (str): The location of the .env file.

    Raises:
        FileNotFoundError: If the .env file is not found at the specified location.
        KeyError: If the OPENAI_API_KEY is not present in the .env file or environment variables.
    """
    api_key = None
    
    # First check environment variables
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        print("✓ Found OpenAI API Key in environment variables\n")
        return api_key

    # If not in environment, try .env file
    try:
        config = dotenv_values(env_location)
        if config:
            api_key = config.get('OPENAI_API_KEY')
            if api_key:
                print(f"✓ Found OpenAI API Key in {env_location}\n")
                return api_key
    except Exception as e:
        print(f"Error reading .env file: {str(e)}")

    # If we get here, no API key was found
    print("ERROR: OPENAI_API_KEY not found in environment variables or .env file")
    sys.exit(1)

def display_list(array):
    """
    Lists an array in a numbered list. Used to check the `label_hierarchy`.
    """
    print("Label hierarchy:\n")
    for i, item in enumerate(array, start=1):
        print(f"{i}. {item}")

release_components = {} 

def parse_date(date_str):
    """Parse a date string into a datetime object, handling different formats.
    
    Args:
        date_str (str): Date string to parse
        
    Returns:
        datetime: Parsed datetime object or datetime.min if parsing fails
    """
    if date_str.lower() in ['n/a', 'tbd']:
        return datetime.datetime.min
        
    # Try different date formats
    formats = [
        "%B %d, %Y",  # April 14, 2025
        "%B %Y",      # November 2024
        "%Y-%m-%d",   # 2025-04-14
        "%m/%d/%Y"    # 04/14/2025
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue
            
    print(f"WARN: Could not parse date '{date_str}' - using minimum date")
    return datetime.datetime.min

def check_github_tag(repo, version):
    """Check if a GitHub tag exists for a given repository and version.
    
    Args:
        repo (str): Repository name (backend, frontend, agents, documentation)
        version (str): Release version
        
    Returns:
        bool: True if tag exists, False otherwise
    """
    try:
        # First check for git tags
        cmd = ['gh', 'api', f'repos/validmind/{repo}/git/refs/tags/{version}']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
            
        # If no git tag found, check releases
        cmd = ['gh', 'api', f'repos/validmind/{repo}/releases/tags/{version}']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
            
        # If still not found, list all tags for debugging
        cmd = ['gh', 'api', f'repos/validmind/{repo}/tags']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            tags = json.loads(result.stdout)
            cmvm_tags = [t['name'] for t in tags if t['name'].startswith('cmvm/')]
            if cmvm_tags:
                return False
            
        return False
            
    except Exception as e:
        print(f"  WARN: Error checking tag {version} in {repo}: {e}")
        return False

def get_github_tag_url(repo, version):
    """Construct GitHub tag URL for a given repository and version.
    
    Args:
        repo (str): Repository name (backend, frontend, agents, documentation)
        version (str): Release version
        
    Returns:
        str: GitHub tag URL
    """
    return f"https://github.com/validmind/{repo}/releases/tag/{version}"

def rate_limited_api_call(cmd, max_retries=DEFAULT_MAX_RETRIES, initial_delay=DEFAULT_INITIAL_DELAY):
    """Make a rate-limited API call with exponential backoff.
    
    Args:
        cmd (list): Command to run
        max_retries (int): Maximum number of retries
        initial_delay (int): Initial delay in seconds
    """
    delay = initial_delay
    for attempt in range(max_retries):
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # If successful or not a rate limit error, return immediately
        if result.returncode == 0 or "API rate limit exceeded" not in result.stderr:
            return result.returncode, result.stdout, result.stderr
            
        # If we hit rate limit, wait with exponential backoff
        if attempt < max_retries - 1:
            # Add jitter to prevent thundering herd
            jitter = random.uniform(*JITTER_RANGE) * delay
            sleep_time = min(DEFAULT_MAX_DELAY, delay + jitter)
            print(f"Rate limit hit, waiting {sleep_time:.1f} seconds before retry...")
            time.sleep(sleep_time)
            delay *= 2  # Exponential backoff
            
    return result.returncode, result.stdout, result.stderr

def get_all_cmvm_tags(repo, version=None, debug=False):
    """Get all tags from a repository using /git/refs/tags for reliability.
    
    Args:
        repo (str): Repository name (backend, frontend, agents, documentation)
        version (str, optional): Specific version to check for
        debug (bool): Whether to show debug output
        
    Returns:
        List[str]: List of tags found in the repository
    """
    try:
        # Get all refs/tags (may require paging for large repos)
        cmd = ['gh', 'api', f'repos/validmind/{repo}/git/refs/tags']
        returncode, stdout, stderr = rate_limited_api_call(cmd)
        
        if returncode == 0:
            tags = json.loads(stdout)
            # Handle both single tag and multiple tags cases
            if isinstance(tags, dict):
                tags = [tags]
            elif not isinstance(tags, list):
                tags = []
                
            tag_names = []
            for t in tags:
                ref = t.get('ref', '')
                if ref.startswith('refs/tags/'):
                    tag_name = ref.replace('refs/tags/', '')
                    # Only include tags that start with cmvm/
                    if tag_name.startswith('cmvm/'):
                        tag_names.append(tag_name)
            if debug:
                # Sort tags using version_key function
                sorted_tags = sorted(tag_names, key=version_key)
                print(f"DEBUG: [get_all_cmvm_tags] {repo} tag list sorted: {sorted_tags}")
            
            # Only check releases if we have tags
            if tag_names:
                # Check for releases in parallel with rate limiting
                def check_release(tag):
                    cmd = ['gh', 'api', f'repos/validmind/{repo}/releases/tags/{tag}']
                    returncode, stdout, stderr = rate_limited_api_call(cmd)
                    if returncode == 0:
                        if debug and (not version or tag == version):
                            print(f"DEBUG: [get_all_cmvm_tags] Release exists for {tag}")
                    return tag
                
                # Use ThreadPoolExecutor with limited concurrency
                max_workers = min(3, len(tag_names))  # Limit concurrent requests
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_tag = {executor.submit(check_release, tag): tag for tag in tag_names}
                    # Process results as they complete
                    for future in as_completed(future_to_tag):
                        try:
                            future.result()
                        except Exception as e:
                            if debug:
                                print(f"DEBUG: [get_all_cmvm_tags] Error checking release for tag {future_to_tag[future]}: {e}")
            elif debug:
                print(f"DEBUG: [get_all_cmvm_tags] No tags found in {repo}")
                
            return tag_names
    except Exception as e:
        if debug:
            print(f"DEBUG: [get_all_cmvm_tags] Error getting tags from {repo}: {e}")
        else:
            print(f"  WARN: Error getting tags from {repo}: {e}")
    return []

def is_merge_pr(title, debug=False):
    """Use OpenAI to classify whether a PR is an automatic merge PR.
    
    Args:
        title (str): The PR title to check
        debug (bool): Whether to show debug output
        
    Returns:
        bool: True if the PR appears to be an automatic merge
    """
    if not title:
        return False
        
    if debug:
        print(f"\nDEBUG: [is_merge_pr] Classifying PR title: {title}")
    
    prompt = MERGE_PR_CLASSIFICATION_PROMPT.format(title=title)
    api_params = get_model_api_params(MODEL_CLASSIFYING, MAX_TOKENS_CLASSIFICATION)
    messages = [
        {
            "role": "system",
            "content": MERGE_PR_CLASSIFICATION_SYSTEM
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    result = robust_openai_call(messages, api_params, debug, "is_merge_pr")
    
    if result:
        result_upper = result.upper()
        if debug:
            print(f"DEBUG: [is_merge_pr] OpenAI response: {result_upper}")
        return result_upper == 'MERGE'
    else:
        if debug:
            print("DEBUG: [is_merge_pr] All OpenAI attempts failed, falling back to keyword matching...")
            
        # Fallback to keyword matching if OpenAI fails
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in MERGE_KEYWORDS)

def get_pr_content(pr_number, repo, debug=False):
    """Get content from a PR's external release notes, PR summary, title, and image URLs.
    
    Args:
        pr_number (str): PR number
        repo (str): Repository name
        debug (bool): Whether to show debug output
        
    Returns:
        tuple: (external_notes, pr_summary, labels, title, pr_body, image_urls) - all may be None
    """
    try:
        cmd = ['gh', 'pr', 'view', pr_number, '--json', 'body,labels,comments,title', '--repo', f'validmind/{repo}']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if debug:
                print(f"DEBUG: Failed to fetch PR #{pr_number} in {repo}")
            return None, None, [], None, None, []
        pr_data = json.loads(result.stdout)
        if debug:
            print(f"DEBUG: [get_pr_content] PR #{pr_number} in {repo} - Title: {pr_data.get('title')}")
            pr_url = f"https://github.com/validmind/{repo}/pull/{pr_number}"
            print(f"DEBUG: [get_pr_content] PR #{pr_number} in {repo} - URL: {pr_url}")
            print(f"DEBUG: [get_pr_content] PR #{pr_number} in {repo} - Labels: {[label['name'] for label in pr_data.get('labels', [])]}")
        
        # Get the title and check if it's a merge PR
        title = pr_data.get('title')
        if is_merge_pr(title):
            if debug:
                print(f"DEBUG: [get_pr_content] PR #{pr_number} in {repo} appears to be an automatic merge, skipping.")
            return None, None, [], title, None, []
            
        # Always build a list of label names
        labels = [label['name'] for label in pr_data.get('labels', [])]
        # Skip PRs with excluded labels but return the title
        if any(label in EXCLUDED_LABELS for label in labels):
            if debug:
                print(f"DEBUG: [get_pr_content] PR #{pr_number} in {repo} has excluded label, skipping.")
            return None, None, labels, title, None, []
        # Skip internal PRs but return the title
        if 'internal' in labels:
            if debug:
                print(f"DEBUG: [get_pr_content] PR #{pr_number} in {repo} is internal, skipping.")
            return None, None, ['internal'], pr_data.get('title'), None, []

        # Extract external release notes using LLM-based content extraction
        external_notes = None
        if pr_data.get('body'):
            if debug:
                print("\nDEBUG: [get_pr_content] Extracting external release notes using LLM")
                print(f"DEBUG: [get_pr_content] PR body length: {len(pr_data['body'])}")
            
            # Use LLM to extract only relevant content
            external_notes = extract_relevant_content(pr_data['body'], debug)
            
            if debug:
                if external_notes:
                    print(f"DEBUG: [get_pr_content] LLM extracted {len(external_notes)} characters of relevant content")
                else:
                    print("DEBUG: [get_pr_content] LLM found no relevant content")

        # Extract PR summary using LLM-based extraction
        pr_summary = None
        comments = pr_data.get('comments', [])
        if debug:
            print(f"DEBUG: [get_pr_content] PR #{pr_number} in {repo} - Number of comments: {len(comments)}")
        
        if comments:
            pr_summary = extract_pr_summary(comments, debug)
            if debug:
                if pr_summary:
                    print(f"DEBUG: [get_pr_content] PR #{pr_number} in {repo} - LLM extracted PR summary")
                else:
                    print(f"DEBUG: [get_pr_content] PR #{pr_number} in {repo} - LLM found no PR summary")
        title = pr_data.get('title')
        pr_body = pr_data.get('body')
        # Extract image URLs from PR body and comments
        image_urls = []
        video_urls = []
        if pr_body:
            # Markdown images
            image_urls += re.findall(r'!\[[^\]]*\]\((https?://[^)]+)\)', pr_body)
            # HTML <img> tags
            image_urls += re.findall(r'<img [^>]*src="([^"]+)"', pr_body)
            # Loom videos (https://www.loom.com/share/...) and embed links
            video_urls += re.findall(r'https://www\.loom\.com/(?:share|embed)/[a-zA-Z0-9\-]+', pr_body)
        for comment in comments:
            cbody = comment.get('body', '')
            # Markdown images
            image_urls += re.findall(r'!\[[^\]]*\]\((https?://[^)]+)\)', cbody)
            # HTML <img> tags
            image_urls += re.findall(r'<img [^>]*src="([^"]+)"', cbody)
            # Loom videos
            video_urls += re.findall(r'https://www\.loom\.com/(?:share|embed)/[a-zA-Z0-9\-]+', cbody)
        if debug:
            print(f"DEBUG: [get_pr_content] PR #{pr_number} in {repo} - Image URLs: {image_urls}")
            print(f"DEBUG: [get_pr_content] PR #{pr_number} in {repo} - Loom video URLs: {video_urls}")
        return external_notes, pr_summary, labels, title, pr_body, image_urls
    except Exception as e:
        if debug:
            print(f"  WARN: Error getting PR content for #{pr_number} in {repo}: {e}")
        return None, None, [], None, None, []

def version_key(tag):
    # Remove prefix
    tag = tag.replace('cmvm/', '')
    # Split RC if present
    if '-rc' in tag:
        base, rc = tag.split('-rc')
        is_rc = 1
        rc_number = int(rc) if rc.isdigit() else 0
    else:
        base = tag
        is_rc = 0
        rc_number = 0
    # Split base into major, minor, patch (pad as needed)
    parts = base.split('.')
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    # RCs sort before their base version (so is_rc=1 sorts before is_rc=0)
    return (major, minor, patch, -is_rc, rc_number)

def get_previous_tag(repo, tag, debug=False):
    """Find the previous tag for a given repo and current tag, using CMVM versioning rules.
    Uses /git/refs/tags as primary, /tags as fallback.
    
    For RC tags (e.g., 25.05-rc1):
    1. First looks for previous RC of same version (e.g., edit_ RCs)
    2. Then tries previous version's final release
    3. Finally tries previous version's RCs in reverse order
    
    For regular releases:
    1. First tries previous regular release
    2. Then tries RCs of current version in reverse order
    """
    tag_names = set()
    # Try /git/refs/tags first
    cmd = ['gh', 'api', f'repos/validmind/{repo}/git/refs/tags']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        tags = json.loads(result.stdout)
        if isinstance(tags, dict):
            tags = [tags]
        for t in tags:
            ref = t.get('ref', '')
            if ref.startswith('refs/tags/'):
                tag_names.add(ref.replace('refs/tags/', ''))
    # Fallback to /tags if no tags found
    if not tag_names:
        cmd = ['gh', 'api', f'repos/validmind/{repo}/tags']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            tags = json.loads(result.stdout)
            for t in tags:
                name = t.get('name', '')
                tag_names.add(name)

    # Extract version info from current tag
    # Handle both cmvm/ and v prefixes
    if tag.startswith('cmvm/'):
        match = re.match(r'cmvm/(\d+)\.(\d+)(?:\.(\d+))?(?:-rc(\d+))?$', tag)
    else:
        match = re.match(r'v(\d+)\.(\d+)(?:\.(\d+))?(?:-rc(\d+))?$', tag)
        
    if not match:
        if debug:
            print(f"DEBUG: [get_previous_tag] {repo} invalid tag format: {tag}")
        return None
    
    major, minor, patch, rc_num = match.groups()
    major, minor = int(major), int(minor)
    patch = int(patch) if patch else None
    is_rc = rc_num is not None
    rc_num = int(rc_num) if rc_num else None
    
    # Filter and sort tags
    valid_tags = []
    for tag_name in tag_names:
        # Handle both cmvm/ and v prefixes
        if tag_name.startswith('cmvm/'):
            match = re.match(r'cmvm/(\d+)\.(\d+)(?:\.(\d+))?(?:-rc(\d+))?$', tag_name)
        else:
            match = re.match(r'v(\d+)\.(\d+)(?:\.(\d+))?(?:-rc(\d+))?$', tag_name)
            
        if not match:
            continue
        tag_major = int(match.group(1))
        tag_minor = int(match.group(2))
        tag_patch = int(match.group(3)) if match.group(3) else None
        tag_rc = int(match.group(4)) if match.group(4) else None
        valid_tags.append({
            'name': tag_name,
            'major': tag_major,
            'minor': tag_minor,
            'patch': tag_patch,
            'rc': tag_rc
        })
    
    # Sort tags by version and RC number (RCs come after their base version)
    valid_tags.sort(key=lambda x: (x['major'], x['minor'], x['patch'] if x['patch'] is not None else 0, float('inf') if x['rc'] is None else x['rc']))
    
    if debug:
        print(f"\nDEBUG: {repo} tag list sorted: {[t['name'] for t in valid_tags]}")
    
    # Find current tag's index
    current_idx = None
    for i, t in enumerate(valid_tags):
        if (t['major'] == major and t['minor'] == minor and 
            ((is_rc and t['rc'] == rc_num) or (not is_rc and t['rc'] is None and t['patch'] == patch))):
            current_idx = i
            break
    
    if current_idx is None:
        if debug:
            print(f"DEBUG: [get_previous_tag] {repo} current_tag {tag} not in tag list.")
        return None
    
    if current_idx == 0:
        if debug:
            print(f"DEBUG: [get_previous_tag] {repo} current_tag {tag} is the first tag.")
        return None
    
    # For RC tags
    if is_rc:
        # First try previous RC of same version
        for t in reversed(valid_tags[:current_idx]):
            if t['major'] == major and t['minor'] == minor and t['rc'] is not None:
                if debug:
                    print(f"DEBUG: [get_previous_tag] {repo} previous RC tag for {tag}: {t['name']}")
                return t['name']
        
        # Then try previous version's final release
        for t in reversed(valid_tags[:current_idx]):
            if t['rc'] is None:
                if debug:
                    print(f"DEBUG: [get_previous_tag] {repo} previous release for {tag}: {t['name']}")
                return t['name']
        
        # Finally try previous version's RCs
        if valid_tags[:current_idx]:
            prev_tag = valid_tags[current_idx - 1]['name']
            if debug:
                print(f"DEBUG: [get_previous_tag] {repo} previous version RC for {tag}: {prev_tag}")
            return prev_tag
    
    # For regular releases
    else:
        # For major.minor releases, look for previous major.minor release (same format)
        if patch is None:
            # Look for previous regular release with major.minor format (no patch)
            for t in reversed(valid_tags[:current_idx]):
                if t['rc'] is None and t['patch'] is None:
                    if debug:
                        print(f"DEBUG: [get_previous_tag] {repo} previous major.minor release for {tag}: {t['name']}")
                    return t['name']
        # For hotfix releases (major.minor.patch), look for previous patch in same version
        else:
            # Look for previous patch in same major.minor version
            for t in reversed(valid_tags[:current_idx]):
                if (t['major'] == major and t['minor'] == minor and 
                    t['rc'] is None and t['patch'] is not None):
                    if debug:
                        print(f"DEBUG: [get_previous_tag] {repo} previous patch in same version for {tag}: {t['name']}")
                    return t['name']
            # Fallback to base major.minor version
            for t in reversed(valid_tags[:current_idx]):
                if (t['major'] == major and t['minor'] == minor and 
                    t['rc'] is None and t['patch'] is None):
                    if debug:
                        print(f"DEBUG: [get_previous_tag] {repo} base version for hotfix {tag}: {t['name']}")
                    return t['name']
    
    # If no appropriate previous tag found, this is an error
    print(f"ERROR: [get_previous_tag] Could not find appropriate previous tag for {tag} in {repo}")
    return None

def get_commits_for_tag(repo, tag, debug=False):
    """Get all PRs merged between the previous tag and this tag, excluding internal PRs.
    If the tag doesn't exist in the repository, return empty list.
    If no previous tag is found, fall back to extracting PRs from the release or tag body.
    """
    try:
        # First check if the current tag exists in this repository
        current_tag_with_prefix = f"cmvm/{tag}" if not tag.startswith('cmvm/') else tag
        cmd = ['gh', 'api', f'repos/validmind/{repo}/git/refs/tags/{current_tag_with_prefix}']
        if debug:
            print(f"DEBUG: [get_commits_for_tag] Checking if tag {current_tag_with_prefix} exists in {repo}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if debug:
                print(f"DEBUG: [get_commits_for_tag] Tag {current_tag_with_prefix} not found in {repo}, skipping")
            return []
        
        prev_tag = get_previous_tag(repo, tag, debug=debug)
        if prev_tag is None:
            print(f"ERROR: [get_commits_for_tag] Failed to find previous tag for {tag} in {repo}")
            sys.exit(1)
        if not prev_tag:
            if debug:
                print(f"DEBUG: [get_commits_for_tag] No previous tag found for {tag} in {repo}, using tag/release body as fallback.")
            # Try to fetch the release body first
            cmd = ['gh', 'api', f'repos/validmind/{repo}/releases/tags/{tag}']
            if debug:
                print(f"DEBUG: [get_commits_for_tag] Release API call: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if debug:
                print(f"DEBUG: [get_commits_for_tag] Release API call for {tag} returned code: {result.returncode}")
                if result.returncode != 0:
                    print(f"DEBUG: [get_commits_for_tag] Release API call error: {result.stderr}")
            body = ''
            if result.returncode == 0:
                try:
                    release_data = json.loads(result.stdout)
                    body = release_data.get('body', '')
                    if debug:
                        print(f"DEBUG: [get_commits_for_tag] Release body length: {len(body)}")
                except Exception as e:
                    if debug:
                        print(f"DEBUG: [get_commits_for_tag] Could not parse release body for {tag} in {repo}: {e}")
            if not body:
                # If no release, try to fetch the annotated tag message
                cmd = ['gh', 'api', f'repos/validmind/{repo}/git/tags/{tag}']
                if debug:
                    print(f"DEBUG: [get_commits_for_tag] Tag API call: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if debug:
                    print(f"DEBUG: [get_commits_for_tag] Tag API call for {tag} returned code: {result.returncode}")
                    if result.returncode != 0:
                        print(f"DEBUG: [get_commits_for_tag] Tag API call error: {result.stderr}")
                if result.returncode == 0:
                    try:
                        tag_data = json.loads(result.stdout)
                        body = tag_data.get('message', '')
                        if debug:
                            print(f"DEBUG: [get_commits_for_tag] Tag message length: {len(body)}")
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: [get_commits_for_tag] Could not parse tag message for {tag} in {repo}: {e}")
            pr_numbers = re.findall(r"https://github.com/validmind/.+/pull/(\d+)", body)
            if debug:
                print(f"DEBUG: [get_commits_for_tag] Found {len(pr_numbers)} PR numbers in body")
            
            # Parallelize PR content fetching
            def fetch_pr_content(pr_number):
                return get_pr_content(pr_number, repo, debug=debug)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_pr = {executor.submit(fetch_pr_content, pr_number): pr_number for pr_number in pr_numbers}
                commits = []
                for future in concurrent.futures.as_completed(future_to_pr):
                    pr_number = future_to_pr[future]
                    try:
                        external_notes, pr_summary, labels, title, pr_body, image_urls = future.result()
                        # Skip merge PRs: labels is empty and both external_notes and pr_summary are None
                        if (not labels and external_notes is None and pr_summary is None):
                            if debug:
                                print(f"DEBUG: [get_commits_for_tag] Skipping merge PR #{pr_number} in {repo}")
                            continue
                        if 'internal' not in labels:
                            commits.append({
                                'pr_number': pr_number,
                                'external_notes': external_notes,
                                'pr_summary': pr_summary,
                                'labels': labels,
                                'title': title,
                                'pr_body': pr_body,
                                'image_urls': image_urls
                            })
                    except Exception as e:
                        if debug:
                            print(f"DEBUG: [fetch_pr_content] Error processing PR #{pr_number}: {e}")
            return commits
            
        # If we have a previous tag, use the commit-range logic
        # Get SHA for previous tag
        prev_tag_with_prefix = f"cmvm/{prev_tag}" if not prev_tag.startswith('cmvm/') else prev_tag
        cmd = ['gh', 'api', f'repos/validmind/{repo}/git/refs/tags/{prev_tag_with_prefix}']
        if debug:
            print(f"DEBUG: [get_commits_for_tag] Previous tag SHA API call: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if debug:
            print(f"DEBUG: [get_commits_for_tag] Previous tag SHA API call for {prev_tag_with_prefix} returned code: {result.returncode}")
            if result.returncode != 0:
                print(f"DEBUG: [get_commits_for_tag] Previous tag SHA API call error: {result.stderr}")
        if result.returncode != 0:
            if debug:
                print(f"DEBUG: [get_commits_for_tag] Could not get SHA for previous tag {prev_tag_with_prefix}")
            return []
        tag_data = json.loads(result.stdout)
        base_sha = tag_data['object']['sha']
        if debug:
            print(f"DEBUG: [get_commits_for_tag] Previous tag {prev_tag_with_prefix} SHA: {base_sha}")
        # Get SHA for current tag
        current_tag_with_prefix = f"cmvm/{tag}" if not tag.startswith('cmvm/') else tag
        cmd = ['gh', 'api', f'repos/validmind/{repo}/git/refs/tags/{current_tag_with_prefix}']
        if debug:
            print(f"DEBUG: [get_commits_for_tag] Current tag SHA API call: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if debug:
            print(f"DEBUG: [get_commits_for_tag] Current tag SHA API call for {current_tag_with_prefix} returned code: {result.returncode}")
            if result.returncode != 0:
                print(f"DEBUG: [get_commits_for_tag] Current tag SHA API call error: {result.stderr}")
        if result.returncode != 0:
            if debug:
                print(f"DEBUG: [get_commits_for_tag] Could not get SHA for current tag {current_tag_with_prefix}")
            return []
        tag_data = json.loads(result.stdout)
        head_sha = tag_data['object']['sha']
        if debug:
            print(f"DEBUG: [get_commits_for_tag] Current tag {current_tag_with_prefix} SHA: {head_sha}")
        # Get all commits in the range (exclusive of base, inclusive of head)
        cmd = ['gh', 'api', f'repos/validmind/{repo}/compare/{base_sha}...{head_sha}']
        if debug:
            print(f"DEBUG: [get_commits_for_tag] Compare API call: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if debug:
            print(f"DEBUG: [get_commits_for_tag] Compare API call returned code: {result.returncode}")
            if result.returncode != 0:
                print(f"DEBUG: [get_commits_for_tag] Compare API call error: {result.stderr}")
        if result.returncode != 0:
            if debug:
                print(f"DEBUG: [get_commits_for_tag] Could not get commits between {base_sha} and {head_sha}")
            return []
        compare_data = json.loads(result.stdout)
        commit_shas = [c['sha'] for c in compare_data.get('commits', [])]
        if debug:
            print(f"DEBUG: [get_commits_for_tag] Found {len(commit_shas)} commits between {prev_tag_with_prefix} and {current_tag_with_prefix}")
            if commit_shas:
                print(f"DEBUG: [get_commits_for_tag] First commit SHA: {commit_shas[0]}")
                print(f"DEBUG: [get_commits_for_tag] Last commit SHA: {commit_shas[-1]}")
        
        # Parallelize PR lookup for commits
        def fetch_prs_for_commit(sha):
            cmd = ['gh', 'api', '-H', 'Accept: application/vnd.github.groot-preview+json', f'repos/validmind/{repo}/commits/{sha}/pulls']
            # In subprocess, pass header as a single string to match shell quoting
            cmd = ['gh', 'api', '-H', 'Accept: application/vnd.github.groot-preview+json', f'repos/validmind/{repo}/commits/{sha}/pulls']
            if debug:
                print(f"DEBUG: [fetch_prs_for_commit] PR lookup API call: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if debug and result.returncode != 0:
                print(f"DEBUG: [fetch_prs_for_commit] Could not get PRs for commit {sha}, code: {result.returncode}")
                print(f"DEBUG: [fetch_prs_for_commit] PR lookup API call error: {result.stderr}")
            if result.returncode != 0:
                return []
            pulls = json.loads(result.stdout)
            return [(str(pr['number']), pr) for pr in pulls]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_sha = {executor.submit(fetch_prs_for_commit, sha): sha for sha in commit_shas}
            pr_numbers = set()
            pr_info = {}
            for future in concurrent.futures.as_completed(future_to_sha):
                sha = future_to_sha[future]
                try:
                    prs = future.result()
                    for pr_number, pr in prs:
                        if pr_number not in pr_numbers:
                            pr_numbers.add(pr_number)
                            pr_info[pr_number] = pr
                except Exception as e:
                    if debug:
                        print(f"DEBUG: [fetch_prs_for_commit] Error processing commit {sha}: {e}")
        
        if debug:
            print(f"DEBUG: Found {len(pr_numbers)} PRs in commits")
            if pr_numbers:
                print(f"DEBUG: PR numbers found: {sorted(pr_numbers)}")
        
        # Parallelize PR content fetching
        def fetch_pr_content(pr_number):
            return get_pr_content(pr_number, repo, debug=debug)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_pr = {executor.submit(fetch_pr_content, pr_number): pr_number for pr_number in pr_numbers}
            commits = []
            for future in concurrent.futures.as_completed(future_to_pr):
                pr_number = future_to_pr[future]
                try:
                    external_notes, pr_summary, labels, title, pr_body, image_urls = future.result()
                    # Skip merge PRs (identified by empty labels and no content)
                    if not labels and external_notes is None and pr_summary is None:
                        if debug:
                            print(f"DEBUG: [get_commits_for_tag] Skipping merge PR #{pr_number} in {repo}")
                        continue
                    if 'internal' not in labels:
                        commits.append({
                            'pr_number': pr_number,
                            'external_notes': external_notes,
                            'pr_summary': pr_summary,
                            'labels': labels,
                            'title': title,
                            'pr_body': pr_body,
                            'image_urls': image_urls
                        })
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Error processing PR #{pr_number}: {e}")
        
        if debug:
            print(f"DEBUG: Processed {len(commits)} PRs for {current_tag_with_prefix}")
            if commits:
                print(f"DEBUG: PR numbers processed: {[c['pr_number'] for c in commits]}")
        return commits
    except Exception as e:
        if debug:
            print(f"DEBUG: Error getting commits for tag {tag} in {repo}: {e}")
        return []

def generate_changelog_content(repo, tag, commits, has_release, download_assets=True):
    """Generate changelog content for a repository and tag.

    Args:
        repo (str): Repository name
        tag (str): Tag name
        commits (List[dict]): List of commit information
        has_release (bool): Whether a release exists for this tag

    Returns:
        str: Formatted changelog content
    """
    repo_name = repo.capitalize()
    tag_url = f"https://github.com/validmind/{repo}/releases/tag/{tag}"
    
    # If no commits at all, the tag likely doesn't exist in this repo
    if not commits:
        content = f"<!--- # {repo_name} --->\n"
        content += f"<!--- Tag {tag} not found in {repo} repository --->\n"
        content += "<!-- No tag found in this repository -->\n"
        return content
    
    # Get the commit SHA for the tag
    try:
        cmd = ['gh', 'api', f'repos/validmind/{repo}/git/refs/tags/{tag}']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            tag_data = json.loads(result.stdout)
            sha = tag_data['object']['sha']
            compare_url = f"Compare API call: gh api repos/validmind/{repo}/compare/{sha}...{sha}"
        else:
            # Fallback to tag-based URL if SHA lookup fails
            compare_url = f"Compare API call: gh api repos/validmind/{repo}/compare/{tag}...{tag}"
    except Exception:
        # Fallback to tag-based URL if any error occurs
        compare_url = f"Compare API call: gh api repos/validmind/{repo}/compare/{tag}...{tag}"
    
    # Filter out internal PRs (those with 'internal' in their labels)
    public_commits = [c for c in commits if not (c.get('labels') and 'internal' in c['labels'])]
    
    if not public_commits:
        # Comment out the heading and tag/release info if no public PRs
        content = f"<!--- # {repo_name} --->\n"
        if has_release:
            content += f"<!--- Release: [{tag}]({tag_url}) --->\n"
            content += f"<!--- {compare_url} --->\n"
        else:
            content += f"<!--- Tag: [{tag}]({tag_url}) --->\n"
            content += f"<!--- {compare_url} --->\n"
        content += "<!-- No public PRs found for this release -->\n"
        return content  # <-- EARLY RETURN, nothing else is executed!
    
    # If there are public PRs, proceed as before but defer the heading decision
    labeled_commits = defaultdict(list)
    for commit in public_commits:
        if commit['labels']:
            for label in commit['labels']:
                if label in label_to_category:
                    labeled_commits[label].append(commit)
                    break
            else:
                labeled_commits['other'].append(commit)
        else:
            labeled_commits['other'].append(commit)
    
    # Generate all PR content and track if any has actual user-facing content
    pr_sections = []
    has_any_real_content = False
    
    for label in label_hierarchy + ['other']:
        if label in labeled_commits and labeled_commits[label]:
            section_content = ""
            # Skip Documentation heading if repository is documentation
            if label == 'documentation' and repo == 'documentation':
                section_content += "\n"
            else:
                section_content += f"{label_to_category.get(label, '<!-- ### Changes with no label -->')}\n\n"
            
            for commit in labeled_commits[label]:
                pr_url = f"https://github.com/validmind/{repo}/pull/{commit['pr_number']}"
                pr_number = commit['pr_number']
                title = commit.get('cleaned_title') or commit.get('title') or f"PR #{pr_number}"
                has_content = bool(commit.get('external_notes') or commit.get('pr_summary') or commit.get('pr_body'))
                is_merge = commit.get('is_merge_pr', False)
                
                if not has_content or is_merge:
                    # Comment out the entire PR section if no content or if it's a merge PR
                    section_content += f"\n<!--- PR #{pr_number}: {pr_url} --->\n"
                    section_content += f"<!--- Labels: {', '.join(commit['labels']) if commit['labels'] else 'none'} --->\n"
                    section_content += f"<!--- ### {title} (#{pr_number}) --->\n"
                    # Check if this is a merge PR and add appropriate comment
                    if is_merge:
                        section_content += f"<!-- Merge PR - not included in release notes. -->\n\n"
                    else:
                        section_content += f"<!-- No release notes or summary provided. -->\n\n"
                    continue
                
                # This PR has real content
                has_any_real_content = True
                section_content += f"\n<!--- PR #{pr_number}: {pr_url} --->\n"
                section_content += f"<!--- Labels: {', '.join(commit['labels']) if commit['labels'] else 'none'} --->\n"
                section_content += f"### {title} (#{pr_number})\n\n"
                # Embed images as markdown using local paths
                for url in commit.get('image_urls', []):
                    local_path = download_image(url, tag, debug=False, download_assets=download_assets)
                    if local_path:
                        rel_path = os.path.relpath(local_path, os.getcwd())
                        section_content += f'![Image]({rel_path})\n'
                if commit['pr_summary']:
                    section_content += f"{update_image_links(commit['pr_summary'], tag, False, download_assets)}\n\n"
                if commit['external_notes']:
                    section_content += f"{update_image_links(commit['external_notes'], tag, False, download_assets)}\n\n"
            
            pr_sections.append(section_content)
    
    # Now decide on the repo heading based on whether we found real content
    if not has_any_real_content:
        # Comment out the entire section if no real content
        content = f"<!--- # {repo_name} --->\n"
        if has_release:
            content += f"<!--- Release: [{tag}]({tag_url}) --->\n"
            content += f"<!--- {compare_url} --->\n"
        else:
            content += f"<!--- Tag: [{tag}]({tag_url}) --->\n"
            content += f"<!--- {compare_url} --->\n"
        content += "<!-- No public PRs found for this release -->\n"
        return content
    else:
        # Use normal heading
        content = f"# {repo_name}\n"
        if has_release:
            content += f"<!--- Release: [{tag}]({tag_url}) --->\n"
            content += f"<!--- {compare_url} --->\n\n"
        else:
            content += f"<!--- Tag: [{tag}]({tag_url}) --->\n"
            content += f"<!--- {compare_url} --->\n\n"
        content += "".join(pr_sections)
    
    return content

def check_github_release(repo, version):
    """Check if a GitHub release exists for a given repository and version.
    
    Args:
        repo (str): Repository name (backend, frontend, agents, documentation)
        version (str): Release version
        
    Returns:
        bool: True if release exists, False otherwise
    """
    try:
        cmd = ['gh', 'api', f'repos/validmind/{repo}/releases/tags/{version}']
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"  WARN: Error checking release {version} in {repo}: {e}")
        return False

def create_release_file(release, overwrite=False, debug=False, edit=False, single=False, internal=False, download_assets=True):
    """Create release note files for a specific version: one per PR, and a new release-notes.qmd, or legacy single file if single=True."""
    import glob
    version = release['version']
    date = release['date']
    version_str = version.split('/')[-1]
    if 'cmvm' in version.lower():
        base_dir = os.path.join(RELEASES_DIR, 'cmvm', version_str)
    else:
        base_dir = os.path.join(RELEASES_DIR, version_str)
    os.makedirs(base_dir, exist_ok=True)
    # Always use the legacy single release-notes.qmd output logic
    output_dir = base_dir
    file_name = "release-notes.qmd"
    file_path = os.path.join(output_dir, file_name)
    if os.path.exists(file_path) and not overwrite:
        print(f"File {file_path} already exists. Use --overwrite to update it.")
        return
    content = []
    repo_contents = []
    all_commits = []
    any_validated = False
    any_edited = False
    def process_pr(commit, repo):
        # Skip internal PRs
        if 'internal' in commit.get('labels', []):
            return False, False
        # Mark merge PRs but don't skip them - they'll be added as comments
        if is_merge_pr(commit.get('title', '')):
            if debug:
                print(f"DEBUG: [process_pr] Marking merge PR #{commit.get('pr_number')} in {repo} for comment inclusion")
            commit['is_merge_pr'] = True
        else:
            commit['is_merge_pr'] = False
        pr_obj = PR(repo_name=repo, pr_number=commit['pr_number'], title=commit.get('title'), body=commit.get('pr_body'), debug=debug)
        validated = False
        edited = False
        if edit:
            # Store all validation summaries
            validation_summaries = []
            
            # PR summary is already processed by LLM extraction, no additional editing needed
            if commit.get('pr_summary'):
                pr_obj.pr_interpreted_summary = commit['pr_summary']
            if commit.get('external_notes'):
                pr_obj.edit_content('notes', commit['external_notes'], EDIT_CONTENT_PROMPT, edit=True)
                commit['external_notes'] = pr_obj.edited_text
                if hasattr(pr_obj, 'validation_summaries'):
                    validation_summaries.extend(pr_obj.validation_summaries)
                elif hasattr(pr_obj, 'validation_summary'):
                    validation_summaries.append(pr_obj.validation_summary)
                # Set edited=True if editing was attempted, regardless of validation status
                edited = True
                if pr_obj.validated:
                    validated = True
            context = ''
            if commit.get('pr_summary'):
                context += f"\nPR Summary: {commit['pr_summary']}"
            if commit.get('external_notes'):
                context += f"\nExternal Notes: {commit['external_notes']}"
            title_prompt = EDIT_TITLE_PROMPT.format(title=commit.get('title', ''), body=context)
            pr_obj.edit_content('title', commit.get('title', ''), title_prompt, edit=True)
            commit['cleaned_title'] = pr_obj.cleaned_title
            if hasattr(pr_obj, 'validation_summaries'):
                validation_summaries.extend(pr_obj.validation_summaries)
            elif hasattr(pr_obj, 'validation_summary'):
                validation_summaries.append(pr_obj.validation_summary)
            # Set edited=True if editing was attempted, regardless of validation status
            edited = True
            if pr_obj.validated:
                validated = True
            
            # Store all validation summaries if any exist
            if validation_summaries:
                commit['validation_summaries'] = validation_summaries
        else:
            commit['cleaned_title'] = commit.get('title', '')
            commit['pr_summary'] = commit.get('pr_summary', '')
            commit['external_notes'] = commit.get('external_notes', '')

        if commit.get('pr_body'):
            commit['pr_body'] = update_image_links(commit['pr_body'], version, debug, download_assets)
        if commit.get('external_notes'):
            commit['external_notes'] = update_image_links(commit['external_notes'], version, debug, download_assets)
        if commit.get('pr_body'):
            commit['pr_body'] = update_image_links(commit['pr_body'], version, debug, download_assets)
        # Store validation and editing status in the commit object
        commit['validated'] = validated
        commit['edited'] = edited
        
        # Also transfer final validation summary to commit if available (fallback for single validations)
        if hasattr(pr_obj, 'validation_summary') and 'validation_summaries' not in commit:
            commit['validation_summary'] = pr_obj.validation_summary
            
        return validated, edited
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for repo in REPOS:
            commits = get_commits_for_tag(repo, version, debug)
            future_to_commit = {executor.submit(process_pr, commit, repo): commit for commit in commits}
            for future in concurrent.futures.as_completed(future_to_commit):
                validated, edited = future.result()
                if validated:
                    any_validated = True
                if edited:
                    any_edited = True
            # Ensure each commit has a 'repo' key
            for commit in commits:
                commit['repo'] = repo
            all_commits.extend(commits)
            has_release = check_github_release(repo, version)
            repo_content = generate_changelog_content(repo, version, commits, has_release, download_assets)
            repo_contents.append(repo_content)
            if repo_content:
                filtered_content = []
                for line in repo_content.split('\n'):
                    if line.strip().startswith('<!---') and 'editor_note' in line:
                        continue
                    if line.strip().startswith('<!--') and 'editor_note' in line:
                        continue
                    filtered_content.append(line)
                content.append('\n'.join(filtered_content))
    
    # --- Add high-level summary logic here ---
    def has_user_facing_content(commit):
        return bool(
            (commit.get('external_notes') and commit['external_notes'].strip()) or
            (commit.get('pr_summary') and commit['pr_summary'].strip())
        )
    def generate_highlevel_summary(commits, char_limit=SUMMARY_CHAR_LIMIT):
        # Only include PRs with user-facing content
        enhancement_titles = [
            c.get('cleaned_title') or c.get('title')
            for c in commits
            if 'enhancement' in (c.get('labels') or []) and has_user_facing_content(c)
        ]
        other_titles = [
            c.get('cleaned_title') or c.get('title')
            for c in commits
            if 'enhancement' not in (c.get('labels') or []) and has_user_facing_content(c)
        ]
        ordered_titles = [t for t in enhancement_titles if t] + [t for t in other_titles if t]
        if not ordered_titles:
            return "This release includes no user-facing changes."
        # Clean and format titles
        def clean_title(title):
            t = title.strip()
            if t.endswith('.'):
                t = t[:-1]
            return t
        cleaned_titles = [clean_title(t) for t in ordered_titles]
        if not cleaned_titles:
            return "This release includes no user-facing changes."
        summary = "This release includes "
        listed = []
        truncated = False
        for i, title in enumerate(cleaned_titles):
            # Lowercase all but the first title
            if i == 0:
                t = title
            else:
                t = title[0].lower() + title[1:] if title else title
            next_list = listed + [t]
            # Use Oxford comma and 'and' if this is the last item and not truncated
            if i == len(cleaned_titles) - 1:
                candidate = summary + (", ".join(next_list[:-1]) + (", and " if len(next_list) > 1 else "") + next_list[-1] if len(next_list) > 1 else next_list[0]) + "."
            else:
                candidate = summary + ", ".join(next_list) + ", and more."
            if len(candidate) > char_limit:
                truncated = True
                break
            listed.append(t)
        if truncated:
            summary += ", ".join(listed) + ", and more."
        else:
            if len(listed) == 1:
                summary += listed[0] + "."
            else:
                summary += ", ".join(listed[:-1]) + ", and " + listed[-1] + "."
        return summary
    def validate_summary(summary, enhancement_titles):
        if enhancement_titles:
            return any(t and t.lower() in summary.lower() for t in enhancement_titles)
        return True
    # --- LLM proofread step ---
    def proofread_summary_with_llm(summary, max_tries=PROOFREAD_MAX_TRIES, debug=False):
        prompt = PROOFREAD_SUMMARY_PROMPT.format(summary=summary)
        for attempt in range(max_tries):
            try:
                api_params = get_model_api_params(MODEL_PROOFREADING, MAX_TOKENS_PROOFREADING)
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": MODEL_PROOFREADING_SYSTEM},
                        {"role": "user", "content": prompt}
                    ],
                    **api_params
                )
                improved = response.choices[0].message.content.strip()
                # Simple validation: must start with the required phrase and not be empty
                if improved.lower().startswith("this release includes") and len(improved) > 30:
                    return improved
                if debug:
                    print(f"Proofread attempt {attempt+1} failed validation: {improved}")
            except Exception as e:
                if debug:
                    print(f"Proofread attempt {attempt+1} failed: {e}")
        # Fallback to original if all attempts fail
        return summary
    enhancement_titles = [
        c.get('cleaned_title') or c.get('title')
        for c in all_commits
        if 'enhancement' in (c.get('labels') or []) and has_user_facing_content(c)
    ]
    summary = generate_highlevel_summary(all_commits)
    summary = proofread_summary_with_llm(summary, max_tries=PROOFREAD_MAX_TRIES, debug=debug)
    is_valid = validate_summary(summary, enhancement_titles)
    # --- End summary logic ---
    
    version_parts = version.replace('cmvm/', '').split('.')
    if '-rc' in version:
        release_type = "Release candidate"
        title_version = version
    elif len(version_parts) == 2:
        release_type = "Release"
        title_version = version
    else:
        release_type = "Hotfix release"
        title_version = version
    all_no_public_prs = all(
        rc.strip().startswith('<!--- ##') and 'No public PRs found for this release' in rc
        for rc in repo_contents
    )
    with open(file_path, 'w') as f:
        f.write("---\n")
        clean_title = title_version.replace('cmvm/', '')
        normalized_version = version.replace('cmvm/', '')
        f.write(f'title: "{clean_title} {release_type} notes"\n')
        if date:
            f.write(f'date: "{date}"\n')
        # Add categories field
        if '-rc' in version:
            yaml_release_type = 'release-candidate'
        elif len(version_parts) == 2:
            yaml_release_type = 'release'
        else:
            yaml_release_type = 'hotfix'
        f.write(f'categories: [cmvm, {normalized_version}, {yaml_release_type}]\n')
        f.write("sidebar: release-notes\n")
        f.write("toc-expand: true\n")
        if edit:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            if any_edited:
                f.write(f'# Content edited by AI - {current_time}\n')
            if any_validated:
                temp = None
                # Get validation temperature from validation summaries if available
                for commit in all_commits:
                    if commit.get('validation_summaries'):
                        max_temp = max((vs.get('validation_temperature', 0) for vs in commit['validation_summaries']), default=0)
                        if temp is None or max_temp > temp:
                            temp = max_temp
                    elif commit.get('validation_summary', {}).get('validation_temperature') is not None:
                        commit_temp = commit['validation_summary']['validation_temperature']
                        if temp is None or commit_temp > temp:
                            temp = commit_temp
                
                if temp is not None:
                    f.write(f'# Content validated by AI (temperature: {temp}) - {current_time}\n')
                else:
                    f.write(f'# Content validated by AI - {current_time}\n')
        if overwrite:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            f.write(f'# Content overwritten from an earlier version - {current_time}\n')
        f.write("---\n\n")
        # Write the summary here
        f.write(summary + "\n\n")
        if not is_valid:
            f.write("<!-- WARNING: Summary may not mention an enhancement PR -->\n\n")
        if all_no_public_prs:
            f.write('::: {.callout-info title="No user-facing changes in this release"}\n')
            f.write('This release includes no public-facing updates to features, bug fixes, or documentation. If you\'re unsure whether any changes affect your deployment, contact <support@validmind.com>.\n')
            f.write(':::\n\n')
        f.write("\n".join(content))
    print(f"\nCreated release notes file: {file_path}")
    
    # --- Begin per-PR file output logic ---
    generated_files = []
    for commit in all_commits:
        repo = commit.get('repo')
        if not repo:
            continue
        # Skip internal PRs
        if 'internal' in (commit.get('labels') or []):
            continue
        # Skip merge PRs
        if is_merge_pr(commit.get('title', '')):
            continue
        # Skip PRs with no content (same logic as in generate_changelog_content)
        has_content = bool(commit.get('external_notes') or commit.get('pr_summary') or commit.get('pr_body'))
        if not has_content:
            if debug:
                print(f"DEBUG: [create_release_file] Skipping per-PR file for #{commit.get('pr_number')} in {repo} (no content)")
            continue
        pr_number = commit.get('pr_number')
        tag_str = version.split('/')[-1]
        repo_dir = os.path.join(RELEASES_DIR, repo, tag_str)
        os.makedirs(repo_dir, exist_ok=True)
        pr_file = os.path.join(repo_dir, f"pr-{pr_number}.qmd")
        if os.path.exists(pr_file):
            print(f"DEBUG: [create_release_file] Checking overwrite for {pr_file} (overwrite={overwrite})")
            if not overwrite:
                print(f"DEBUG: [create_release_file] Skipping {pr_file} (already exists, overwrite is False)")
                print(f"DEBUG: [create_release_file] File {pr_file} already exists. Use --overwrite to update it.")
                continue
            else:
                print(f"DEBUG: [create_release_file] Overwriting {pr_file} (overwrite is True)")
        else:
            print(f"DEBUG: [create_release_file] Writing new file {pr_file}")
        # Prepare YAML header
        labels = commit.get('labels') or []
        valid_labels = labels if internal else [label for label in labels if label in label_hierarchy]
        categories = [repo, normalized_version, yaml_release_type] + valid_labels
        pr_title = commit.get('cleaned_title') or commit.get('title') or f"PR #{pr_number}"
        yaml_header = [
            '---',
            f'title: "{pr_title} (#{pr_number})"',
            # Categories are computed from:
            # - repo: The repository name (e.g. "backend")
            # - normalized_version: The version number (e.g. "25.05.04") 
            # - yaml_release_type: The release type (e.g. "hotfix")
            # - labels: PR labels (e.g. ["enhancement", "breaking-change"])
            f'categories: [{", ".join(categories)}]',
            f'sidebar: release-notes',
            f'toc-expand: true',
            f'date: "{date}"' if date else '',
        ]
        
        # Add metadata about editing status
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if commit.get('edited', False):
            yaml_header.append(f'# Content edited by AI - {current_time}')
        if commit.get('validated', False):
            # Add validation temperature information if available
            validation_summaries = commit.get('validation_summaries', [])
            temp = None
            if validation_summaries:
                # Get the highest validation temperature used across all validations
                temp = max((vs.get('validation_temperature', 0) for vs in validation_summaries), default=0)
            elif commit.get('validation_summary', {}).get('validation_temperature') is not None:
                temp = commit['validation_summary']['validation_temperature']
            
            if temp is not None:
                yaml_header.append(f'# Content validated by AI (temperature: {temp}) - {current_time}')
            else:
                yaml_header.append(f'# Content validated by AI - {current_time}')
                
        if overwrite:
            yaml_header.append(f'# Content overwritten from an earlier version - {current_time}')
        yaml_header.append(f'# PR URL: https://github.com/validmind/{repo}/pull/{pr_number}')
        yaml_header.append('---\n\n')
        # Prepare content (summary, notes, body)
        content_parts = []
        # Use edited summary content (stored in pr_summary after editing, with fallback to pr_interpreted_summary)
        if commit.get('pr_summary'):
            content_parts.append(update_image_links(commit['pr_summary'], version, debug, download_assets))
        elif commit.get('pr_interpreted_summary'):
            content_parts.append(update_image_links(commit['pr_interpreted_summary'], version, debug, download_assets))
        # Use edited content (stored back in external_notes after editing)
        if commit.get('external_notes'):
            content_parts.append(update_image_links(commit['external_notes'], version, debug, download_assets))
        content = '\n\n'.join([c for c in content_parts if c])
        
        # Add validation summary at the end if available
        validation_comment = generate_validation_comment(commit, debug)
        if validation_comment:
            content += validation_comment
            
        # Write file
        with open(pr_file, 'w') as f:
            f.write('\n'.join(yaml_header))
            f.write(content)
        generated_files.append(pr_file)
    if generated_files:
        print("\nGenerated per-PR release notes files:")
        for fpath in generated_files:
            print(f"  - {fpath}")
    # --- End per-PR file output logic ---
    return

def parse_release_tables(qmd_file_path, version=None, debug=False):
    """Parse release tables from the customer-managed-validmind-releases.qmd file.
    
    Args:
        qmd_file_path (str): Path to the QMD file containing release tables
        version (str, optional): Specific version to parse, if None parse all
        debug (bool): Whether to show debug output
        
    Returns:
        List[dict]: List of release information dictionaries
        set: Set of seen versions
    """
    releases = []
    seen_versions = set()  # Track seen versions for analysis
    
    print("Fetching releases from table ...")
    
    # Normalize version for comparison if provided
    normalized_version = None
    if version:
        normalized_version = version.replace('cmvm/', '')
    
    try:
        if debug:
            print(f"DEBUG: Reading file: {qmd_file_path}")
        with open(qmd_file_path, 'r') as f:
            content = f.read()
            
        # Parse major releases
        major_start = content.find('### Major releases')
        if major_start == -1:
            major_start = content.find('Major Releases:')
        if major_start != -1:
            # Find the next section or end of file
            next_section = content.find('###', major_start + 1)
            if next_section == -1:
                next_section = len(content)
            major_table = content[major_start:next_section]
            if debug:
                print(f"DEBUG: Found major table with length: {len(major_table)}")
            for line in major_table.split('\n'):
                # Skip commented lines and table separators
                if line.strip().startswith('<!--') or line.strip().startswith('--'):
                    continue
                if '|' in line and not line.startswith('--') and not line.startswith('<!--'):
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 8:
                        version_str = parts[0]
                        # Skip header row, non-version rows, and version 00.00
                        if not version_str[0].isdigit() or version_str == '00.00':
                            continue
                        # Compare normalized versions if a specific version is requested
                        if normalized_version and version_str != normalized_version:
                            continue
                        date = parts[1]
                        git_sha = parts[6].strip('`').strip("'")
                        is_hotfix = '**Yes**' in parts[4]
                        is_rc = 'Yes' in parts[5]
                        
                        if version_str in seen_versions and not normalized_version:
                            print(f"INFO: Version {version_str} appears in both major and hotfix & release candidate tables")
                        seen_versions.add(version_str)
                        
                        if git_sha.lower() in ['n/a', 'tbd', 'xxx'] and not normalized_version:
                            print(f"INFO: Version {version_str} has invalid Git SHA: {git_sha}")
                        if date.lower() in ['n/a'] and not normalized_version:
                            print(f"INFO: Version {version_str} has invalid date: {date}")
                            
                        release = create_release_object(
                            version=version_str,
                            date=date,
                            git_sha=git_sha,
                            is_hotfix=is_hotfix,
                            is_rc=is_rc,
                            table='major',
                            add_prefix=True  # Always add cmvm/ prefix
                        )
                        releases.append(release)
                        if debug:
                            print(f"DEBUG: Added major release: {version_str}")
        
        # Parse hotfix & release candidate releases
        hotfix_start = content.find('### Hotfix and release candidates')
        if hotfix_start != -1:
            # Find the next section or end of file
            next_section = content.find('###', hotfix_start + 1)
            if next_section == -1:
                next_section = len(content)
            hotfix_table = content[hotfix_start:next_section]
            if debug:
                print(f"DEBUG: Found hotfix table with length: {len(hotfix_table)}")
            for line in hotfix_table.split('\n'):
                # Skip commented lines and table separators
                if line.strip().startswith('<!--') or line.strip().startswith('--'):
                    continue
                if '|' in line and not line.startswith('--') and not line.startswith('<!--'):
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 8:
                        version_str = parts[0]
                        # Skip header row, non-version rows, and version 00.00
                        if not version_str[0].isdigit() or version_str == '00.00':
                            continue
                        # Compare normalized versions if a specific version is requested
                        if normalized_version and version_str != normalized_version:
                            continue
                        date = parts[1]
                        git_sha = parts[6].strip('`').strip("'")
                        is_hotfix = '**Yes**' in parts[4]
                        is_rc = 'Yes' in parts[5]
                        
                        if version_str in seen_versions and not normalized_version:
                            print(f"INFO: Version {version_str} appears in both major and hotfix & release candidate tables")
                        seen_versions.add(version_str)
                        
                        if git_sha.lower() in ['n/a', 'tbd', 'xxx'] and not normalized_version:
                            print(f"INFO: Version {version_str} has invalid Git SHA: {git_sha}")
                        if date.lower() in ['n/a'] and not normalized_version:
                            print(f"INFO: Version {version_str} has invalid date: {date}")
                            
                        release = create_release_object(
                            version=version_str,
                            date=date,
                            git_sha=git_sha,
                            is_hotfix=is_hotfix,
                            is_rc=is_rc,
                            table='hotfix',
                            add_prefix=True  # Always add cmvm/ prefix
                        )
                        releases.append(release)
                        if debug:
                            print(f"DEBUG: Added hotfix release: {version_str}")
    
    except Exception as e:
        print(f"Error parsing release tables: {e}")
        if debug:
            import traceback
            print("DEBUG: Full traceback:")
            print(traceback.format_exc())
        return [], set()
        
    # Sort releases by date in descending order, with version_key as secondary sort
    releases.sort(key=lambda x: (parse_date(x['date']), version_key(x['version'])), reverse=True)
    
    if debug:
        print(f"DEBUG: Found {len(releases)} total releases")
        print(f"DEBUG: Found {len(seen_versions)} unique version(s)")
        if releases:
            print("DEBUG: First few releases:")
            for r in releases[:3]:
                print(f"  - {r['version']} ({r['date']})")
    
    # Analyze the data only if no specific version requested
    if not normalized_version and debug:
        print("\nRelease Analysis:")
        print("----------------")
        print(f"Total releases found: {len(releases)}")
        # Count releases by type
        major_releases = [r for r in releases if r['table'] == 'major']
        hotfix_releases = [r for r in releases if r['table'] == 'hotfix' and not r['is_rc']]
        rc_releases = [r for r in releases if r['is_rc']]
        print(f"Major releases: {len(major_releases)}")
        print(f"Hotfix releases: {len(hotfix_releases)}")
        print(f"Release candidates: {len(rc_releases)}\n")
    
    return releases, seen_versions

def process_releases(releases, overwrite, seen_versions, debug=False, version=None, edit=False, single=False, internal=False, download_assets=True):
    """Process all releases and create release note files.
    
    Args:
        releases: List of release dictionaries
        overwrite: Whether to overwrite existing files
        seen_versions: Set of versions that have been seen
        debug: Whether to show debug output
        version: Specific version to process (if any)
        edit: Whether to edit content using OpenAI
        single: Whether to write a single release-notes.qmd file (legacy mode)
    """
    print("\nProcessing release(s) ...")
      
    # Group releases by version
    releases_by_version = {}
    for release in releases:
        # Skip if this isn't the version we're looking for
        if version:
            # Remove cmvm/ prefix from both versions for comparison
            release_version = release['version'].replace('cmvm/', '')
            requested_version = version.replace('cmvm/', '')
            if release_version != requested_version:
                if debug:
                    print(f"DEBUG: Skipping {release['version']} - not the requested version")
                continue
            if debug:
                print(f"DEBUG: Found {release['version']} - our requested version")
        
        # Group releases by version
        if release['version'] not in releases_by_version:
            releases_by_version[release['version']] = []
        releases_by_version[release['version']].append(release)
        
        # Add to seen versions
        seen_versions.add(release['version'].replace('cmvm/', ''))
    
    print(f"Found {len(releases_by_version)} unique version(s) to process")
    
    # Process each version's releases
    for version_key, version_releases in releases_by_version.items():
        print(f"\nProcessing version {version_key} ...")
        
        # Check if the release exists in GitHub - parallelize across repos
        def check_repo_for_tag(repo):
            return check_github_tag(repo, version_key)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_repo = {executor.submit(check_repo_for_tag, repo): repo for repo in REPOS}
            tag_exists = False
            for future in concurrent.futures.as_completed(future_to_repo):
                repo = future_to_repo[future]
                try:
                    if future.result():
                        tag_exists = True
                        if debug:
                            print(f"DEBUG: Found tag in {repo}")
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Error checking tag in {repo}: {e}")
                
        if not tag_exists:
            print(f"WARNING: Tag {version_key} not found in any repository")
            continue
            
        print("Creating release file ...")
        # Create the release file once per version
        create_release_file(version_releases[0], overwrite, debug, edit, single, internal, download_assets)
        print(f"✓ Completed processing version {version_key}")
        
        # If a specific tag was requested by the user, we're done after processing it
        if version is not None:
            break
    
    print("✓ Finished processing all specified releases")

def collect_github_urls():
    """Collects release URLs from GitHub.
    
    Returns:
        List[ReleaseURL]: A list of ReleaseURL objects
    """
    releases, seen_versions = get_releases_from_github()
    
    if not releases:
        print("ERROR: No releases found in GitHub")
        sys.exit(1)
        
    urls = []
    repos = ["backend", "frontend", "agents", "documentation", "validmind-library"]
    
    for release in releases:
        # Construct GitHub URL based on the git_sha
        # If git_sha starts with 'cmvm/', it's a tag, otherwise it's a commit SHA
        if release['git_sha'].startswith('cmvm/'):
            url = f"https://github.com/validmind/backend/releases/tag/{release['git_sha']}"
        else:
            url = f"https://github.com/validmind/backend/releases/tag/{release['version']}"
        
        urls.append(ReleaseURL(url))
        print(f"Discovered release {release['version']} ({release['date']})")
        
        # Check for tags in all repos
        print("Found tags:")
        for repo in repos:
            tag_url = f"https://github.com/validmind/{repo}/releases/tag/cmvm%2F{release['version']}"
            print(f"  - {tag_url}")
        print()
    
    return urls

def count_repos(urls):
    """Counts occurrences of each repository in the given URLs.

    Args:
        urls (List[ReleaseURL]): A list of ReleaseURL objects

    Prints:
        Repository counts in the format 'repo_name: count'
    """
    print("RELEASE TAGS ADDED BY REPO:\n")
    repo_names = [url.extract_repo_name() for url in urls if url.extract_repo_name()]
    
    counts = Counter(repo_names)
    for repo, count in counts.items():
        print(f"{repo}: {count}")

def get_release_date():
    """Gets the release date from GitHub.
    
    Returns:
        datetime: The release date
    """
    releases, _ = get_releases_from_github()
    
    if not releases:
        print("ERROR: No releases found in GitHub")
        sys.exit(1)
        
    # Get the most recent release date
    latest_release = releases[0]
    date_str = latest_release['date']
    
    try:
        release_date = datetime.datetime.strptime(date_str, "%B %d, %Y")
        print(f"Release date: {release_date}\n")
        return release_date
    except ValueError:
        print(f"ERROR: Invalid date format in release: {date_str}")
        sys.exit(1)

def create_release_folder(formatted_release_date):
    """
    Creates a directory for the release notes based on the provided release date
    and returns the output file path.

    Args:
        formatted_release_date (str): The formatted release date string.

    Returns:
        str: The path to the release notes file.
    """
    # Parse the input date
    parsed_date = datetime.datetime.strptime(formatted_release_date, "%Y-%b-%d")
    year = parsed_date.year
    formatted_date = parsed_date.strftime("%Y-%b-%d").lower()  # e.g., "2025-jan-17"
    directory_path = f"../site/releases/{year}/{formatted_date}/"
    output_file = f"{directory_path}release-notes.qmd"

    # Create directory and output file
    os.makedirs(directory_path, exist_ok=True)
    print(f"Created release folder: {directory_path}")

    return output_file, year

def create_release_qmd(output_file, original_release_date):
    """
    Writes metadata to a file with a title set to the original release date.

    Args:
        output_file (str): The path to the file to write.
        original_release_date (str): The title to include in the metadata.
    """
    with open(output_file, "w") as file:
        file.write(f"---\ntitle: \"{original_release_date}\"\n---\n\n")
    print(f"Created release notes file: {output_file}")

def update_release_components(release_components, categories):
    """
    Updates a dictionary of release components with the given categories.

    Parameters:
        release_components (dict): The dictionary to update.
        categories (dict): The categories to add to the release components.

    Returns:
        dict: The updated release components dictionary.
    """
    release_components.update(categories)
    if get_ipython():  # Check if running in Jupyter Notebook
        print(f"Set up {len(release_components)} components:")
    else:
        print(f"Set up {len(release_components)} components:\n" + "\n".join(release_components))
    return release_components

def set_names(github_urls):
    """
    Iterates over a list of URL objects, calling the `set_repo_and_tag_name` method on each.

    Parameters:
        github_urls (list): A list of objects, each having the method `set_repo_and_tag_name`.

    Returns:
        None
    """
    # Mapping of repo names to headers
    repo_to_header = {
        "validmind/frontend": "FRONTEND",
        "validmind/documentation": "DOCUMENTATION",
        "validmind/agents": "VALIDMIND LIBRARY",
    }

    print("Assigning repo and tag names ...\n")

    # Group URLs by repo name for better formatting
    grouped_urls = {}
    for url_obj in github_urls:
        url_obj.set_repo_and_tag_name()
        if url_obj.repo_name not in grouped_urls:
            grouped_urls[url_obj.repo_name] = []
        grouped_urls[url_obj.repo_name].append(url_obj)

    # Print output in the desired format
    for repo_name, urls in grouped_urls.items():
        header = repo_to_header.get(repo_name, repo_name.upper())
        print(f"{header}:\n")
        for url_obj in urls:
            print(f"URL: {url_obj.url}\n Repo name: {url_obj.repo_name}\n Tag name: {url_obj.tag_name}\n")

def extract_urls(github_urls):
    """
    Extracts pull request (PR) objects from a list of GitHub URLs.

    Args:
        github_urls (iterable): An iterable containing GitHub URL objects that 
                               have an `extract_prs` method.

    Returns:
        None: The `extract_prs` method modifies the URL objects in-place.
    """
    for url in github_urls:
        url.extract_prs()
        print()

def populate_data(urls):
    """
    Populates pull request data for a list of URLs.

    Args:
        urls (iterable): An iterable of objects with a `populate_pr_data` method.
    """
    for url in urls:
        url.populate_pr_data()
        print()

def edit_release_notes(github_urls, editing_instructions_body, debug=False):
    """
    Processes a list of GitHub URLs to extract and edit release notes for pull requests.

    Args:
        github_urls (list): List of GitHub URL objects containing pull requests.
        editing_instructions_body (str): Instructions for editing the text with OpenAI.
        debug (bool): Whether to show debug output

    Returns:
        None
    """

    print("Editing release notes content ...")

    if debug:
        print(f"DEBUG: [edit_release_notes] Starting with {len(github_urls)} URLs")
        print(f"DEBUG: [edit_release_notes] Editing instructions length: {len(editing_instructions_body)}")
    for url in github_urls:
        if debug:
            print(f"DEBUG: [edit_release_notes] Processing URL: {url.url}")
            print(f"DEBUG: [edit_release_notes] Number of PRs in URL: {len(url.prs)}")
        for pr in url.prs:
            if pr.data_json:
                if debug:
                    print(f"DEBUG: [edit_release_notes] Found data_json for PR #{pr.pr_number} in {pr.repo_name}")
                print(f"Editing content of PR #{pr.pr_number} from {pr.repo_name} ...\n") 
                # Get the external notes that were already classified in get_pr_content
                external_notes, pr_summary, labels, title, pr_body, image_urls = get_pr_content(pr.pr_number, pr.repo_name, debug)
                if external_notes:
                    if debug:
                        print(f"DEBUG: [edit_release_notes] Editing external notes for PR #{pr.pr_number}")
                    pr.edit_content('notes', external_notes, editing_instructions_body)
                else:
                    if debug:
                        print(f"DEBUG: [edit_release_notes] No external notes found for PR #{pr.pr_number}")
        print()



def edit_titles(github_urls, debug=False):
    """
    Updates the titles of pull requests (PRs) based on provided JSON data and cleaning instructions.

    Parameters:
        github_urls (list): A list of GitHub URLs, each containing PRs to process.
        debug (bool): Whether to show debug output
    """
    for url in github_urls:
        for pr in url.prs:
            if pr.data_json:
                print(f"Editing title for PR #{pr.pr_number} in {pr.repo_name}...\n")
                prompt = EDIT_TITLE_PROMPT.format(title=pr.data_json['title'], body=pr.data_json.get('body', ''))
                if debug:
                    print(f"DEBUG: [edit_titles] Editing title for PR #{pr.pr_number} in {pr.repo_name}")
                pr.edit_content('title', pr.data_json['title'], prompt)
                print()
        print()

def set_labels(github_urls):
    """
    Processes a list of GitHub URLs and extracts pull request labels, printing them.

    Args:
        github_urls (list): A list of GitHub URL objects, each containing pull requests (prs).
    """
    print(f"Attaching labels to PRs ...\n\n")
    for url in github_urls:
        for pr in url.prs:
            if pr.data_json:
                pr.labels = [label['name'] for label in pr.data_json['labels']]
                print(f"PR #{pr.pr_number} from {pr.repo_name}: {pr.labels}\n")
        print()

def assign_details(github_urls):
    """
    Processes a list of GitHub URLs and extracts details for PRs with data in `data_json`.

    Args:
        github_urls (list): A list of objects representing GitHub URLs, each containing PRs.

    Returns:
        None
    """
    print(f"Compiling PR data ...\n\n")
    for url in github_urls:
        for pr in url.prs:
            if pr.data_json:
                pr.pr_details = {
                    'pr_number': pr.pr_number,
                    'title': pr.cleaned_title,
                    'full_title': pr.data_json['title'],
                    'url': pr.data_json['url'],
                    'labels': ", ".join(pr.labels),
                    'notes': pr.edited_text
                }
                print(f"PR #{pr.pr_number} from {pr.repo_name} compiled.\n")
        print()

def assemble_release(github_urls, label_hierarchy):
    """
    Assigns PRs from a list of GitHub release URLs to release components based on their labels.

    Parameters:
        github_urls (list): A list of GitHub URL objects, each containing PRs.
        label_hierarchy (list): A prioritized list of labels to determine component assignment.

    Returns:
        dict: A dictionary where keys are labels from the hierarchy (or 'other') and values are lists of PR details.
    """
    # Initialize release_components as a defaultdict with lists
    release_components = defaultdict(list)

    # Process PRs and assign them to components based on label hierarchy
    unassigned_prs = []  # Track PRs that do not match any label in the hierarchy

    for url in github_urls:
        for pr in url.prs:
            if pr.data_json:
                print(f"Assembling PR #{pr.pr_number} from {pr.repo_name} for release notes...\n")
                assigned = False
                for priority_label in label_hierarchy:
                    if priority_label in pr.labels:
                        release_components[priority_label].append(pr.pr_details)
                        assigned = True
                        break
                if not assigned:
                    unassigned_prs.append(pr.pr_details)
        print()

    # Add unassigned PRs to the 'other' category
    release_components['other'].extend(unassigned_prs)

    # Convert defaultdict to a regular dict and ensure 'other' is at the end
    result = {label: release_components[label] for label in label_hierarchy if label in release_components}
    if 'other' in release_components:
        result['other'] = release_components['other']

    return result

def release_output(output_file, release_components, label_to_category):
    """
    Appends release notes to the specified file.

    Args:
        output_file (str): Path to the file to append.
        release_components (dict): Release notes categorized by labels.
        label_to_category (dict): Mapping of labels to formatted categories.

    Returns:
        None
    """
    try:
        with open(output_file, "a") as file:
            write_file(file, release_components, label_to_category)
            print(f"Assembled release notes added to {file.name}\n")
    except Exception as e:
        print(f"Failed to write to {output_file}: {e}")

# def upgrade_info(output_file):
#     """
#     Appends the upgrade information single-source to the end of the new release notes.

#     Args:
#         output_file (str): Path to the file to append.

#     Returns:
#         None
#     """
#     include_directive = "\n\n{{< include /releases/_how-to-upgrade.qmd >}}\n"

#     try:
#         with open(output_file, "a") as file:
#             file.write(include_directive)
#             print(f"Include _how-to-upgrade.qmd added to {file.name}")
#     except Exception as e:
#         print(f"Failed to include _how-to-upgrade.qmd to {output_file}: {e}")

def write_file(file, release_components, label_to_category):
    """Writes each component of the release notes into a file
    Args:
        file - desired file path
        release_components
        label_to_category

    Modifies: 
        file
    """
    for label, release_component in release_components.items():
        if release_component:  # Only write heading if there are PRs
            output_lines = [f"{label_to_category.get(label, '### Unlabelled changes')}\n\n"]
            last_line_was_blank = False

            for pr in release_component:
                pr_lines = [
                    f"<!---\nPR #{pr['pr_number']}: {pr['full_title']}\n",
                    f"URL: {pr['url']}\n",
                    f"Labels: {pr['labels'] if pr['labels'] else 'none'}\n",
                    f"--->\n"
                ]
                # Output title warning comment if present
                if hasattr(pr, 'cleaned_title_comment') and pr.cleaned_title_comment:
                    pr_lines.append(pr.cleaned_title_comment)
                pr_lines.append(f"### {pr['title']}\n")
                pr_lines.append(f"<!--- Source: {pr['url']} --->\n\n")
                # Output content warning comment if present
                if hasattr(pr, 'content_warning_comment') and pr.content_warning_comment:
                    pr_lines.append(pr.content_warning_comment)
                if pr['notes']:
                    pr_lines.append(f"{pr['notes']}\n\n")
                
                # Add validation summary if available
                if hasattr(pr, 'validation_summary') or 'validation_summary' in pr:
                    validation_comment = generate_validation_comment(pr, debug=False)
                    if validation_comment:
                        pr_lines.append(validation_comment)
                
                for line in pr_lines:
                    if line.strip() == "":
                        if last_line_was_blank:
                            continue
                        last_line_was_blank = True
                    else:
                        last_line_was_blank = False
                    output_lines.append(line)
            # Write processed lines to file
            file.writelines(output_lines)

def create_release_object(version, date=None, git_sha=None, is_hotfix=False, is_rc=False, table=None):
    """Create a release object with the given information.
    
    Args:
        version (str): Release version
        date (str, optional): Release date
        git_sha (str, optional): Git SHA or tag
        is_hotfix (bool): Whether this is a hotfix
        is_rc (bool): Whether this is a release candidate
        table (str, optional): Source of release information ('github' or 'table')
        
    Returns:
        dict: Release object
    """
    return {
        'version': version,
        'date': date,
        'git_sha': git_sha,
        'is_hotfix': is_hotfix,
        'is_rc': is_rc,
        'table': table
    }

def get_releases_from_github(version=None, debug=False):
    """Get CMVM releases from GitHub.
    
    Args:
        version (str, optional): Specific version to get, if None get all
        debug (bool): Whether to print debug information
        
    Returns:
        List[dict]: List of release information dictionaries
        set: Set of seen versions
    """
    releases = []
    seen_versions = set()
    
    print(f"Fetching tags and releases from GitHub ...")
    
    # Normalize version for comparison if provided
    normalized_version = None
    if version:
        normalized_version = version  # Keep the full version including cmvm/ prefix
    
    # Get all CMVM tags from each repository in parallel with rate limiting
    def get_repo_tags(repo):
        try:
            tags = get_all_cmvm_tags(repo, version=version, debug=debug)
            # Filter tags if a specific version is requested
            if normalized_version:
                tags = [tag for tag in tags if tag == normalized_version]
            return [(repo, tag) for tag in tags]
        except Exception as e:
            if debug:
                print(f"DEBUG: [get_releases_from_github] Error getting tags from {repo}: {e}")
            return []
    
    # Use ThreadPoolExecutor with limited concurrency
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_repo = {executor.submit(get_repo_tags, repo): repo for repo in REPOS}
        repo_tags = []
        for future in as_completed(future_to_repo):
            repo = future_to_repo[future]
            try:
                tags = future.result()
                if tags:
                    repo_tags.extend(tags)
                    if debug:
                        print(f"DEBUG: [get_releases_from_github] Found {len(tags)} tags in {repo}")
            except Exception as e:
                if debug:
                    print(f"DEBUG: Error processing tags from {repo}: {e}")
    
    if debug:
        print(f"DEBUG: [get_releases_from_github] Found {len(repo_tags)} tags across all repos")
        for repo, tag in repo_tags:
            print(f"DEBUG: [get_releases_from_github] Tag {tag} in {repo}")
    
    if not repo_tags:
        print(f"ERROR: [get_releases_from_github] No tags found for version {version if version else 'any'}")
        return [], set()
    
    # Process tags in parallel with rate limiting
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_tag = {executor.submit(process_tag, repo_tag): repo_tag for repo_tag in repo_tags}
        for future in as_completed(future_to_tag):
            repo, tag = future_to_tag[future]
            try:
                release = future.result()
                if release:
                    # Only add if we haven't seen this version before
                    if release['version'] not in seen_versions:
                        releases.append(release)
                        seen_versions.add(release['version'])
                        if debug:
                            print(f"DEBUG: [get_releases_from_github] Added release: {release['version']} from {repo}")
            except Exception as e:
                if debug:
                    print(f"DEBUG: [get_releases_from_github] Error processing tag {tag} in {repo}: {e}")
    
    if not releases:
        print(f"ERROR: [get_releases_from_github] No releases found for version {version if version else 'any'}")
        return [], set()
    
    # Sort releases by date in descending order, with version_key as secondary sort
    releases.sort(key=lambda x: (parse_date(x['date']), version_key(x['version'])), reverse=True)
    
    if debug:
        print(f"DEBUG: [get_releases_from_github] Processed {len(releases)} total release(s), {len(seen_versions)} unique version(s)")
                
    return releases, seen_versions

def process_tag(repo_tag):
    """Process a tag to extract release information.
    
    Args:
        repo_tag (tuple): Tuple of (repo, tag) to process
        
    Returns:
        dict: Release information or None if processing fails
    """
    repo, tag = repo_tag
    try:
        # Get tag data first (more fundamental, always exists)
        cmd = ['gh', 'api', f'repos/validmind/{repo}/git/refs/tags/{tag}']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            tag_data = json.loads(result.stdout)
            # Get the commit SHA from the tag reference
            commit_sha = tag_data.get('object', {}).get('sha')
            if not commit_sha:
                print(f"  WARN: No commit SHA found for tag {tag} in {repo}")
                return None
                
            # Get the commit data to get the date
            cmd = ['gh', 'api', f'repos/validmind/{repo}/git/commits/{commit_sha}']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                commit_data = json.loads(result.stdout)
                date = commit_data.get('committer', {}).get('date')
                if date:
                    # Convert ISO format to desired format
                    date = datetime.datetime.fromisoformat(date.replace('Z', '+00:00')).strftime("%B %d, %Y")
                else:
                    date = "N/A"
                    
                # Check if release exists (for metadata) but don't use its date
                release_data = None
                cmd = ['gh', 'api', f'repos/validmind/{repo}/releases/tags/{tag}']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    release_data = json.loads(result.stdout)
                    # Always use tag creation date, not release published date
                
                return create_release_object(
                    version=tag,
                    date=date,
                    git_sha=commit_sha,  # Use actual commit SHA
                    is_hotfix=False,  # Default to False, will be updated if found in table
                    is_rc='-rc' in tag,  # Check if it's an RC based on tag name
                    table='github'
                )
            else:
                print(f"  WARN: Could not get commit data for {tag} in {repo}")
                return None
        else:
            print(f"  WARN: Could not get tag data for {tag} in {repo}")
            return None
            
    except Exception as e:
        print(f"  WARN: Error processing tag {tag} in {repo}: {e}")
        return None

def download_with_playwright(url, output_path, browser_profile_dir=None, headless=True):
    """
    Download an image from a web page using Playwright, supporting authenticated sessions via storage state.
    Handles redirects and HTML pages with <img> tags.
    """
    from playwright.sync_api import sync_playwright
    import os
    from urllib.parse import urljoin
    print(f"DEBUG: [download_with_playwright] Attempting to download {url} to {output_path}")
    print(f"DEBUG: [download_with_playwright] CWD: {os.getcwd()}")
    storage_state_path = os.path.join(os.path.dirname(__file__), "auth.json")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=storage_state_path)
        try:
            page = context.new_page()
            # --- Authentication check ---
            page.goto("https://github.com/")
            page.wait_for_load_state("networkidle")
            if page.query_selector("text=Sign in") is not None:
                print("WARNING: [download_with_playwright] Playwright browser is not authenticated. Please log in to GitHub and save auth.json before running this script.")
                return False
            # Try to download the image or follow redirects
            response = page.goto(url)
            if not response:
                print(f"DEBUG: [download_with_playwright] No response from {url}")
                return False
            status = response.status
            content_type = response.headers.get('content-type', '')
            print(f"DEBUG: [download_with_playwright] Initial response status: {status}, content-type: {content_type}")
            # Handle redirects
            if status in [301, 302, 303, 307, 308]:
                redirect_url = response.headers.get('location')
                if redirect_url:
                    print(f"DEBUG: [download_with_playwright] Following redirect to {redirect_url}")
                    response = page.goto(redirect_url)
                    status = response.status if response else None
                    content_type = response.headers.get('content-type', '') if response else ''
            # If direct image, save it
            if response and content_type.startswith('image/'):
                with open(output_path, 'wb') as f:
                    f.write(response.body())
                print(f"DEBUG: [download_with_playwright] Image saved to {output_path} (direct image)")
                return True
            # If not, look for <img> tag
            img = page.query_selector('img')
            if img:
                img_src = img.get_attribute('src')
                print(f"DEBUG: [download_with_playwright] Found <img> tag with src: {img_src}")
                if img_src and not img_src.startswith('data:'):
                    img_url = urljoin(page.url, img_src)
                    print(f"DEBUG: [download_with_playwright] Downloading actual image from: {img_url}")
                    img_response = page.goto(img_url)
                    if img_response and img_response.headers.get('content-type', '').startswith('image/'):
                        with open(output_path, 'wb') as f:
                            f.write(img_response.body())
                        print(f"DEBUG: [download_with_playwright] Image saved to {output_path} (from <img> src)")
                        return True
                    else:
                        print(f"DEBUG: [download_with_playwright] Failed to download image from <img> src: {img_url}")
                else:
                    print(f"DEBUG: [download_with_playwright] <img> src is a data URL or missing, skipping.")
            else:
                print(f"DEBUG: [download_with_playwright] No <img> tag found on the page.")
            return False
        except Exception as e:
            print(f"DEBUG: [download_with_playwright] Exception: {e}")
            return False
        finally:
            context.close()
            browser.close()

def download_image(url, tag, debug=False, download_assets=True):
    """Download an image or video from a URL and save it to a tag-specific folder."""
    import subprocess
    import imghdr
    
    # If asset downloading is disabled, return None
    if not download_assets:
        if debug:
            print(f"DEBUG: [download_image] Asset downloading not specified with --download-assets, skipping {url}")
        return None
        
    try:
        # Determine if this is a cmvm tag or not
        tag_str = tag.split('/')[-1]
        if 'cmvm' in tag.lower():
            releases_dir = os.path.join('releases', 'cmvm', tag_str)
        else:
            releases_dir = os.path.join('releases', tag_str)
        os.makedirs(releases_dir, exist_ok=True)
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = f"image_{hash(url)}.png"
        elif not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp4', '.mov', '.qt']):
            filename = f"{filename}.png"
        local_path = os.path.join(releases_dir, filename)
        # Try GitHub CLI first
        if 'github.com/user-attachments/' in url:
            asset_id = url.split('/')[-1]
            try:
                if debug:
                    print(f"DEBUG: [download_image] Trying GitHub CLI for asset {asset_id}")
                cmd = [
                    'gh', 'api', f'/user-attachments/assets/{asset_id}'
                ]
                with open(local_path, 'wb') as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)
                if debug:
                    print(f"DEBUG: [download_image] gh api return code: {result.returncode}")
                    if result.stderr:
                        print(f"DEBUG: [download_image] gh api stderr: {result.stderr[:200]}")
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    img_type = imghdr.what(local_path)
                    if img_type is not None:
                        if debug:
                            print(f"DEBUG: [download_image] Successfully downloaded file to {local_path} via gh CLI (type: {img_type})")
                        return local_path
                    else:
                        if debug:
                            print(f"DEBUG: [download_image] File at {local_path} is not a valid image, deleting.")
                        os.remove(local_path)
                else:
                    if debug:
                        print(f"DEBUG: [download_image] gh CLI failed or file is empty.")
            except Exception as e:
                if debug:
                    print(f"DEBUG: [download_image] gh CLI download failed: {e}")
            # If CLI fails, try GitHub REST API
            api_url = f"https://api.github.com/user-attachments/assets/{asset_id}"
            headers = {}
            github_token = os.getenv('GITHUB_TOKEN')
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            try:
                response = requests.get(api_url, headers=headers, stream=True)
                if debug:
                    print(f"DEBUG: [download_image] GitHub API response status: {response.status_code}")
                if response.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    img_type = imghdr.what(local_path)
                    if img_type is not None:
                        if debug:
                            print(f"DEBUG: [download_image] Successfully downloaded file to {local_path} via GitHub API (type: {img_type})")
                        return local_path
                    else:
                        if debug:
                            print(f"DEBUG: [download_image] File at {local_path} is not a valid image, deleting.")
                        os.remove(local_path)
                else:
                    if debug:
                        print(f"DEBUG: [download_image] GitHub API failed or file is empty.")
            except Exception as e:
                if debug:
                    print(f"DEBUG: [download_image] GitHub API download failed: {e}")
            # Playwright fallback for other URLs
            try:
                if debug:
                    print(f"DEBUG: [download_image] Trying Playwright fallback for {url}")
                output_path = local_path
                # NOTE: To use an authenticated session, you must first log in manually with Playwright and save auth.json.
                success = download_with_playwright(
                    url, output_path,
                    browser_profile_dir=None,
                    headless=True
                )
                if debug:
                    print(f"DEBUG: [download_image] Playwright download result: {success}")
                    print(f"DEBUG: [download_image] File exists after Playwright: {os.path.exists(output_path)}")
                if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    img_type = imghdr.what(output_path)
                    if img_type is not None:
                        if debug:
                            print(f"DEBUG: [download_image] Playwright successfully downloaded file to {output_path} (type: {img_type})")
                        return output_path
                    else:
                        if debug:
                            print(f"DEBUG: [download_image] File at {output_path} is not a valid image, deleting.")
                        os.remove(output_path)
                else:
                    if debug:
                        print(f"DEBUG: [download_image] Playwright failed for {url}")
            except Exception as e:
                if debug:
                    print(f"DEBUG: [download_image] Playwright fallback failed: {e}")
        return None
    except Exception as e:
        if debug:
            print(f"DEBUG: [download_image] Failed to download image/video from {url}: {e}")
        return None

def update_image_links(content, tag, debug=False, download_assets=True):
    """Update image links in markdown content to use local paths.
    
    Args:
        content (str): Markdown content containing image links
        tag (str): Tag name to use for folder organization
        debug (bool): Whether to show debug output
        download_assets (bool): Whether to download assets or leave URLs unchanged
        
    Returns:
        str: Updated markdown content with local image paths (if download_assets=True)
    """
    if not content:
        return content
    
    # If asset downloading is disabled, return content unchanged
    if not download_assets:
        if debug:
            print(f"DEBUG: [update_image_links] Asset downloading disabled, leaving URLs unchanged")
        return content

    def replace_image_link(match):
        # Markdown image
        if match.group(1) is not None and match.group(2) is not None:
            alt_text = match.group(1)
            url = match.group(2)
            local_path = download_image(url, tag, debug, download_assets)
            if local_path:
                rel_path = os.path.relpath(local_path, os.getcwd())
                return f'![{alt_text}]({rel_path})'
            else:
                return match.group(0)
        # HTML <img> tag
        elif match.group(3) is not None and match.group(4) is not None and match.group(5) is not None:
            before_src = match.group(3)
            url = match.group(4)
            after_src = match.group(5)
            local_path = download_image(url, tag, debug, download_assets)
            if local_path:
                rel_path = os.path.relpath(local_path, os.getcwd())
                return f'<img {before_src}src="{rel_path}"{after_src}>'
            else:
                return match.group(0)
        else:
            return match.group(0)

    pattern = r'!\[([^\]]*)\]\((https?://[^)]+)\)|<img ([^>]*?)src=["\"](https?://[^"\"]+)["\"]([^>]*)>'
    updated_content = re.sub(pattern, replace_image_link, content)

    if debug:
        print(f"DEBUG: [update_image_links] Image links updated.")
    
    return updated_content

def main():
    """Generate release notes for Customer-Managed ValidMind (CMVM) releases.
    
    This script generates release notes for CMVM releases by:
    1. Reading release information from GitHub
    2. Processing each release to create a release note file
    3. Handling both major releases and hotfixes
    """
    parser = argparse.ArgumentParser(description='Generate CMVM release notes')
    parser.add_argument('--tag', help='Specific tag to process (examples: cmvm/25.05, 25.05.02, 25.05.02-rc1)')
    parser.add_argument('--overwrite', action='store_true',
                      help='Overwrite existing release note files')
    parser.add_argument('--debug', action='store_true',
                      help='Show debug output')
    parser.add_argument('--edit', action='store_true',
                      help='Edit content using OpenAI')
    parser.add_argument('--internal', action='store_true',
                      help='Include all PR labels, not just category labels')
    parser.add_argument('--download-assets', action='store_true',
                      help='Download images and videos from URLs to local files')
    args = parser.parse_args()

    try:
        # Show minimal startup info
        print("Generating CMVM release notes...\n")
        
        # Use .env location in repository root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        env_location = os.path.join(repo_root, ".env")
        
        # Setup OpenAI client (always needed for classify_section and is_merge_pr)
        env_location = get_env_location()
        api_key = setup_openai_api(env_location)
        global client
        client = openai.OpenAI(api_key=api_key)

        # Get release information from GitHub
        version = args.tag if args.tag else None
        releases, _ = get_releases_from_github(version=version, debug=args.debug)
        
        if not releases:
            print("ERROR: No releases found")
            sys.exit(1)
            
        # Process releases (check tags and create files)
        # Create a new empty set for seen_versions
        seen_versions = set()
        process_releases(releases, args.overwrite, seen_versions, debug=args.debug, version=args.tag, edit=args.edit, single=False, internal=args.internal, download_assets=args.download_assets)
            
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

