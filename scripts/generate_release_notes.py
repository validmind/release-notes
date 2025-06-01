# Static configuration
RELEASES_DIR = "releases"
REPOS = ["backend", "frontend", "agents", "documentation", "validmind-library"]

# Label hierarchy for organizing release notes
label_hierarchy = ["highlight", "enhancement", "breaking-change", "deprecation", "bug", "documentation"]

# Labels that should exclude PRs from release notes
EXCLUDED_LABELS = ["internal", "auto-merge"]

# Keywords that indicate a PR is an automatic merge
MERGE_KEYWORDS = ["main branch", "branch merged", "merged into", "merge main", "merge staging"]

label_to_category = {
    "highlight": "### Release highlights",
    "enhancement": "### Enhancements",
    "breaking-change": "### Breaking changes",
    "deprecation": "### Deprecations",
    "bug": "### Bug fixes",
    "documentation": "### Documentation"
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
    "What and why",
    "Release notes",
    "Dependencies",
    "Screenshots"
]

EXCLUDED_SECTIONS = [
    "Checklist",
    "Deployment",
    "Review",
    "Testing",
    "Internal"
]

# --- Editing prompt and static editing info ---
EDIT_TITLE_PROMPT = (
    "Edit the following PR title for release notes:\n"
    "- Remove any ticket numbers, branch names, prefixes or double quotes (e.g., '9870:', 'hotfix:', 'nibz/cherry pick/1420', etc.).\n"
    "- Remove any 'Title:' prefix if present.\n"
    "- Enclose technical terms (words with underscores or file extensions like .py, .lock, etc.) in backticks.\n"
    "- Use sentence-style capitalization (capitalize only the first word and proper nouns).\n"
    "- Make the title clear and concise for end users.\n"
    "- Limit the title to 80 characters or less.\n"
    "- Use the PR body for context if needed.\n\n"
    "{title}\n"
    "{body}"
)

# --- Content editing instructions ---
EDIT_CONTENT_INSTRUCTIONS = (
    "When editing content:\n"
    "- Remove all template headings (e.g., 'What', 'Why', 'External Release Notes', 'Breaking Changes', 'Screenshots/Videos', 'PR Summary')\n"
    "- Maintain the original meaning and technical accuracy.\n"
    "- Use clear, professional language suitable for end users.\n"
    "- Format technical terms and code references consistently, using backticks for code and file names.\n"
    "- Keep the content concise and focused on user-facing changes.\n"
    "- Uppercase all acronyms (e.g., 'LLM', 'API', 'UI', 'REST').\n"
    "- Ensure proper names are spelled correctly (e.g., 'ValidMind', 'GitHub', 'OpenAI').\n"
    "- Follow Quarto's Markdown formatting: add blank lines between all block elements (headings, lists, paragraphs, code blocks, callouts, etc.).\n"
    "- Ensure a space after each list marker (e.g., '- Item').\n"
    "- Start each list item with a capital letter and use appropriate punctuation.\n"
    "- Do not leave empty sections or headings without a clear placeholder comment.\n"
    "- Do not refer to the 'PR body' or 'PR summary' in the content.\n"
    "- DO NOT add, remove, or modify any comment tags (<!-- ... --> or <!--- ... --->), and ensure they remain on their own lines.\n"
    "- Do not add any new sections, images, or headings that are not present in the original content. Only rephrase or clarify existing content; do not invent or supplement with additional examples, screenshots, or headings."
)

# --- Content validation instructions ---
VALIDATION_INSTRUCTIONS = (
    "You are a judge evaluating the quality of edited content.\n"
    "For {content_type}, check if the edit:\n"
    "1. Maintains the core meaning and facts\n"
    "2. Uses proper formatting and structure\n"
    "3. Is clear and professional\n"
    "4. Doesn't add unsupported information\n"
    "5. For titles: Is properly capitalized and punctuated\n"
    "6. For summaries/notes: Has proper paragraph structure\n"
    "7. Does not contain any unwanted sections (Checklist, Deployment Notes, Areas Needing Special Review, etc.)\n"
    "8. Does not add any new sections, images, or headings that are not present in the original content.\n"
    "If any unwanted sections or new content are found, respond with 'FAIL: Contains unwanted or invented sections/images/headings'.\n"
    "Otherwise, respond with only 'PASS' or 'FAIL' followed by a brief reason."
)

# --- Section classification prompt ---
SECTION_CLASSIFICATION_PROMPT = (
    "Classify if this section should be included in public release notes.\n"
    "Section title: {title}\n"
    "Section content: {content}\n\n"
    "Rules:\n"
    "1. Include sections that describe user-facing changes, features, or improvements\n"
    "2. Include sections about dependencies, breaking changes, or upgrade notes\n"
    "3. Include sections with screenshots or media\n"
    "4. Exclude internal notes, checklists, deployment steps, or review points\n"
    "5. Exclude sections about testing, QA, or development processes\n"
    "6. Exclude sections marked as internal or for team use only\n\n"
    "Respond with only 'INCLUDE' or 'EXCLUDE'."
)

# --- Heading level adjustment prompt ---
HEADING_LEVEL_PROMPT = (
    "Fix heading levels in the markdown content while maintaining proper hierarchy:\n"
    "1. The first heading (PR title) should be level 3 (###)\n"
    "2. Section headings (like 'External Release Notes', 'Breaking Changes', etc.) should be level 4 (####)\n"
    "3. All other headings should be at least level 5 (#####)\n"
    "4. Preserve the relative hierarchy between headings\n"
    "5. Only change the number of # symbols at the start of headings\n"
    "6. Keep everything else exactly the same\n\n"
    "Content:\n"
    "{content}"
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
from urllib.parse import urlparse

ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

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
    
    try:
        prompt = SECTION_CLASSIFICATION_PROMPT.format(
            title=section_title,
            content=section_content
        )
        
        if debug:
            print("DEBUG: [classify_section] Sending prompt to OpenAI...")
            
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a release notes classifier. Your job is to determine if a section should be included in public release notes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=10,
            temperature=0.0
        )
        
        result = response.choices[0].message.content.strip().upper()
        if debug:
            print(f"DEBUG: [classify_section] OpenAI response: {result}")
        return result == 'INCLUDE'
        
    except Exception as e:
        if debug:
            print(f"DEBUG: [classify_section] Error in OpenAI classification: {e}")
            print("DEBUG: [classify_section] Falling back to pattern matching...")
            
        # Fallback to basic pattern matching if OpenAI fails
        normalized_title = section_title.lower()
        normalized_title = re.sub(r'[^\w\s]', '', normalized_title)
        
        if debug:
            print(f"DEBUG: [classify_section] Normalized title: {normalized_title}")
        
        # Check excluded patterns first
        for excluded in EXCLUDED_SECTIONS:
            if excluded.lower() in normalized_title:
                if debug:
                    print(f"DEBUG: [classify_section] Matched excluded pattern: {excluded}")
                return False
                
        # Then check included patterns
        for included in INCLUDED_SECTIONS:
            if included.lower() in normalized_title:
                if debug:
                    print(f"DEBUG: [classify_section] Matched included pattern: {included}")
                return True
                
        if debug:
            print("DEBUG: No patterns matched, excluding section")
        return False

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
        
        self.pr_auto_summary = None
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
        
    def extract_pr_summary_comment(self):
        """Takes the github bot's comment containing an auto-generated summary of the PR.

        Modifies:
            self.pr_auto_summary
        """
        # Run the GitHub CLI command and capture the output
        cmd = f'gh pr view {self.pr_number} --repo {self.repo_name} --comments'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            output = result.stdout
            
            # Extract the content under '# PR Summary' from comments by github-actions
            lines = output.splitlines()
            capture = False
            summary_content = []

            for line in lines:
                if "github-actions" in line:
                    capture = False
                if "# PR Summary" in line:
                    capture = True
                    continue
                if capture:
                    if "## Test Suggestions" in line:
                        break
                    summary_content.append(line)

            # Join and print the captured summary content
            summary = "\n".join(summary_content).strip()
            if summary:
                self.pr_auto_summary = summary
            else:
                print(f"No PR summary found for #{self.pr_number} from {self.repo_name}\n")
        else:
            print(f"Failed to fetch comments: {result.stderr}")

    def extract_pr_summary(self):
        """Extract the PR summary from the PR body
        
        Returns:
            str: The PR summary or None if not found
        """
        # Try new template format first
        match = re.search(r"## What and why\?\s*(.+?)(?=^## |\Z)", self.pr_body, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()
            
        # Fallback to old format
        match = re.search(r"## Summary\s*(.+?)(?=^## |\Z)", self.pr_body, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()
            
        return None

    def edit_content(self, content_type, content, editing_instructions, edit=False):
        """Unified function to edit PR content (summaries, titles, or release notes)."""
        if not edit:
            # If not editing, just use the original content
            if content_type == 'title':
                self.cleaned_title = content.rstrip('.')
            elif content_type == 'summary':
                self.pr_interpreted_summary = content
                self.edited_text = content
            elif content_type == 'notes':
                self.edited_text = content
            return

        if self.debug:
            print(f"DEBUG: [edit_content] PR #{self.pr_number} in {self.repo_name} - Editing {content_type}")
            print(f"DEBUG: [edit_content] Original content (first 100 chars): {repr(content[:100]) if content else None}")
            print(f"DEBUG: [edit_content] Editing instructions (first 100 chars): {repr(editing_instructions[:100]) if editing_instructions else None}")
        print(f"Editing {content_type} for PR #{self.pr_number} in {self.repo_name} ...")
        
        try:
            if self.debug:
                print(f"DEBUG: [edit_content] Making OpenAI API call for PR #{self.pr_number}")
            
            # Combine the specific editing instructions with the general content instructions
            full_instructions = f"{editing_instructions}\n\n{EDIT_CONTENT_INSTRUCTIONS}\n\nIMPORTANT: Maintain the scope of this specific PR (#{self.pr_number}). Do not merge content from other PRs or add information not present in the original content."
            
            # Initialize variables for retry loop
            max_attempts = 10
            initial_delay = 1
            delay = initial_delay
            max_delay = 30  # Cap maximum delay at 30 seconds
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

                # Make API call
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a release notes editor. Your job is to edit content for clarity and user-facing release notes while maintaining the original PR's scope. Follow the instructions exactly."
                        },
                        {
                            "role": "user",
                            "content": f"Instructions:\n{full_instructions_with_dedup}\n\nContent to edit:\n{content_to_edit}"
                        }
                    ],
                    max_tokens=4096,
                    temperature=temperature,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                current_edit = response.choices[0].message.content.strip()
                if self.debug:
                    print(f"DEBUG: [edit_content] Attempt {attempt + 1} content (first 100 chars): {repr(current_edit[:100]) if current_edit else None}")
                
                # Validate current attempt
                is_valid, validation_result, content_for_reedit = self.validate_edit(content_type, content, current_edit, edit)
                if self.debug:
                    print(f"DEBUG: [edit_content] Attempt {attempt + 1} validation result: {is_valid}")
                
                if is_valid:
                    edited_content = current_edit
                    self.last_validation_result = validation_result
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
                    print(f"WARN: All {max_attempts} content edit attempts failed for {content_type} in PR #{self.pr_number}")
                    if self.debug:
                        print(f"Validation result: {validation_result}")
                    print(f"Failure patterns: {failure_patterns}")
                    if content_for_reedit:
                        print(f"Content available for reedit with validation message: {content_for_reedit['validation_message']}")
                    edited_content = content  # Use original content if all attempts fail
                    self.last_validation_result = validation_result
            
            # Set the content based on type
            if content_type == 'title':
                self.cleaned_title = edited_content.rstrip('.')
                if self.debug:
                    print(f"DEBUG: [edit_content] Set cleaned_title: {self.cleaned_title}")
            elif content_type == 'summary':
                self.pr_interpreted_summary = edited_content
                self.edited_text = self.pr_interpreted_summary
                if self.debug:
                    print(f"DEBUG: [edit_content] Set pr_interpreted_summary and edited_text")
            elif content_type == 'notes':
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
            elif content_type == 'summary':
                self.pr_interpreted_summary = content
                self.edited_text = content
            elif content_type == 'notes':
                self.edited_text = content

    def validate_edit(self, content_type, original_content, edited_content, edit=False):
        """Uses LLM to validate edits by checking for common issues.
        
        Args:
            content_type (str): Type of content that was edited
            original_content (str): The original content before editing
            edited_content (str): The edited content to validate
            edit (bool): Whether to perform validation
            
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

        # Structured validation criteria based on content type
        validation_criteria = {
            'title': [
                "Is properly capitalized and punctuated",
                "Does not contain ticket numbers or branch names",
                "Is clear and concise for end users",
                "Is 80 characters or less"
            ],
            'summary': [
                "Maintains core meaning and facts",
                "Uses proper formatting and structure",
                "Is clear and professional",
                "Does not contain unwanted sections"
            ],
            'notes': [
                "Maintains technical accuracy",
                "Uses consistent formatting",
                "Is user-focused",
                "Does not contain internal notes"
            ]
        }

        # Build detailed validation prompt
        validation_prompt = VALIDATION_INSTRUCTIONS.format(content_type=content_type)
        if content_type in validation_criteria:
            validation_prompt += "\n\nSpecific criteria to check:\n"
            for criterion in validation_criteria[content_type]:
                validation_prompt += f"- {criterion}\n"

        max_retries = 10
        retry_delay = 1
        max_delay = 30  # Cap maximum delay at 30 seconds
        last_error = None

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": validation_prompt
                        },
                        {
                            "role": "user",
                            "content": f"Original content: {original_content}\n\nEdited content: {edited_content}"
                        }
                    ],
                    max_tokens=100,
                    temperature=0.0
                )
                result = response.choices[0].message.content.strip()
                
                if self.debug:
                    print(f"DEBUG: [validate_edit] Validation LLM response: {result}")
                
                # Add validation result to info
                validation_info['validation_result'] = result
                validation_info['attempt_number'] = attempt + 1
                
                # Determine if validation passed
                is_valid = result.startswith('PASS')
                
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

            except Exception as e:
                last_error = str(e)
                validation_info['error'] = last_error
                if attempt < max_retries - 1:
                    # Use a more moderate growth rate (1.5x instead of 2x)
                    retry_delay = min(max_delay, retry_delay * 1.5)
                    # Add small random jitter (±20%)
                    jitter = random.uniform(-0.2, 0.2) * retry_delay
                    sleep_time = max(0.5, retry_delay + jitter)  # Ensure minimum 0.5s delay
                    print(f"\nValidation attempt {attempt + 1} failed for {content_type} edit in PR #{self.pr_number}: {last_error}")
                    print(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"\nAll validation attempts failed for {content_type} edit in PR #{self.pr_number}. Last error: {last_error}")
                    return True, f"FAIL: Exception during validation - {last_error}", None

    def extract_dependencies(self):
        """Extract dependencies and breaking changes from the PR body
        
        Returns:
            str: The dependencies and breaking changes or None if not found
        """
        match = re.search(r"## Dependencies, breaking changes, and deployment notes\s*(.+?)(?=^## |\Z)", self.pr_body, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

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

# Initialize OpenAI client
env_location = get_env_location()
api_key = setup_openai_api(env_location)
client = openai.OpenAI(api_key=api_key)

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

def rate_limited_api_call(cmd, max_retries=3, initial_delay=1):
    """Make a rate-limited API call with exponential backoff.
    
    Args:
        cmd (list): Command to run
        max_retries (int): Maximum number of retries
        initial_delay (int): Initial delay in seconds
        
    Returns:
        tuple: (returncode, stdout, stderr)
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
            jitter = random.uniform(0, 0.1 * delay)
            sleep_time = delay + jitter
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

def adjust_heading_levels(content, min_level=4, debug=False):
    """Adjust heading levels in content using LLM or fallback to basic adjustment.
    
    Args:
        content (str): The content to adjust
        min_level (int): Minimum heading level (default: 4)
        debug (bool): Whether to show debug output
        
    Returns:
        str: Content with adjusted heading levels
    """
    if not content:
        if debug:
            print("DEBUG: [adjust_heading_levels] Empty content, returning as is")
        return content
        
    if debug:
        print(f"\nDEBUG: [adjust_heading_levels] Adjusting heading levels to minimum {min_level}")
        print(f"DEBUG: [adjust_heading_levels] Input content (first 200 chars): {content[:200]}")
    
    # Try LLM-based adjustment first
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a markdown heading level adjuster. Your job is to adjust heading levels while maintaining proper hierarchy. Follow these rules:\n1. The first heading should be level 3 (###)\n2. Section headings should be level 4 (####)\n3. Subsection headings should be level 5 (#####)\n4. Preserve the relative hierarchy between headings\n5. Only change the number of # symbols at the start of headings\n6. Keep everything else exactly the same"
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=4096,
            temperature=0.0
        )
        
        result = response.choices[0].message.content.strip()
        
        if debug:
            print(f"DEBUG: [adjust_heading_levels] LLM result (first 200 chars): {result[:200]}")
            print("\nDEBUG: [adjust_heading_levels] Heading changes:")
            for i, (in_line, out_line) in enumerate(zip(content.split('\n'), result.split('\n'))):
                if in_line != out_line and '#' in in_line:
                    print(f"  Line {i+1}:")
                    print(f"    Input:  {in_line}")
                    print(f"    Output: {out_line}")
        
        return result
        
    except Exception as e:
        if debug:
            print(f"DEBUG: [adjust_heading_levels] LLM adjustment failed: {e}")
            print("DEBUG: [adjust_heading_levels] Falling back to basic adjustment")
    
    # Fall back to basic adjustment
    lines = content.split('\n')
    result_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            result_lines.append(line)
            continue
            
        # Strip leading whitespace before checking for #
        stripped = line.lstrip()
        if stripped.startswith('#'):
            # Count leading #s
            heading_level = len(stripped) - len(stripped.lstrip('#'))
            # Get the text after the #s and any whitespace
            text = stripped[heading_level:].strip()
            if text:  # Only adjust if there's actual text
                # Add #s until we reach min_level
                new_level = max(heading_level, min_level)
                # Preserve original indentation
                indent = line[:len(line) - len(line.lstrip())]
                result_lines.append(indent + '#' * new_level + ' ' + text)
            else:
                result_lines.append(line)
        else:
            result_lines.append(line)
    
    result = '\n'.join(result_lines)
    
    if debug:
        print(f"DEBUG: [adjust_heading_levels] Basic adjustment result (first 200 chars): {result[:200]}")
        print("\nDEBUG: [adjust_heading_levels] Heading changes:")
        for i, (in_line, out_line) in enumerate(zip(lines, result_lines)):
            if in_line != out_line and '#' in in_line:
                print(f"  Line {i+1}:")
                print(f"    Input:  {in_line}")
                print(f"    Output: {out_line}")
    
    return result

def is_merge_pr(title):
    """Check if a PR title indicates it's an automatic merge PR.
    
    Args:
        title (str): The PR title to check
        
    Returns:
        bool: True if the PR appears to be an automatic merge
    """
    if not title:
        return False
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
            print(f"DEBUG: PR #{pr_number} in {repo} - Title: {pr_data.get('title')}")
            print(f"DEBUG: PR #{pr_number} in {repo} - Labels: {[label['name'] for label in pr_data.get('labels', [])]}")
            pr_url = f"https://github.com/validmind/{repo}/pull/{pr_number}"
            print(f"DEBUG: PR #{pr_number} in {repo} - URL: {pr_url}")
        
        # Get the title and check if it's a merge PR
        title = pr_data.get('title')
        if is_merge_pr(title):
            if debug:
                print(f"DEBUG: PR #{pr_number} in {repo} appears to be an automatic merge, skipping.")
            return None, None, [], title, None, []
            
        # Always build a list of label names
        labels = [label['name'] for label in pr_data.get('labels', [])]
        # Skip PRs with excluded labels but return the title
        if any(label in EXCLUDED_LABELS for label in labels):
            if debug:
                print(f"DEBUG: PR #{pr_number} in {repo} has excluded label, skipping.")
            return None, None, labels, title, None, []
        # Skip internal PRs but return the title
        if 'internal' in labels:
            if debug:
                print(f"DEBUG: PR #{pr_number} in {repo} is internal, skipping.")
            return None, None, ['internal'], pr_data.get('title'), None, []

        # Extract external release notes using section classification
        external_notes = None
        if pr_data.get('body'):
            if debug:
                print("\nDEBUG: Extracting external release notes")
                print(f"DEBUG: PR body length: {len(pr_data['body'])}")
            
            # Extract only the sections we want to keep
            sections = []
            
            # Split the body into sections with more flexible matching
            section_pattern = r"##\s*([^\n]+)\s*(.+?)(?=^##\s*|\Z)"
            matches = re.finditer(section_pattern, pr_data['body'], re.DOTALL | re.MULTILINE)
            
            for match in matches:
                section_title = match.group(1).strip()
                section_content = match.group(2).strip()
                
                if debug:
                    print(f"\nDEBUG: Found section: {section_title}")
                    print(f"DEBUG: Section content (first 200 chars): {section_content[:200]}")
                    print(f"DEBUG: Passing to adjust_heading_levels (section_content):\n{section_content}")
                # Use OpenAI to classify the section
                if classify_section(section_title, section_content, debug):
                    if debug:
                        print(f"DEBUG: Including section: {section_title}")
                    # Adjust heading levels in section content to be at least level 4
                    adjusted_content = adjust_heading_levels(section_content, min_level=4, debug=debug)
                    # Add the section with adjusted heading level
                    sections.append(f"#### {section_title}\n{adjusted_content}")
                else:
                    if debug:
                        print(f"DEBUG: Excluding section: {section_title}")
            
            # If we found any sections, combine them
            if sections:
                if debug:
                    print(f"\nDEBUG: Found {len(sections)} sections to include")
                external_notes = "\n\n".join(sections)
            else:
                if debug:
                    print("\nDEBUG: No sections found, trying fallback patterns")
                
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
                    match = re.search(pattern, pr_data['body'], re.DOTALL | re.IGNORECASE)
                    if match:
                        if debug:
                            print(f"DEBUG: Found match with pattern: {pattern}")
                            print(f"DEBUG: Passing to adjust_heading_levels (extracted_text):\n{match.group(1).strip()}")
                        extracted_text = match.group(1).strip()
                        # Adjust heading levels in extracted text
                        adjusted_text = adjust_heading_levels(extracted_text, min_level=4, debug=debug)
                        external_notes = adjusted_text
                        break

        # Extract PR summary (robust: search all comments for '# PR Summary')
        pr_summary = None
        comments = pr_data.get('comments', [])
        if debug:
            print(f"DEBUG: PR #{pr_number} in {repo} - Number of comments: {len(comments)}")
        for i, comment in enumerate(comments):
            body = comment.get('body', '')
            if debug:
                print(f"DEBUG: PR #{pr_number} in {repo} - Comment {i} full body: {body!r}")
            if "# PR Summary" in body:
                match = re.search(r"(# PR Summary\s*.+?)(?=^## |\Z)", body, re.DOTALL | re.MULTILINE)
                if match:
                    pr_summary = match.group(1).strip()
                    if debug:
                        print(f"DEBUG: Passing to adjust_heading_levels (pr_summary):\n{pr_summary}")
                    # Adjust heading levels in PR summary
                    pr_summary = adjust_heading_levels(pr_summary, min_level=4, debug=debug)
                    if debug:
                        print(f"DEBUG: PR #{pr_number} in {repo} - Found PR summary: {pr_summary[:80]!r}")
                    break
        if pr_summary is None and debug:
            print(f"DEBUG: PR #{pr_number} in {repo} - No PR summary found.")
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
            print(f"DEBUG: PR #{pr_number} in {repo} - Image URLs: {image_urls}")
            print(f"DEBUG: PR #{pr_number} in {repo} - Loom video URLs: {video_urls}")
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
    1. First looks for previous RC of same version (e.g., looks for edit_ RCs)
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
        # First try previous regular release
        for t in reversed(valid_tags[:current_idx]):
            if t['rc'] is None:
                if debug:
                    print(f"DEBUG: [get_previous_tag] {repo} previous non-RC tag for {tag}: {t['name']}")
                return t['name']
        
        # Then try RCs of current version
        for t in reversed(valid_tags[:current_idx]):
            if t['major'] == major and t['minor'] == minor and t['rc'] is not None:
                if debug:
                    print(f"DEBUG: [get_previous_tag] {repo} previous version RC for {tag}: {t['name']}")
                return t['name']
    
    # Fallback to immediate previous tag if no better match found
    if debug:
        print(f"DEBUG: [get_previous_tag] {repo} fallback previous tag for {tag}: {valid_tags[current_idx - 1]['name']}")
    return valid_tags[current_idx - 1]['name']

def get_commits_for_tag(repo, tag, debug=False):
    """Get all PRs merged between the previous tag and this tag, excluding internal PRs.
    If no previous tag is found, fall back to extracting PRs from the release or tag body.
    """
    try:
        prev_tag = get_previous_tag(repo, tag, debug=debug)
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

def generate_changelog_content(repo, tag, commits, has_release):
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
        content = f"<!--- ## {repo_name} --->\n"
        if has_release:
            content += f"<!--- Release: [{tag}]({tag_url}) --->\n"
            content += f"<!--- {compare_url} --->\n"
        else:
            content += f"<!--- Tag: [{tag}]({tag_url}) --->\n"
            content += f"<!--- {compare_url} --->\n"
        content += "<!-- No public PRs found for this release -->\n"
        return content  # <-- EARLY RETURN, nothing else is executed!
    
    # If there are public PRs, proceed as before
    content = f"## {repo_name}\n"
    if has_release:
        content += f"<!--- Release: [{tag}]({tag_url}) --->\n"
        content += f"<!--- {compare_url} --->\n\n"
    else:
        content += f"<!--- Tag: [{tag}]({tag_url}) --->\n"
        content += f"<!--- {compare_url} --->\n\n"
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
    for label in label_hierarchy + ['other']:
        if label in labeled_commits and labeled_commits[label]:
            # Skip Documentation heading if repository is documentation
            if label == 'documentation' and repo == 'documentation':
                content += "\n"
            else:
                content += f"{label_to_category.get(label, '<!-- ### Changes with no label -->')}\n\n"
            for commit in labeled_commits[label]:
                pr_url = f"https://github.com/validmind/{repo}/pull/{commit['pr_number']}"
                pr_number = commit['pr_number']
                title = commit.get('cleaned_title') or commit.get('title') or f"PR #{pr_number}"
                has_content = bool(commit.get('external_notes') or commit.get('pr_summary') or commit.get('pr_body'))
                if not has_content:
                    # Comment out the entire PR section if no content
                    content += f"\n<!--- PR #{pr_number}: {pr_url} --->\n"
                    content += f"<!--- Labels: {', '.join(commit['labels']) if commit['labels'] else 'none'} --->\n"
                    content += f"<!--- ### {title} (#{pr_number}) --->\n"
                    content += f"<!-- No release notes or summary provided. -->\n\n"
                    continue
                content += f"\n<!--- PR #{pr_number}: {pr_url} --->\n"
                content += f"<!--- Labels: {', '.join(commit['labels']) if commit['labels'] else 'none'} --->\n"
                content += f"### {title} (#{pr_number})\n\n"
                # Embed images as markdown using local paths
                for url in commit.get('image_urls', []):
                    local_path = download_image(url, tag, debug=False)
                    if local_path:
                        rel_path = os.path.relpath(local_path, os.getcwd())
                        content += f'![Image]({rel_path})\n'
                if commit['external_notes']:
                    content += f"{update_image_links(commit['external_notes'], tag, False)}\n\n"
                if commit['pr_summary']:
                    content += f"{update_image_links(commit['pr_summary'], tag, False)}\n\n"
                # If no notes or summary, use PR body as fallback
                if not commit['external_notes'] and not commit['pr_summary']:
                    if commit.get('pr_body'):
                        content += f"{update_image_links(commit['pr_body'], tag, False)}\n\n"
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

def create_release_file(release, overwrite=False, debug=False, edit=False):
    """Create a release note file for a specific version.
    
    Args:
        release: Dictionary containing release information
        overwrite: Whether to overwrite existing files
        debug: Whether to show debug output
        edit: Whether to edit content using OpenAI
    """
    version = release['version']
    date = release['date']
    
    # Convert version to file name format (e.g., cmvm/25.03.02 -> 25_03_02.qmd)
    file_version = version.split('/')[-1].replace('.', '_')
    file_name = f"{file_version}.qmd"
    
    # Determine the output directory based on whether the tag contains 'cmvm'
    if 'cmvm' in version.lower():
        output_dir = os.path.join(RELEASES_DIR, 'cmvm')
    else:
        output_dir = RELEASES_DIR
        
    file_path = os.path.join(output_dir, file_name)
    
    # Check if file exists and handle overwrite at the start
    if os.path.exists(file_path) and not overwrite:
        print(f"File {file_name} already exists. Use --overwrite to update it.")
        return
        
    # Create releases directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate content for each repository
    content = []
    repo_contents = []
    all_commits = []
    any_validated = False
    any_edited = False
    
    def process_pr(commit, repo):
        """Process a single PR's content."""
        if 'internal' not in commit.get('labels', []):
            if debug:
                print(f"DEBUG: [process_pr] Processing PR #{commit['pr_number']} in {repo}")
            pr_obj = PR(repo_name=repo, pr_number=commit['pr_number'], title=commit.get('title'), body=commit.get('pr_body'), debug=debug)
            validated = False
            edited = False
            
            if edit:
                # Edit summary if present
                if commit.get('pr_summary'):
                    if debug:
                        print(f"DEBUG: [process_pr] Editing summary for PR #{commit['pr_number']} in {repo}")
                        print(f"DEBUG: [process_pr] Summary content: {commit['pr_summary'][:200]}...")
                    pr_obj.edit_content('summary', commit['pr_summary'], "Edit this PR summary for clarity and user-facing release notes.", edit=True)
                    commit['pr_summary'] = pr_obj.pr_interpreted_summary
                    if pr_obj.validated:
                        validated = True
                        edited = True
                
                # Edit notes if present
                if commit.get('external_notes'):
                    if debug:
                        print(f"DEBUG: [process_pr] Editing notes for PR #{commit['pr_number']} in {repo}")
                        print(f"DEBUG: [process_pr] Notes content: {commit['external_notes'][:200]}...")
                    pr_obj.edit_content('notes', commit['external_notes'], "Edit these external release notes for clarity and user-facing release notes.", edit=True)
                    commit['external_notes'] = pr_obj.edited_text
                    if pr_obj.validated:
                        validated = True
                        edited = True
                
                # Always edit title, using summary/notes as context if available
                context = ''
                if commit.get('pr_summary'):
                    context += f"\nPR Summary: {commit['pr_summary']}"
                if commit.get('external_notes'):
                    context += f"\nExternal Notes: {commit['external_notes']}"
                if debug:
                    print(f"DEBUG: [process_pr] Editing title for PR #{commit['pr_number']} in {repo}")
                    print(f"DEBUG: [process_pr] Title: {commit.get('title', '')}")
                    print(f"DEBUG: [process_pr] Context: {context[:200]}...")
                title_prompt = EDIT_TITLE_PROMPT.format(title=commit.get('title', ''), body=context)
                pr_obj.edit_content('title', commit.get('title', ''), title_prompt, edit=True)
                commit['cleaned_title'] = pr_obj.cleaned_title
                if pr_obj.validated:
                    validated = True
                    edited = True
            else:
                # If not editing, just use the original content
                commit['cleaned_title'] = commit.get('title', '')
                commit['pr_summary'] = commit.get('pr_summary', '')
                commit['external_notes'] = commit.get('external_notes', '')
            
            # Update image links in content
            if commit.get('pr_summary'):
                commit['pr_summary'] = update_image_links(commit['pr_summary'], version, debug)
            if commit.get('external_notes'):
                commit['external_notes'] = update_image_links(commit['external_notes'], version, debug)
            if commit.get('pr_body'):
                commit['pr_body'] = update_image_links(commit['pr_body'], version, debug)
            
            return validated, edited
        return False, False
    
    # Process PRs concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for repo in REPOS:
            commits = get_commits_for_tag(repo, version, debug)
            # Submit all PR processing tasks
            future_to_commit = {executor.submit(process_pr, commit, repo): commit for commit in commits}
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_commit):
                validated, edited = future.result()
                if validated:
                    any_validated = True
                if edited:
                    any_edited = True
            
            all_commits.extend(commits)
            has_release = check_github_release(repo, version)
            repo_content = generate_changelog_content(repo, version, commits, has_release)
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
    
    # Determine release type based on version format
    version_parts = version.replace('cmvm/', '').split('.')
    if '-rc' in version:
        release_type = "release candidate"
        title_version = version
    elif len(version_parts) == 2:
        release_type = "release"
        title_version = version
    else:
        release_type = "hotfix release"
        title_version = version
    
    # Check if all repo_contents are in the 'no public PRs' state
    all_no_public_prs = all(
        rc.strip().startswith('<!--- ##') and 'No public PRs found for this release' in rc
        for rc in repo_contents
    )
    
    # Write the file
    with open(file_path, 'w') as f:
        f.write("---\n")
        clean_title = title_version.replace('cmvm/', '')
        f.write(f'title: "{clean_title} {release_type} notes"\n')
        if date:
            f.write(f'date: "{date}"\n')
        f.write("sidebar: validmind-installation\n")
        f.write("toc-expand: true\n")
        
        # Add metadata about editing and validation
        if edit:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            if any_edited:
                f.write(f'# Content edited by AI - {current_time}\n')
            if any_validated:
                f.write(f'# Content validated by AI - {current_time}\n')
        if overwrite:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            f.write(f'# Content overwritten from an earlier version - {current_time}\n')
            
        f.write("---\n\n")
        if all_no_public_prs:
            f.write('::: {.callout-info title="No user-facing changes in this release"}\n')
            f.write('This release includes no public-facing updates to features, bug fixes, or documentation. If you\'re unsure whether any changes affect your deployment, contact <support@validmind.com>.\n')
            f.write(':::\n\n')
        f.write("\n".join(content))
    print(f"\nCreated release file: {output_dir}/{file_name}")

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

def process_releases(releases, overwrite, seen_versions, debug=False, version=None, edit=False):
    """Process all releases and create release note files.
    
    Args:
        releases: List of release dictionaries
        overwrite: Whether to overwrite existing files
        seen_versions: Set of versions that have been seen
        debug: Whether to show debug output
        version: Specific version to process (if any)
        edit: Whether to edit content using OpenAI
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
        create_release_file(version_releases[0], overwrite, debug, edit)
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

def auto_summary(github_urls, summary_instructions, debug=False):
    """
    Processes GitHub PRs by fetching comments, extracting summaries, and converting 
    summaries to release notes based on given instructions.

    Args:
        github_urls (list): A list of GitHub URLs, each containing PR data.
        summary_instructions (str): Instructions for converting summaries to release notes.
        debug (bool): Whether to show debug output
    """
    for url in github_urls:
        for pr in url.prs:
            if pr.data_json:
                print(f"Fetching GitHub comment from PR #{pr.pr_number} of {pr.repo_name}...\n")
                pr.extract_pr_summary_comment()
                if debug:
                    print(f"DEBUG: [auto_summary] Editing summary for PR #{pr.pr_number} in {pr.repo_name}")
                pr.edit_content('summary', pr.pr_auto_summary, summary_instructions)
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

def upgrade_info(output_file):
    """
    Appends the upgrade information single-source to the end of the new release notes.

    Args:
        output_file (str): Path to the file to append.

    Returns:
        None
    """
    include_directive = "\n\n{{< include /releases/_how-to-upgrade.qmd >}}\n"

    try:
        with open(output_file, "a") as file:
            file.write(include_directive)
            print(f"Include _how-to-upgrade.qmd added to {file.name}")
    except Exception as e:
        print(f"Failed to include _how-to-upgrade.qmd to {output_file}: {e}")

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
                    f"--->\n### {pr['title']}\n",
                    f"<!--- Source: {pr['url']} --->\n\n"
                ]
                
                if pr['notes']:
                    pr_lines.append(f"{pr['notes']}\n\n")
                
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
                print(f"DEBUG: Error getting tags from {repo}: {e}")
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
                        print(f"DEBUG: Found {len(tags)} tags in {repo}")
            except Exception as e:
                if debug:
                    print(f"DEBUG: Error processing tags from {repo}: {e}")
    
    if debug:
        print(f"DEBUG: Found {len(repo_tags)} tags across all repos")
        for repo, tag in repo_tags:
            print(f"DEBUG: Tag {tag} in {repo}")
    
    if not repo_tags:
        print(f"ERROR: No tags found for version {version if version else 'any'}")
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
                            print(f"DEBUG: Added release: {release['version']} from {repo}")
            except Exception as e:
                if debug:
                    print(f"DEBUG: Error processing tag {tag} in {repo}: {e}")
    
    if not releases:
        print(f"ERROR: No releases found for version {version if version else 'any'}")
        return [], set()
    
    # Sort releases by date in descending order, with version_key as secondary sort
    releases.sort(key=lambda x: (parse_date(x['date']), version_key(x['version'])), reverse=True)
    
    if debug:
        print(f"DEBUG: Processed {len(releases)} total release(s), {len(seen_versions)} unique version(s)")
                
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
                    
                # Get release data only if needed (for additional metadata)
                release_data = None
                cmd = ['gh', 'api', f'repos/validmind/{repo}/releases/tags/{tag}']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    release_data = json.loads(result.stdout)
                    # Use release date if available (more accurate for actual release)
                    release_date = release_data.get('published_at')
                    if release_date:
                        date = datetime.datetime.fromisoformat(release_date.replace('Z', '+00:00')).strftime("%B %d, %Y")
                
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

def test_heading_conversion():
    """Test the heading conversion logic with a real PR body sample and print only changed lines."""
    test_content = """## What
This PR introduces offline feature flag support and improves the feature flags codebase by:
- Adding offline feature flag functionality through environment variables
- Adding comprehensive docstrings to all feature flag functions
- Refactoring feature flag access to use a new centralized context-aware function
- Removing unused feature flags (FLAG_REDIS_ENABLED, FLAG_CASBIN_RELOAD_ENABLED, FLAG_AUTH_CONFIG)
- Adding type hints to improve code maintainability

Before: Feature flags were only accessible through LaunchDarkly and required an active connection.
After: Feature flags can be configured through environment variables when LaunchDarkly integration is not available, with improved code documentation and type safety.

## Why
- Enables feature flag functionality in environments where LaunchDarkly integration is not possible (e.g., VM deployments)
- Improves code maintainability through better documentation and type hints
- Centralizes feature flag access through a single function to reduce code duplication
- Removes technical debt by cleaning up unused feature flags

## External Release Notes
Added support for offline feature flags configuration through environment variables, enabling feature flag functionality in environments without LaunchDarkly integration."""
    
    print("Testing heading conversion with real PR body sample...")
    print("\nInput:")
    print(test_content)
    print("\nInput lines (repr):")
    for i, line in enumerate(test_content.split('\n')):
        print(f"{i+1}: {repr(line)}")
    print("\nOutput:")
    result = adjust_heading_levels(test_content, debug=False)
    print(result)
    print("\nChanged lines:")
    input_lines = test_content.split('\n')
    output_lines = result.split('\n')
    for i, (in_line, out_line) in enumerate(zip(input_lines, output_lines)):
        if in_line != out_line:
            print(f"Line {i+1}:")
            print(f"  Input:  {in_line}")
            print(f"  Output: {out_line}")
    return result

# Playwright-based asset downloader for GitHub user-attachments
# Requires: pip install playwright && playwright install
# Example usage:
# download_with_playwright(
#     url="https://github.com/user-attachments/assets/6f2ae03c-cb9d-4ee1-bde7-016cde360fe5",
#     output_path="downloaded_image.png",
#     browser_profile_dir=None,  # Or path to your Chrome/Edge user profile for authentication
#     headless=True
# )

def download_with_playwright(url, output_path, browser_profile_dir=None, headless=True):
    """
    Download an image or video from a GitHub user-attachments URL using Playwright.
    Args:
        url (str): The asset URL to download
        output_path (str): Where to save the downloaded file
        browser_profile_dir (str, optional): Path to a user profile directory for authentication (if needed)
        headless (bool): Whether to run browser in headless mode
    Returns:
        bool: True if download succeeded, False otherwise
    
    Note: This function is defined after __main__ since it's an optional helper that is only imported
    and used when needed for downloading assets. Keeping it separate from the core functionality
    allows the script to run without Playwright installed.
    """
    try:
        from playwright.sync_api import sync_playwright
        import time
        print(f"DEBUG: [download_with_playwright] Attempting to download {url} to {output_path}")
        with sync_playwright() as p:
            browser_args = {}
            if browser_profile_dir:
                browser_args["user_data_dir"] = browser_profile_dir
                print(f"DEBUG: [download_with_playwright] Using browser profile: {browser_profile_dir}")
            browser = p.chromium.launch_persistent_context(
                user_data_dir=browser_profile_dir or "./.pw-profile",
                headless=headless
            )
            page = browser.new_page()
            print(f"DEBUG: [download_with_playwright] Navigating to URL")
            page.goto(url)
            # Wait for network and element
            page.wait_for_load_state("networkidle")
            # Try to find an <img> or <video> tag
            if page.locator("img").count() > 0:
                print(f"DEBUG: [download_with_playwright] Found image element, taking screenshot")
                img = page.locator("img").first
                img.screenshot(path=output_path)
                browser.close()
                return True
            elif page.locator("video").count() > 0:
                print(f"DEBUG: [download_with_playwright] Found video element")
                video = page.locator("video").first
                # Try to get the video src
                src = video.get_attribute("src")
                if src:
                    print(f"DEBUG: [download_with_playwright] Downloading video from source: {src}")
                    # Download the video file
                    import requests
                    r = requests.get(src, stream=True)
                    with open(output_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    browser.close()
                    return True
            # Fallback: try to download the page content (may work for direct file links)
            print(f"DEBUG: [download_with_playwright] No media elements found, trying page content fallback")
            content = page.content()
            with open(output_path, 'wb') as f:
                f.write(content.encode('utf-8'))
            browser.close()
            return True
    except Exception as e:
        print(f"[download_with_playwright] Failed to download {url}: {e}")
        return False

def download_image(url, tag, debug=False):
    """Download an image or video from a URL and save it to a tag-specific folder."""
    try:
        releases_dir = os.path.join('releases', tag)
        os.makedirs(releases_dir, exist_ok=True)
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = f"image_{hash(url)}.png"
        elif not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.mp4', '.mov', '.qt']):
            filename = f"{filename}.png"
        local_path = os.path.join(releases_dir, filename)
        headers = {}
        github_token = os.getenv('GITHUB_TOKEN')
        if github_token:
            headers['Authorization'] = f'token {github_token}'
        # Try original URL first
        if debug:
            print(f"DEBUG: [download_image] Trying original URL: {url}")
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            if debug:
                print(f"DEBUG: [download_image] Successfully downloaded file to {local_path}")
            return local_path
        except requests.exceptions.RequestException as e:
            if debug:
                print(f"DEBUG: [download_image] Failed to download with original URL: {e}")
        # Fallback: try rewriting old user-attachments URLs
        if 'github.com/user-attachments/' in url and '/assets/' not in url:
            asset_id = url.split('/')[-1]
            raw_url = f"https://raw.githubusercontent.com/validmind/user-attachments/main/{asset_id}"
            if debug:
                print(f"DEBUG: [download_image] Trying fallback raw URL: {raw_url}")
            try:
                response = requests.get(raw_url, headers=headers, stream=True)
                response.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                if debug:
                    print(f"DEBUG: [download_image] Successfully downloaded file to {local_path} (fallback)")
                return local_path
            except requests.exceptions.RequestException as e:
                if debug:
                    print(f"DEBUG: [download_image] Fallback also failed: {e}")
        # Playwright fallback for user-attachments URLs
        if 'github.com/user-attachments/' in url:
            if debug:
                print(f"DEBUG: [download_image] Trying Playwright fallback for {url}")
            try:
                # Import here to avoid dependency if not needed
                from scripts.generate_release_notes import download_with_playwright
            except ImportError:
                # If running as a script, download_with_playwright is already in scope
                pass
            output_path = local_path
            # Try Playwright download
            success = download_with_playwright(url, output_path, browser_profile_dir=None, headless=True)
            if success:
                if debug:
                    print(f"DEBUG: [download_image] Playwright: Successfully downloaded file to {output_path}")
                return output_path
            else:
                if debug:
                    print(f"DEBUG: [download_image] Playwright failed for {url}")
        return None
    except Exception as e:
        if debug:
            print(f"DEBUG: [download_image] Failed to download image/video from {url}: {e}")
        return None

def update_image_links(content, tag, debug=False):
    """Update image links in markdown content to use local paths.
    
    Args:
        content (str): Markdown content containing image links
        tag (str): Tag name to use for folder organization
        debug (bool): Whether to show debug output
        
    Returns:
        str: Updated markdown content with local image paths
    """
    if not content:
        return content

    def replace_image_link(match):
        # Markdown image
        if match.group(1) is not None and match.group(2) is not None:
            alt_text = match.group(1)
            url = match.group(2)
            local_path = download_image(url, tag, debug)
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
            local_path = download_image(url, tag, debug)
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
    args = parser.parse_args()

    try:
        # Show minimal startup info
        print("Generating CMVM release notes...\n")
        
        # Use .env location in repository root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        env_location = os.path.join(repo_root, ".env")
        
        # Setup OpenAI if editing is enabled
        if args.edit:
            api_key = setup_openai_api(env_location)
            # Initialize OpenAI client with API key
            openai_client = openai.OpenAI(api_key=api_key)
            # Make client available globally
            global client
            client = openai_client

        # Get release information from GitHub
        version = args.tag if args.tag else None
        releases, _ = get_releases_from_github(version=version, debug=args.debug)
        
        if not releases:
            print("ERROR: No releases found")
            sys.exit(1)
            
        # Process releases (check tags and create files)
        # Create a new empty set for seen_versions
        seen_versions = set()
        process_releases(releases, args.overwrite, seen_versions, debug=args.debug, version=args.tag, edit=args.edit)
            
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

