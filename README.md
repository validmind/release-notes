# ValidMind Release Notes Generator

This repository contains a Python script that automatically generates release notes for ValidMind's software releases. The script processes pull requests from multiple repositories and creates formatted release notes in Quarto Markdown format.

## Overview

The release notes generator:
- Processes pull requests from multiple repositories (backend, frontend, agents, documentation)
- Categorizes changes based on labels (highlights, enhancements, breaking changes, deprecations, bug fixes, documentation)
- Formats and edits content for clarity and consistency
- Generates release notes in Quarto Markdown format
- Maintains a release history table

## Features

- Automated PR processing and categorization
- Content validation and formatting
- Support for multiple repositories
- Integration with GitHub API
- Quarto Markdown output
- Release history tracking

## Requirements

- Python 3.x
- GitHub CLI (`gh`)
- Required Python packages (see `scripts/requirements.txt`):
  - python-dotenv
  - requests

## Usage

1. Ensure you have the required dependencies installed:
   ```bash
   pip install -r scripts/requirements.txt
   ```

2. Set up your GitHub authentication:
   - Install and configure the GitHub CLI
   - Ensure you have appropriate permissions to access the repositories

3. Run the script:
   ```bash
   python scripts/generate_release_notes.py
   ```

## Output

The script generates:
- Release notes in Quarto Markdown format
- A release history table
- Categorized changes based on PR labels
- Formatted and validated content

## Repository Structure

- `scripts/` - Contains the main Python script and requirements
- `site/` - Output directory for generated release notes
  - `installation/` - Contains release notes and release history
  - `releases/` - Individual release documentation

## Contributing

When contributing to this repository, please ensure:
- PR titles are clear and descriptive
- External release notes are included in PR descriptions
- Appropriate labels are applied to PRs
- Content follows the established formatting guidelines 