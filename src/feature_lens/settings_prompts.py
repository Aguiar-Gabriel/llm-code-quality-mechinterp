"""
Canonical prompts used to build LLM inputs for code evaluation (paper v2).

Replace the strings below with the exact canonical text from the paper
when available. Keep this module as the single source of truth.
"""

SYSTEM_PROMPT_PAPER_V2 = (
    "You are a precise code analysis assistant. Read the Java class and "
    "reason step-by-step about potential issues, style, and correctness."
)

USER_PROMPT_PAPER_V2 = (
    "Given the following Java source file, analyze its structure and behavior. "
    "Focus on clarity, potential bugs, and design trade-offs. Provide a concise assessment."
)
