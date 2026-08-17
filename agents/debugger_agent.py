import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def clean_extracted_code(raw_text: str) -> str:
    """Strips think tags, markdown code blocks, and stray commentary lines."""
    text = str(raw_text)
    
    # 1. Remove reasoning / think blocks
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    
    # 2. Extract content inside ```python ... ```
    if "```python" in text:
        text = text.split("```python")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    lines = text.strip().split("\n")
    cleaned_lines = []
    
    # 3. Filter out accidental markdown lines
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("* ", "- ", "### ", "## ", "**Note:", "Note:")):
            continue
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines).strip()


class DebuggerAgent:
    def __init__(self, model_name: str = "qwen/qwen3.6-27b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.llm = ChatGroq(
            model=model_name,
            temperature=0.0,
            api_key=api_key
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Python Debugger and ML Engineer.
Fix broken Python ML code based on the runtime error trace.

CRITICAL CODE RULES:
1. Return ONLY pure executable Python code inside a ```python ... ``` block.
2. Remove any conversational prose, markdown bullets (* or -), or commentary lines that caused syntax errors.
3. Fix all indentation, unclosed literals, syntax, and KeyError bugs directly.
4. Ensure all necessary library imports (pandas, numpy, sklearn, joblib) are present at the top.
5. Standard 4-space indentation throughout."""),
            ("user", """--- BROKEN CODE ---
{broken_code}

--- ERROR / STACK TRACE ---
{error_logs}

Provide the complete, corrected, and runnable Python script.""")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def fix_code(self, broken_code: str, error_logs: str) -> str:
        """Invokes LLM to repair code based on error trace."""
        raw_response = self.chain.invoke({
            "broken_code": broken_code,
            "error_logs": error_logs
        })

        return clean_extracted_code(raw_response)

    def save_script(self, code: str, output_path: str = "sandbox_workspace/generated_pipeline.py") -> str:
        """Saves corrected code back into the workspace."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        return output_path