import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class DebuggerAgent:
    def __init__(self, model_name: str = "qwen/qwen3.6-27b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.llm = ChatGroq(
            model=model_name,
            temperature=0.1,
            api_key=api_key
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Python Debugger and ML Engineer.
Your task is to fix broken Python machine learning code based on the runtime error trace.

CRITICAL REQUIREMENTS:
1. Return ONLY pure executable Python code inside a standard markdown block: ```python ... ```.
2. Fix all syntax errors, unclosed parentheses, missing imports, data type mismatches, and column key errors.
3. Do NOT add conversational prose, explanations, or analysis outside the markdown block.
4. Ensure the entire script is complete and runnable end-to-end."""),
            ("user", """--- BROKEN CODE ---
{broken_code}

--- ERROR / STACK TRACE ---
{error_logs}

Provide the complete, corrected Python script.""")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def fix_code(self, broken_code: str, error_logs: str) -> str:
        """Invokes the LLM to patch the broken code using the runtime error."""
        raw_response = self.chain.invoke({
            "broken_code": broken_code,
            "error_logs": error_logs
        })

        text = str(raw_response)
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        if "```python" in text:
            text = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        return text.strip()

    def save_script(self, code: str, output_path: str = "sandbox_workspace/generated_pipeline.py") -> str:
        """Saves corrected code back into the workspace."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        return output_path