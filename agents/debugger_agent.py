import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def clean_extracted_code(raw_text: str) -> str:
    text = str(raw_text)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "```python" in text:
        text = text.split("```python")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    lines = text.strip().split("\n")
    cleaned_lines = []
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
            max_tokens=4096,
            api_key=api_key
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Python Debugger and ML Engineer.
Fix broken Python ML code based on the runtime error trace.

CRITICAL DEBUGGING RULES:
1. PRESERVE CORE CONTEXT:
   - Data loading path: `df = pd.read_csv(r"{dataset_path}")`
   - Target column: `target_col = '{target_column}'`
2. FIX SYNTAX & UNCLOSED BRACKETS:
   - Ensure all dictionaries, lists, and function calls are properly closed.
   - Keep any string mapping dictionaries concise (do NOT write massive multi-line dictionaries).
3. MANDATORY METRIC PRINTS:
   - Print evaluation metrics (Accuracy/F1 or RMSE/MAE) to stdout.
4. NO PLACEHOLDERS: Output the FULL runnable script without '...' or truncation.
5. Return ONLY executable Python code inside ```python ... ```."""),
            ("user", """Dataset Path: {dataset_path}
Target Column: {target_column}

--- BROKEN CODE ---
{broken_code}

--- ERROR / STACK TRACE ---
{error_logs}

Provide the complete, corrected, fully written Python script.""")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def fix_code(self, broken_code: str, error_logs: str, dataset_path: str, target_column: str) -> str:
        raw_response = self.chain.invoke({
            "broken_code": broken_code,
            "error_logs": error_logs,
            "dataset_path": dataset_path,
            "target_column": target_column
        })
        return clean_extracted_code(raw_response)

    def save_script(self, code: str, output_path: str = "sandbox_workspace/generated_pipeline.py") -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        return output_path