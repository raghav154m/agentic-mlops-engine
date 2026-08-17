import os
from agents.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
        if stripped in ("...", "…") or stripped.startswith(("* ", "- ", "### ", "## ", "**Note:", "Note:")):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

class DebuggerAgent:
    def __init__(self):
        # Using our centralized LLM manager!
        self.llm = get_llm(temperature=0.0, max_tokens=4096)

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Python Debugger and ML Engineer.
Fix broken Python ML code based on the runtime error trace.

CRITICAL DEBUGGING RULES:
1. Write the COMPLETE, full-length runnable script inside ```python ... ```.
2. NEVER use ellipses '...' or placeholders anywhere.
3. PRESERVE:
   - Data loading: `df = pd.read_csv(r"{dataset_path}")`
   - Target column: `target_col = '{target_column}'`
4. Always print evaluation metrics to stdout and serialize the fitted model using joblib.dump."""),
            ("user", """Dataset Path: {dataset_path}
Target Column: {target_column}

--- BROKEN CODE ---
{broken_code}

--- ERROR / STACK TRACE ---
{error_logs}

Provide the complete, corrected, and fully written Python script.""")
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