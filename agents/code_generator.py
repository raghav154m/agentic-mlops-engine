import os
import re
import json
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
    
    # 3. Filter out accidental markdown lines that leak into python files
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("* ", "- ", "### ", "## ", "**Note:", "Note:")):
            continue
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines).strip()


class CodeGeneratorAgent:
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
            ("system", """You are an expert Production ML Engineer.
Convert the provided JSON Data Strategy Plan into standard, modular, top-to-bottom executable Python code.

RULES:
1. Output ONLY executable Python code inside a ```python ... ``` code block.
2. Absolutely NO conversational text, NO bullet points (* or -), and NO markdown explanations inside or outside the code block.
3. Load the dataset from: {dataset_path}
4. Perform preprocessing, model training, evaluation, and save the model to 'artifacts/model.joblib' using joblib.
5. Keep all code strictly formatted with 4 spaces of indentation."""),
            ("user", """Dataset Path: {dataset_path}
Data Strategy Plan:
{strategy_json}

Write the complete Python script based on the strategy above.""")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def generate_code(self, dataset_path: str, strategy: dict) -> str:
        """Generates raw Python code from the strategy JSON."""
        raw_response = self.chain.invoke({
            "dataset_path": dataset_path,
            "strategy_json": json.dumps(strategy, indent=2)
        })

        return clean_extracted_code(raw_response)

    def save_script(self, code: str, output_path: str = "sandbox_workspace/generated_pipeline.py") -> str:
        """Saves generated code into workspace."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        return output_path