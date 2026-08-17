import os
import json
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


class CodeGeneratorAgent:
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
            ("system", """You are an expert Production ML Engineer.
Write clean, concise, top-to-bottom executable Python code that builds a leakage-free Scikit-Learn pipeline.

MANDATORY RULES:
1. DATASET & TARGET:
   - Ingest: `df = pd.read_csv(r"{dataset_path}")`
   - Target: `target_col = '{target_column}'`
   - Split: `X = df.drop(columns=[target_col])` and `y = df[target_col]`
2. CLEANING:
   - Drop ID/Name/Email columns from X if present: `drop_cols = [c for c in ['Student_ID', 'id', 'Name', 'Email', 'Date'] if c in X.columns]; X = X.drop(columns=drop_cols, errors='ignore')`
   - For string/numeric cleanup, use simple vectorized operations. Example:
     `word_map = {{'ninety': '90', 'eighty': '80', 'seventy': '70'}}`
     Use `pd.to_numeric(series.astype(str).str.strip().str.replace('%','').replace(word_map), errors='coerce')`. Keep dictionaries short.
3. ZERO DATA LEAKAGE:
   - Encapsulate all imputation (SimpleImputer), scaling (StandardScaler), and encoding (OneHotEncoder) inside `Pipeline` and `ColumnTransformer`.
4. EVALUATION & METRICS:
   - For classification: Use StratifiedKFold, compute cross_val_predict, and print accuracy/classification_report.
   - For regression: Use KFold / cross_val_predict, and print RMSE and MAE.
5. SERIALIZATION: Save final fitted model using `joblib.dump(model_pipeline, 'artifacts/model.joblib')`.
6. Output ONLY pure, complete executable Python code inside ```python ... ``` without ellipses or markdown notes."""),
            ("user", """Dataset Path: {dataset_path}
Target Column: {target_column}
Strategy Plan:
{strategy_json}

Write the COMPLETE runnable Python script.""")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def generate_code(self, dataset_path: str, target_column: str, strategy: dict) -> str:
        raw_response = self.chain.invoke({
            "dataset_path": dataset_path,
            "target_column": target_column,
            "strategy_json": json.dumps(strategy, indent=2)
        })
        return clean_extracted_code(raw_response)

    def save_script(self, code: str, output_path: str = "sandbox_workspace/generated_pipeline.py") -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        return output_path