import os
import json
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

class CodeGeneratorAgent:
    def __init__(self):
        # Using our centralized LLM manager!
        self.llm = get_llm(temperature=0.0, max_tokens=4096)

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Production ML Engineer.
Write clean, concise, 100% complete Scikit-Learn pipeline code.

ABSOLUTE MANDATORY RULES:
1. NEVER USE '...' OR PLACEHOLDERS. Write every single line of code in full.
2. INGESTION & TARGET:
   df = pd.read_csv(r"{dataset_path}")
   target_col = '{target_column}'
   X = df.drop(columns=[target_col])
   y = df[target_col]
3. DROP IDENTIFIERS:
   drop_cols = [c for c in ['id', 'user_id', 'Student_ID', 'Name', 'Email', 'Date'] if c in X.columns]
   X = X.drop(columns=drop_cols, errors='ignore')
4. DATA PREPROCESSING (ZERO LEAKAGE):
   - Clean string numbers: handle '%' and convert with pd.to_numeric(errors='coerce').
   - Wrap numeric features in Pipeline(SimpleImputer(strategy='median'), StandardScaler()).
   - Wrap categorical features in Pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', drop='first')).
   - Combine with ColumnTransformer.
5. MODEL & EVALUATION:
   - For classification: Use StratifiedKFold, cross_val_predict, and print accuracy and classification_report.
   - For regression: Use KFold, cross_val_predict, and print RMSE and R2 score.
6. SERIALIZATION:
   joblib.dump(model_pipeline, 'artifacts/model.joblib')
7. Output ONLY executable Python code inside a single ```python ... ``` block."""),
            ("user", """Dataset Path: {dataset_path}
Target Column: {target_column}
Strategy JSON:
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