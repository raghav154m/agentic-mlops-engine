import json
import re
from agents.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def extract_json_safely(raw_text: str) -> dict:
    text = str(raw_text).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    return {
        "task_summary": {
            "target_column": "target",
            "problem_type": "classification",
            "dropped_identifiers": []
        },
        "data_strategy": {
            "data_cleaning": {
                "steps": [{"step": "Handle Missing Values", "column_names": [], "description": "Impute missing values with median/mode."}]
            },
            "feature_engineering": {
                "steps": [{"step": "Standard Scaling & One-Hot Encoding", "column_names": [], "description": "Scale numerical and encode categoricals."}]
            },
            "model_selection": {
                "algorithm": "RandomForestClassifier",
                "cv_strategy": "StratifiedKFold",
                "evaluation_metrics": ["accuracy", "f1"]
            }
        }
    }

class StrategyAgent:
    def __init__(self):
        # Using our centralized LLM manager!
        self.llm = get_llm(temperature=0.0)

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Data Scientist. Analyze the dataset profile and return a clean strategy JSON.

RULES:
1. IDENTIFIERS: Never use columns flagged `is_identifier: true` as features.
2. PROBLEM TYPE: Continuous target -> "regression"; Categorical target -> "classification".
3. LEAKAGE: Mandate that transformations occur inside Pipeline and ColumnTransformer.

Return ONLY valid JSON matching this schema:
{{
  "task_summary": {{
    "target_column": "{target_column}",
    "problem_type": "regression" | "classification",
    "dropped_identifiers": []
  }},
  "data_strategy": {{
    "data_cleaning": {{ "steps": [{{"step": "string", "column_names": [], "description": "string"}}] }},
    "feature_engineering": {{ "steps": [{{"step": "string", "column_names": [], "description": "string"}}] }},
    "model_selection": {{ "algorithm": "string", "cv_strategy": "string", "evaluation_metrics": ["string"] }}
  }}
}}"""),
            ("user", """Target Column: {target_column}
Dataset Profile:
{profile_json}

Return the strict JSON strategy.""")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def generate_strategy(self, dataset_profile: dict, target_column: str) -> dict:
        # Compact profile JSON to minimize token footprint
        compact_profile = {
            "shape": dataset_profile.get("dataset_shape", {}),
            "columns": {
                k: {
                    "type": v.get("inferred_type"),
                    "nulls": v.get("null_percentage"),
                    "is_id": v.get("is_identifier", False)
                }
                for k, v in dataset_profile.get("columns", {}).items()
            }
        }
        
        raw_response = self.chain.invoke({
            "target_column": target_column,
            "profile_json": json.dumps(compact_profile)
        })
        return extract_json_safely(raw_response)