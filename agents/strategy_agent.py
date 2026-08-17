import os
import re
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def extract_json_safely(raw_text: str) -> dict:
    """Robustly extracts and parses JSON even if the LLM adds markdown or commentary."""
    text = str(raw_text).strip()
    
    # 1. Remove reasoning / think blocks
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # 2. Strip standard markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # 3. Direct JSON load attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. Regex fallback: find outermost { ... }
    json_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 5. Fallback Default Strategy if LLM returns unexpected structure
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
            ("system", """You are an expert Principal Data Scientist and MLOps Architect.
Analyze the dataset profile JSON and formulate a structured, leakage-free machine learning strategy.

CRITICAL RULES:
1. IDENTIFIERS: Never use columns flagged with `is_identifier: true` as features or target.
2. PROBLEM TYPE:
   - Continuous/numeric target -> "regression" (e.g., Ridge, RandomForestRegressor).
   - Categorical/binary target -> "classification" (e.g., LogisticRegression, RandomForestClassifier).
3. DATA LEAKAGE: Mandate that all statistical imputation, scaling, and one-hot encoding occur inside a Scikit-Learn Pipeline / ColumnTransformer.

Return ONLY a valid JSON object matching this schema:
{{
  "task_summary": {{
    "target_column": "{target_column}",
    "problem_type": "regression" | "classification",
    "dropped_identifiers": ["list_of_id_columns"]
  }},
  "data_strategy": {{
    "data_cleaning": {{
      "steps": [
        {{"step": "string", "column_names": ["list"], "description": "string"}}
      ]
    }},
    "feature_engineering": {{
      "steps": [
        {{"step": "string", "column_names": ["list"], "description": "string"}}
      ]
    }},
    "model_selection": {{
      "algorithm": "string",
      "cv_strategy": "string",
      "evaluation_metrics": ["metric1", "metric2"]
    }}
  }}
}}"""),
            ("user", """Target Column: {target_column}
Dataset Profile:
{profile_json}

Generate the strict JSON transformation strategy.""")
        ])

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def generate_strategy(self, dataset_profile: dict, target_column: str) -> dict:
        raw_response = self.chain.invoke({
            "target_column": target_column,
            "profile_json": json.dumps(dataset_profile, indent=2)
        })

        return extract_json_safely(raw_response)