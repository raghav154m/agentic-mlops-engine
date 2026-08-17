import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class StrategyAgent:
    def __init__(self, model_name: str = "qwen/qwen3.6-27b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")
            
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.1,
            api_key=api_key
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Data Scientist and MLOps Engineer.
Analyze the dataset profile and create a concrete, production-grade preprocessing, feature engineering, and modeling strategy.

CRITICAL: Return ONLY valid JSON wrapped in ```json ... ``` without conversational commentary.

Schema:
{{
  "data_strategy": {{
    "data_cleaning": {{
      "steps": [
        {{
          "step": "string",
          "description": "string",
          "column_names": ["string"]
        }}
      ]
    }},
    "feature_engineering": {{
      "steps": [
        {{
          "step": "string",
          "description": "string",
          "column_names": ["string"]
        }}
      ]
    }},
    "model_selection": {{
      "steps": [
        {{
          "step": "string",
          "description": "string",
          "column_names": ["string"]
        }}
      ]
    }}
  }}
}}"""),
            ("user", """Dataset Profile:
{dataset_profile}

Target Column: {target_column}

Formulate the data transformation and modeling strategy.""")
        ])

    def generate_strategy(self, dataset_profile: dict, target_column: str) -> dict:
        formatted_prompt = self.prompt_template.format_messages(
            dataset_profile=json.dumps(dataset_profile, indent=2),
            target_column=target_column
        )
        
        response = self.llm.invoke(formatted_prompt)
        text = str(response.content).strip()
        
        # Remove think tags if present
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
            
        # Extract json content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        else:
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace != -1:
                text = text[first_brace:last_brace + 1]
        
        try:
            return json.loads(text)
        except Exception as e:
            return {
                "error": f"Failed to parse LLM strategy response: {str(e)}",
                "raw_response": str(response.content)
            }


if __name__ == "__main__":
    from profiler.dataset_profiler import DatasetProfiler

    profiler = DatasetProfiler("data/sample_dataset.csv")
    profile_data = profiler.generate_profile()

    agent = StrategyAgent()
    strategy = agent.generate_strategy(
        dataset_profile=profile_data,
        target_column="churn"
    )

    print("=========================================")
    print(" LLM Strategy Agent Generated Strategy ")
    print("=========================================")
    print(json.dumps(strategy, indent=2))