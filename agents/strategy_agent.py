import json
import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()


class StrategyAgent:
    """LLM Strategy Agent using Groq and Llama 3.1."""

    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.llm = ChatGroq(model=model_name, temperature=0.0)

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert MLOps Data Strategy Agent. "
                    "Analyze the provided JSON dataset profile and target column. "
                    "Output a structured JSON plan detailing data cleaning, feature engineering, and model selection. "
                    "Respond with ONLY a valid JSON object.",
                ),
                (
                    "user",
                    "Dataset Profile:\n{dataset_profile}\n\nTarget Column: {target_column}",
                ),
            ]
        )

    def generate_strategy(
        self, dataset_profile: Dict[str, Any], target_column: str
    ) -> Dict[str, Any]:
        formatted_prompt = self.prompt_template.format_messages(
            dataset_profile=json.dumps(dataset_profile, indent=2),
            target_column=target_column,
        )

        response = self.llm.invoke(formatted_prompt)

        try:
            content = str(response.content).strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            return json.loads(content)
        except Exception as e:
            return {
                "error": f"Failed to parse LLM strategy response: {str(e)}",
                "raw_response": str(response.content),
            }


if __name__ == "__main__":
    from profiler.dataset_profiler import DatasetProfiler

    profiler = DatasetProfiler("data/sample_dataset.csv")
    profile_data = profiler.generate_profile()

    agent = StrategyAgent()
    strategy = agent.generate_strategy(
        dataset_profile=profile_data, target_column="churn"
    )

    print("==========================================")
    print(" LLM Strategy Agent Generated Strategy    ")
    print("==========================================")
    print(json.dumps(strategy, indent=2))