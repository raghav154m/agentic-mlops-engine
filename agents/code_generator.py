import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class CodeGeneratorAgent:
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
            ("system", """You are an expert Production ML & Data Engineer.
Your task is to convert a JSON-based Data Strategy Plan into executable, robust Python code.

CRITICAL REQUIREMENTS:
1. Return ONLY pure executable Python code inside a standard markdown code block: ```python ... ```.
2. Do NOT add conversational prose, explanations, or commentary.
3. The script must be completely self-contained and run end-to-end without interactive prompts.
4. Load the dataset from: {dataset_path}
5. Use pandas, numpy, and scikit-learn for all data cleaning, transformations, and model training.
6. Ensure standard print statements are included to track pipeline progress and final evaluation metrics."""),
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
        
        # Clean think tags if emitted by reasoning model
        text = str(raw_response)
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        
        # Extract code inside markdown blocks safely
        if "```python" in text:
            text = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
            
        return text.strip()

    def save_script(self, code: str, output_path: str = "sandbox_workspace/generated_pipeline.py") -> str:
        """Saves generated code directly into the sandbox workspace."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        return output_path


if __name__ == "__main__":
    from profiler.dataset_profiler import DatasetProfiler
    from agents.strategy_agent import StrategyAgent

    dataset_path = "data/sample_dataset.csv"
    
    print(" 1. Profiling Dataset...")
    profiler = DatasetProfiler(dataset_path)
    profile = profiler.generate_profile()
    
    print(" 2. Formulating Strategy...")
    strategy_agent = StrategyAgent()
    strategy = strategy_agent.generate_strategy(dataset_profile=profile, target_column="churn")
    
    print(" 3. Generating Pipeline Code...")
    coder = CodeGeneratorAgent()
    generated_code = coder.generate_code(dataset_path=dataset_path, strategy=strategy)
    
    saved_file = coder.save_script(generated_code)
    print(f"\n✅ Python Pipeline Generated and Saved to: {saved_file}\n")
    print("--- Generated Code Snippet ---")
    print(generated_code[:400] + "\n... [truncated] ...")