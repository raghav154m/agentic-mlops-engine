import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# We are exclusively using the high-capacity chat models verified on your account
ALLOWED_MODELS = [
    "openai/gpt-oss-20b",    # Very fast, excellent at coding
    "openai/gpt-oss-120b",   # Massive capacity, highly intelligent
    "groq/compound"          # Groq's high-tier routing model
]


def get_llm(temperature: float = 0.0, max_tokens: int = 4096, model_index: int = 0) -> ChatGroq:
    """
    Returns a ChatGroq instance using verified models from your specific account tier.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please check .env file.")

    # Select model based on index to allow failover
    selected_model = ALLOWED_MODELS[model_index % len(ALLOWED_MODELS)]
    print(f"🤖 [LLM Engine] Active Model Selected: {selected_model}")

    return ChatGroq(
        model=selected_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )