import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  

APP_TITLE = "AI Business Intelligence Agent"
UPLOAD_DIR = "uploads"
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0

os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_llm(model_name: str = MODEL_NAME, temperature: float = TEMPERATURE) -> ChatGroq:
    """Create a ChatGroq LLM instance. Requires GROQ_API_KEY to already be set in the environment."""
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return ChatGroq(model=model_name, temperature=temperature)


# python -m pip install -r requirements.txt
# python -m streamlit run main.py
