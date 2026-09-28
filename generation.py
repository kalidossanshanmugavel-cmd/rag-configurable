import requests
from openai import OpenAI
from google import genai

from config import CONFIG, LLM_API_KEY


def generate_answer(prompt):
    """Generate the final answer using the configured LLM provider."""

    provider = CONFIG["llm_provider"]
    model_name = CONFIG["llm_model"]

    match provider:

        case "ollama":
            response = requests.post(
                CONFIG["ollama_url"],
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )

            response.raise_for_status()

            return response.json()["response"]

        case "openai":
            if not LLM_API_KEY:
                raise ValueError("OPENAI_API_KEY is not configured")

            client = OpenAI(api_key=LLM_API_KEY)

            response = client.responses.create(
                model=model_name,
                input=prompt
            )

            return response.output_text

        case "gemini":
            if not LLM_API_KEY:
                raise ValueError("GEMINI_API_KEY is not configured")

            client = genai.Client(api_key=LLM_API_KEY)

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            return response.text

        case _:
            raise ValueError(
                f"Unsupported LLM provider: {provider}"
            )
