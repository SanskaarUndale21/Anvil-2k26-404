"""Shared Gemini client -- import from here so we init once."""
from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, GEMINI_MODEL

client = genai.Client(api_key=GOOGLE_API_KEY)


def generate(system: str, user: str) -> str:
    """Simple wrapper: returns response text."""
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.4,
        ),
    )
    return resp.text.strip()
