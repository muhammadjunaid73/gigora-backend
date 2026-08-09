import google.generativeai as genai
import os

# Add multiple keys to avoid 429 quota errors
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]


def get_working_model():
    for key in API_KEYS:
        if not key:
            continue

        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            model.generate_content("Test")
            return model

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                continue  # Try next key
            raise e

    raise Exception("All API keys exhausted - add more keys")


def generate_proposal(job_post: str, tone: str = "professional") -> dict:
    model = get_working_model()

    response = model.generate_content(
        f"""You are an expert freelancer proposal writer.
Write a {tone} proposal for: {job_post}
Return JSON only: {{"proposal": "text", "word_count": 150}}"""
    )

    import json
    return json.loads(response.text)