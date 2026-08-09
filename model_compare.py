import os
import time
import asyncio
import re
from typing import Optional

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

# TEMP DEBUG — remove once confirmed working. Prints only whether each
# key was found (and its last 4 chars), never the full key.
def _mask(k):
    return f"...{k[-4:]}" if k and len(k) > 4 else "MISSING/EMPTY"

print("MODEL_COMPARE DEBUG - GEMINI_API_KEY:", _mask(GEMINI_API_KEY))
print("MODEL_COMPARE DEBUG - GROQ_API_KEY:", _mask(GROQ_API_KEY))
print("MODEL_COMPARE DEBUG - COHERE_API_KEY:", _mask(COHERE_API_KEY))

# Adjust model names here if your provider account has access to
# different/newer models than these defaults.
GEMINI_MODEL = "gemini-3.1-flash-lite"  # more established than 3.5-flash — less likely to hit capacity-overload 503s
GROQ_MODEL = "llama-3.3-70b-versatile"
COHERE_MODEL = "command-a-03-2025"  # was "command-r-plus" — retired in Cohere's April 2026 deprecation wave

REQUEST_TIMEOUT_SECONDS = 30


class ModelCompareRequest(BaseModel):
    job_post: str
    platform: Optional[str] = "Upwork"
    skill: Optional[str] = "Web Dev"
    tone: Optional[str] = "Professional"


def build_prompt(req: ModelCompareRequest) -> str:
    return (
        f"You are a freelance proposal writer. Write a {req.tone.lower()} "
        f"proposal (under 200 words) for this {req.platform} job post, "
        f"highlighting {req.skill} expertise where relevant.\n\n"
        f"Job post:\n{req.job_post}"
    )


def score_proposal(text: str, job_post: str) -> float:
    """
    Lightweight heuristic scorer (1-10) — no extra API call/cost.
    Rewards: being close to the ideal ~150-220 word length, mentioning
    keywords from the job post, and including a call-to-action.
    Swap this for an LLM-as-judge call later if you want higher-fidelity
    quality scoring (e.g. have Gemini rate all 3 outputs 1-10).
    """
    if not text:
        return 0.0

    words = text.split()
    word_count = len(words)

    # Length score: peaks around 150-220 words, tapers off outside that.
    ideal_low, ideal_high = 150, 220
    if ideal_low <= word_count <= ideal_high:
        length_score = 10
    else:
        distance = min(abs(word_count - ideal_low), abs(word_count - ideal_high))
        length_score = max(0, 10 - distance / 15)

    # Keyword overlap: how many distinct job-post keywords appear in the proposal.
    job_keywords = set(re.findall(r"[a-zA-Z]{4,}", job_post.lower()))
    proposal_words = set(re.findall(r"[a-zA-Z]{4,}", text.lower()))
    overlap = len(job_keywords & proposal_words)
    keyword_score = min(10, overlap * 0.8)

    # Call-to-action bonus: does it end with something inviting a reply?
    cta_terms = ["let's", "happy to", "look forward", "available", "schedule", "discuss"]
    cta_score = 10 if any(term in text.lower() for term in cta_terms) else 5

    final = round((length_score * 0.4) + (keyword_score * 0.4) + (cta_score * 0.2), 1)
    return max(0.0, min(10.0, final))


async def call_gemini(client: httpx.AsyncClient, prompt: str) -> dict:
    start = time.perf_counter()
    # Gemini returns 503 when Google's servers are temporarily overloaded
    # (common right after a new model launches) — this is not a config
    # problem, so a couple of quick retries usually clears it.
    max_attempts = 3
    last_error = None

    for attempt in range(max_attempts):
        try:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"model": "Gemini", "proposal": text, "speed_ms": round((time.perf_counter() - start) * 1000), "error": None}
        except httpx.HTTPStatusError as err:
            last_error = err
            if err.response.status_code == 503 and attempt < max_attempts - 1:
                await asyncio.sleep(1.5 * (attempt + 1))  # 1.5s, then 3s
                continue
            break
        except Exception as err:
            last_error = err
            break

    return {"model": "Gemini", "proposal": "", "speed_ms": round((time.perf_counter() - start) * 1000), "error": str(last_error)}


async def call_groq(client: httpx.AsyncClient, prompt: str) -> dict:
    start = time.perf_counter()
    try:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return {"model": "Groq (Llama 3.3)", "proposal": text, "speed_ms": round((time.perf_counter() - start) * 1000), "error": None}
    except Exception as err:
        return {"model": "Groq (Llama 3.3)", "proposal": "", "speed_ms": round((time.perf_counter() - start) * 1000), "error": str(err)}


async def call_cohere(client: httpx.AsyncClient, prompt: str) -> dict:
    start = time.perf_counter()
    try:
        resp = await client.post(
            "https://api.cohere.com/v1/chat",
            headers={"Authorization": f"Bearer {COHERE_API_KEY}"},
            json={"model": COHERE_MODEL, "message": prompt},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("text", "").strip()
        return {"model": "Cohere (Command R+)", "proposal": text, "speed_ms": round((time.perf_counter() - start) * 1000), "error": None}
    except Exception as err:
        return {"model": "Cohere (Command R+)", "proposal": "", "speed_ms": round((time.perf_counter() - start) * 1000), "error": str(err)}


@router.post("/model-compare")
async def model_compare(req: ModelCompareRequest):
    prompt = build_prompt(req)

    async with httpx.AsyncClient() as client:
        # Parallel calls — total time is roughly the SLOWEST provider,
        # not the sum of all three.
        gemini_result, groq_result, cohere_result = await asyncio.gather(
            call_gemini(client, prompt),
            call_groq(client, prompt),
            call_cohere(client, prompt),
        )

    results = [gemini_result, groq_result, cohere_result]
    for r in results:
        r["score"] = score_proposal(r["proposal"], req.job_post) if not r["error"] else 0.0

    successful = [r for r in results if not r["error"]]
    winner = max(successful, key=lambda r: r["score"])["model"] if successful else None

    return {"results": results, "winner": winner}