import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# OpenAI SDK (Responses API)
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-mini")
PROMPT_PATH = os.getenv("PROMPT_PATH", "prompt.txt")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")

# Read system prompt once at startup
try:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTIONS = f.read().strip()
except FileNotFoundError:
    raise RuntimeError(f"Prompt file not found at: {PROMPT_PATH}")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="AI Moderation API", version="1.0.0")


# ---------- Schemas ----------
class InPayload(BaseModel):
    text: str = Field(..., description="원문 텍스트")


class OutPayload(BaseModel):
    hate: Literal[0, 1]
    spam: Literal[0, 1]


# ---------- Helpers ----------
WORD_HATE = re.compile(r"\bhate\b", flags=re.IGNORECASE)
WORD_SPAM = re.compile(r"\bspam\b", flags=re.IGNORECASE)
WORD_BENIGN = re.compile(r"\bbenign\b", flags=re.IGNORECASE)

def truncate(s: str, limit: int = 1000) -> str:
    return s[:limit] if len(s) > limit else s

def postprocess_to_flags(output_text: str) -> OutPayload:
    """
    규칙:
    - 출력에 'hate'가 들어가 있으면 hate=1
    - 출력에 'spam'이 들어가 있으면 spam=1
    - 그 외: 둘 다 0
    """
    txt = (output_text or "").strip().lower()

    hate = 1 if WORD_HATE.search(txt) else 0
    spam = 1 if WORD_SPAM.search(txt) else 0

    if hate == 0 and spam == 0:
        if WORD_BENIGN.search(txt):
            return OutPayload(hate=0, spam=0)
        return OutPayload(hate=0, spam=0)

    return OutPayload(hate=hate, spam=spam)


# ---------- Routes ----------
@app.post("/classify", response_model=OutPayload)
def classify(payload: InPayload):
    user_text = truncate(payload.text, 1000)

    try:
        # Responses API 호출
        resp = client.responses.create(
            model=MODEL_NAME,
            instructions=SYSTEM_INSTRUCTIONS,   # 시스템 역할 프롬프트
            input=user_text,                    # 사용자 입력
            max_output_tokens=16,               # 한 단어만 기대
        )
        output_text = resp.output_text or ""

        '''
        print("GPT Output: ")
        print(output_text)
        print("--------------------")
        '''
        
    except Exception as e:
        # 외부 API 에러
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    # Parsing -> hate/spam flag
    result = postprocess_to_flags(output_text)
    return result
