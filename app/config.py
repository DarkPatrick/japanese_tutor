from dotenv import dotenv_values
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

_cfg = dotenv_values(str(BASE_DIR.parent / ".env"))

OPENAI_API_KEY = _cfg.get("OPENAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = _cfg.get("TELEGRAM_BOT_TOKEN", "")
OPENAI_TEXT_MODEL = _cfg.get("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
OPENAI_TTS_MODEL = str(_cfg.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"))
OPENAI_TTS_VOICE = str(_cfg.get("OPENAI_TTS_VOICE", "alloy"))

PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.txt"
PDF_DIR = BASE_DIR / "data" / "pdfs"
CACHE_PATH = BASE_DIR / "data" / "kb_cache.json"
OUT_AUDIO_DIR = BASE_DIR / "data" / "out_audio"

OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

print(PROMPT_PATH)
print(TELEGRAM_BOT_TOKEN)