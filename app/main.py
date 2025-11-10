import asyncio
import json
from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime, timezone

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, MessageHandler, ContextTypes, filters
from telegram.helpers import escape_markdown

from config import TELEGRAM_BOT_TOKEN, PROMPT_PATH
from tts import synth_dialogue_to_mp3
from openai_client import ChatGPTAgent

# ─────────────────────────────────────────────────────────────────────────────
# System prompt + JSON schema (строгий формат ответа)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = Path(PROMPT_PATH).read_text(encoding="utf-8")


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "jp_teacher_bot_response",
        "schema": {
            "type": "object",
            "properties": {
                "Student": {
                    "type": "string",
                    "description": "Видимое ученику: объяснения, задания/тесты, либо разбор его ответов с корректировками и мини-списком новых слов."
                },
                "Bot": {
                    "type": "object",
                    "description": "Служебная информация для бота (не показывается ученику). ДОЛЖНА быть максимально строгой и короткой.",
                    "properties": {
                        "level": {
                            "type": "string",
                            "description": "Текущая оценка уровня ученика.",
                            "enum": ["N5", "N4", "N3", "N2", "N1"]
                        },
                        "score": {
                            "type": "integer",
                            "description": "Оценка прогресса 0..1000 (чем выше, тем ближе к N1).",
                            "minimum": 0,
                            "maximum": 1000
                        },
                        "audio_script": {
                            "type": "string",
                            "description": "Текст диалога для аудирования. Формат реплик: 'A: ...\\nB: ...'. Включи хотя бы одно слово-маркер: audio / jp-audio / audio-script."
                        },
                        "tech_stats": {
                            "type": "string",
                            "description": "Необязательные заметки учителя о прогрессе, планах, истории уроков."
                        }
                    },
                    "additionalProperties": False
                }
            },
            "required": ["Student", "Bot"],
            "additionalProperties": False
        },
        "strict": True
    }
}

# Инициализация агента (использует .env внутри openai_client.py)
agent = ChatGPTAgent()

telegram_user_id = 91738308
try:
    agent.delete_chat(str(telegram_user_id))
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Папка для персистентных данных студентов
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STUDENTS_DIR = BASE_DIR / "data" / "students"
STUDENTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_TG_TEXT = 4000  # чуть меньше реального лимита


async def reply_student_text(update, text: str):
    if not text:
        await update.message.reply_text("Пустое поле Student.")
        return
    # безопасно экранируем под MarkdownV2
    safe = escape_markdown(text, version=2)
    # режем по блокам, чтобы не превысить лимит
    for i in range(0, len(safe), MAX_TG_TEXT):
        chunk = safe[i:i+MAX_TG_TEXT]
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN_V2)

def student_dir(user_id: Union[int, str]) -> Path:
    p = STUDENTS_DIR / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def save_score(user_id: Union[int, str], score: int):
    p = student_dir(user_id) / "score.json"
    payload = {"score": int(score), "updated_at_utc": utc_now_iso()}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def append_tech_stats(user_id: Union[int, str], tech_stats: str):
    if not tech_stats.strip():
        return
    p = student_dir(user_id) / "tech_stats.txt"
    stamp = utc_now_iso()
    with p.open("a", encoding="utf-8") as f:
        f.write(f"\n--- [{stamp}] ---\n{tech_stats.strip()}\n")
    # также держим «последнюю версию» для удобства инжекта
    (student_dir(user_id) / "tech_stats_latest.txt").write_text(tech_stats.strip(), encoding="utf-8")

def load_latest_tech_stats(user_id: Union[int, str]) -> str:
    p = student_dir(user_id) / "tech_stats_latest.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""

def should_inject_tech_stats_today(user_id: Union[int, str]) -> bool:
    flag = student_dir(user_id) / "last_tech_stats_sent_at.txt"
    if not flag.exists():
        return True
    try:
        last_iso = flag.read_text(encoding="utf-8").strip()
        last_dt = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
    except Exception:
        return True
    now_dt = datetime.now(timezone.utc)
    return (now_dt.date() != last_dt.date())

def mark_tech_stats_sent_now(user_id: Union[int, str]):
    (student_dir(user_id) / "last_tech_stats_sent_at.txt").write_text(utc_now_iso(), encoding="utf-8")

def inject_daily_tech_stats(user_text: str, user_id: Union[int, str]) -> str:
    """
    Раз в день (UTC) прикладываем к сообщению техстатс,
    если он сохранён и ещё не отправлялся сегодня.
    """
    if not should_inject_tech_stats_today(user_id):
        return user_text
    latest = load_latest_tech_stats(user_id)
    if not latest:
        return user_text
    stamp = utc_now_iso()
    injected = (
        f"{user_text}\n\n"
        f"[BOT_TECH_STATS_UTC {stamp}]\n"
        f"{latest}"
    )
    mark_tech_stats_sent_now(user_id)
    return injected

def script_to_dialogue_list(script: str) -> List[Dict[str, str]]:
    dialogue = []
    for raw in script.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            sp, jp = line.split(":", 1)
            dialogue.append({"speaker": sp.strip(), "jp": jp.strip()})
        else:
            dialogue.append({"speaker": "", "jp": line})
    return dialogue

# ─────────────────────────────────────────────────────────────────────────────
# Telegram: единый обработчик текстовых сообщений (бот — прокси к ассистенту)
# ─────────────────────────────────────────────────────────────────────────────
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    tg_user_id: Union[int, str] = update.effective_user.id

    # гарантируем чат для данного Telegram-пользователя
    chat_id = agent.ensure_user_chat(
        telegram_user_id=tg_user_id,
        system_prompt=SYSTEM_PROMPT,
        response_format=RESPONSE_FORMAT,
        title=f"user:{tg_user_id}"
    )

    # раз в день прикладываем tech_stats к запросу в ассистента
    user_text_for_agent = inject_daily_tech_stats(user_text, tg_user_id)

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        # отправляем запрос ассистенту
        assistant_raw = await asyncio.to_thread(agent.send_message, chat_id, user_text_for_agent)

        # пытаемся распарсить строгий JSON (по схеме)
        try:
            payload = json.loads(assistant_raw)
        except json.JSONDecodeError:
            # await update.message.reply_text(assistant_raw[:4000])
            await reply_student_text(update, assistant_raw[:MAX_TG_TEXT])
            return

        # сохраняем score / tech_stats (если пришли)
        bot_data = payload.get("Bot") or {}
        if isinstance(bot_data.get("score"), int):
            save_score(tg_user_id, int(bot_data["score"]))
        if isinstance(bot_data.get("tech_stats"), str) and bot_data["tech_stats"].strip():
            append_tech_stats(tg_user_id, bot_data["tech_stats"])

        # обработка аудирования
        audio_script = (bot_data.get("audio_script") or "").strip()
        if audio_script:
            dialogue = script_to_dialogue_list(audio_script)
            await update.message.chat.send_action(ChatAction.RECORD_VOICE)
            audio_path = await asyncio.to_thread(synth_dialogue_to_mp3, dialogue)
            await update.message.reply_audio(audio=open(audio_path, "rb"), title="Аудирование")

        # видимый студенту текст
        student_text = (payload.get("Student") or "").strip()
        if student_text:
            # await update.message.reply_text(student_text, parse_mode=ParseMode.MARKDOWN)
            await reply_student_text(update, student_text)
        else:
            await update.message.reply_text("Пустое поле Student.")

    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

def run():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling()

if __name__ == "__main__":
    run()
