import asyncio
import json
from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime, timezone
import re
import os

from dotenv import dotenv_values
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
SECRETS = dotenv_values(".env")
STUDENTS_DIR = SECRETS.get("STUDENTS_DIR", "students")


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "jp_teacher_bot_response",
        "schema": {
            "type": "object",
            "properties": {
                "Student": {
                    "type": "string",
                    "description": "Видимое ученику: объяснения, задания/тесты, разбор, таблица новых слов."
                },
                "Bot": {
                    "type": "object",
                    "description": "Служебная информация для бота.",
                    "properties": {
                        "level": {"type": "string", "enum": ["N5","N4","N3","N2","N1"]},
                        "score": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "audio_script": {
                            "type": "string",
                            "description": "Диалог для озвучки; пустая строка, если не аудирование."
                        },
                        "tech_stats": {"type": "string"},
                        "stats": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                "level": {"type": "string", "enum": ["N5","N4","N3","N2","N1"]},
                                "type": {"type": "string", "enum": ["кандзи","лексика","грамматика","чтение","аудирование","закрепление"]},
                                "title": {"type": "string"},
                                "tries": {"type": "integer", "minimum": 0},
                                "successes": {"type": "integer", "minimum": 0},
                                "comments": {"type": "string"},
                                "kanji": {"type": "string"},
                                "word": {"type": "string"},
                                "kana": {"type": "string"},
                                "romaji": {"type": "string"},
                                "ru": {"type": "string"},
                                "source": {"type": "string", "enum": ["wasabi","generated"]}
                                },
                                "required": ["level","type","title","tries","successes","comments"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["level","score","audio_script","stats"],
                    "additionalProperties": False
                }
            },
            "required": ["Student","Bot"],
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
# STUDENTS_DIR = BASE_DIR / "data" / "students"
# STUDENTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_TG_TEXT = 4000  # чуть меньше реального лимита
RE_FENCED_AUDIO = re.compile(r"```(?:audio|jp-audio|audio-script)\s*[\s\S]*?```", re.IGNORECASE)
RE_SPEAKER_LINES = re.compile(r"^(?:[A-ZА-ЯЁ]{1,2}\s*:\s*.+)$", re.MULTILINE)


def extract_json_objects(raw: str):
    """
    Возвращает список строк, каждая из которых — один валидный JSON-объект.
    Работает по балансировке фигурных скобок на верхнем уровне.
    Игнорирует текст вне объектов.
    """
    objs = []
    depth = 0
    start = None
    in_str = False
    esc = False

    for i, ch in enumerate(raw):
        if depth == 0:
            if ch == '{':
                depth = 1
                start = i
        else:
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start is not None:
                        objs.append(raw[start:i+1])
                        start = None
    return objs

def repair_common_json_glitches(raw: str) -> str:
    s = raw

    # 1) Лишняя ] сразу после "audio_script":  ..."audio_script": "...."] , "stats"
    s = re.sub(
        r'("audio_script"\s*:\s*"(?:[^"\\]|\\.)*")\s*\]\s*,',
        r'\1,',
        s,
        flags=re.DOTALL
    )

    # 2) Запятая перед закрывающей } (висячая запятая на конце объекта)
    s = re.sub(r',\s*}', r'}', s)

    # 3) Двойные JSONы подряд без разделителя → попытаемся вытащить по фигурным скобкам
    # (основную работу у нас уже делает extract_json_objects, оставим как есть)

    return s

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

def abs_students_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]  # подняться из app/ к корню
    return (project_root / STUDENTS_DIR).resolve()

def student_dir(user_id: int | str) -> Path:
    p = abs_students_dir() / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def save_score(user_id: Union[int, str], score: int):
    p = student_dir(user_id) / "score.json"
    payload = {"score": int(score), "updated_at_utc": utc_now_iso()}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def append_stats(user_id: Union[int, str], stats: list):
    """
    Сохраняет КАЖДЫЙ элемент Bot.stats отдельной строкой в students/<id>/stats.json
    Формат строки: {"timestamp_utc": "...", "stat": {...}}
    """
    if not isinstance(stats, list):
        return
    p = student_dir(user_id) / "stats.json"
    stamp = utc_now_iso()
    with p.open("a", encoding="utf-8") as f:
        for item in stats:
            if isinstance(item, dict):
                f.write(json.dumps({"timestamp_utc": stamp, "stat": item}, ensure_ascii=False) + "\n")

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
    """
    Превращает сценарий вида:
      A: こんにちは
      B：元気です
    в [{"speaker":"A","jp":"こんにちは"}, ...].
    Допускает полноширинный двоеточие и случайные пробелы.
    """
    dialogue = []
    for raw in script.splitlines():
        line = raw.strip()
        if not line:
            continue
        # split по первому ":" или "："
        parts = re.split(r"\s*[:：]\s*", line, maxsplit=1)
        if len(parts) == 2 and parts[0]:
            sp, jp = parts[0].strip(), parts[1].strip()
        else:
            sp, jp = "", line
        dialogue.append({"speaker": sp, "jp": jp})
    return dialogue


def strip_dialogue_from_student(text: str) -> str:
    if not text:
        return text
    t = RE_FENCED_AUDIO.sub("", text)
    # Если много реплик, часто полезно убрать подряд идущие "A: ..." строки
    t = RE_SPEAKER_LINES.sub("", t)
    # подчистка лишних пустых строк
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def save_last_audio_script(user_id: Union[int, str], script: str):
    student_dir(user_id).joinpath("last_audio_script.txt").write_text(script, encoding="utf-8")
    student_dir(user_id).joinpath("awaiting_dialog_dump.flag").write_text(utc_now_iso(), encoding="utf-8")

def load_last_audio_script(user_id: Union[int, str]) -> str:
    p = student_dir(user_id) / "last_audio_script.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""

def is_awaiting_dialog_dump(user_id: Union[int, str]) -> bool:
    return (student_dir(user_id) / "awaiting_dialog_dump.flag").exists()

def clear_awaiting_dialog_dump(user_id: Union[int, str]):
    f = student_dir(user_id) / "awaiting_dialog_dump.flag"
    if f.exists():
        f.unlink()

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
        # tg_user_id = update.effective_user.id
        print("tg_user_id =", tg_user_id)
        assistant_raw = await asyncio.to_thread(agent.send_message, chat_id, user_text_for_agent, tg_user_id=tg_user_id)

        # Сначала попробуем прямой парсинг всего ответа (вдруг уже валиден)
        try:
            payload = json.loads(assistant_raw)
            objects = [assistant_raw]
        except Exception:
            # Чиним распространённые баги формата
            fixed = repair_common_json_glitches(assistant_raw)
            # Пытаемся ещё раз целиком
            try:
                payload = json.loads(fixed)
                objects = [fixed]
            except Exception:
                # Падаем на мульти-объекты: вытаскиваем каждый по балансировке скобок
                objects = extract_json_objects(fixed)

        if not objects:
            await reply_student_text(update, assistant_raw[:MAX_TG_TEXT])
            return


        # 2) обрабатываем все объекты по порядку
        for obj_str in objects:
            try:
                payload = json.loads(obj_str)
            except Exception:
                # если вдруг один из кусочков битый — покажем как текст
                await reply_student_text(update, obj_str[:MAX_TG_TEXT])
                continue

            bot_data = payload.get("Bot") or {}

            # сохраняем score / tech_stats / stats
            if isinstance(bot_data.get("score"), int):
                save_score(tg_user_id, int(bot_data["score"]))
            if isinstance(bot_data.get("tech_stats"), str) and bot_data["tech_stats"].strip():
                append_tech_stats(tg_user_id, bot_data["tech_stats"])
            stats_field = bot_data.get("stats")
            if isinstance(stats_field, list):
                append_stats(tg_user_id, stats_field)
            
            try:
                agent.sync_user_stats_to_vs(tg_user_id)
            except Exception as e:
                print(f"Failed to sync stats to VS for user {tg_user_id}: {e}")

            # аудирование (если есть)
            audio_script = (bot_data.get("audio_script") or "").strip()
            if audio_script:
                dialogue = script_to_dialogue_list(audio_script)
                await update.message.chat.send_action(ChatAction.RECORD_VOICE)
                audio_path = await asyncio.to_thread(synth_dialogue_to_mp3, dialogue)
                await update.message.reply_audio(audio=open(audio_path, "rb"), title="Аудирование")
                # remove temporary audio file
                os.remove(audio_path)

                save_last_audio_script(tg_user_id, audio_script)

            # видимая часть для ученика
            student_text = (payload.get("Student") or "").strip()
            if audio_script:
                student_text = strip_dialogue_from_student(student_text)
            # Если это уже не аудио-ответ (audio_script пуст),
            # и у нас есть «ожидание» показать исходный диалог — приложим его в конец Student
            if not (bot_data.get("audio_script") or "").strip() and is_awaiting_dialog_dump(tg_user_id):
                last_script = load_last_audio_script(tg_user_id).strip()
                if last_script:
                    # добавим аккуратно подзаголовок и диалог A:/B:
                    addendum = "\n\n**Исходный диалог:**\n" + "\n".join(
                        line if line.strip() else ""
                        for line in last_script.splitlines()
                    )
                    student_text = (student_text or "") + addendum
                clear_awaiting_dialog_dump(tg_user_id)

            await reply_student_text(update, student_text if student_text else "Пустое поле Student.")


    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")

def run():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling()

if __name__ == "__main__":
    run()
