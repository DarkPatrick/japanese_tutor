from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Dict, List
import os
import re

from pydub import AudioSegment
from pydub.utils import which
from contextlib import suppress

from openai_client import client
from config import OPENAI_TTS_MODEL, OPENAI_TTS_VOICE, OUT_AUDIO_DIR



FFMPEG_BIN = which("ffmpeg")
FFPROBE_BIN = which("ffprobe")

AudioSegment.converter = FFMPEG_BIN
AudioSegment.ffprobe = FFPROBE_BIN

if not FFMPEG_BIN or not FFPROBE_BIN:
    raise RuntimeError(
        "ffmpeg/ffprobe не найдены в PATH. "
        "Установи ffmpeg и убедись, что команды `ffmpeg` и `ffprobe` доступны."
    )


import re
from contextlib import suppress
from typing import Dict, List

# маппинг голосов по спикеру
SPEAKER_VOICES: Dict[str, str] = {
    "A": OPENAI_TTS_VOICE,  # из .env
    "B": "verse",           # подбери второй голос
    "C": "fable",           # можно добавить ещё
}

PAUSE_MS = 300  # пауза между репликами

def normalize_speaker_label(s: str) -> str:
    """
    Приводит метку спикера к латинской A..Z (учитывая кириллицу А/В/С и т.п.),
    берёт только первый символ.
    """
    if not s:
        return ""
    s = s.strip()
    # заменим кириллические аналоги на латиницу для A/B/C
    repl = {"А": "A", "В": "B", "С": "C", "а": "A", "в": "B", "с": "C"}
    c = repl.get(s[0], s[0])
    return c.upper()

def _pick_voice_for_speaker(speaker: str) -> str:
    key = normalize_speaker_label(speaker)
    return SPEAKER_VOICES.get(key, OPENAI_TTS_VOICE)

def prepare_tts_text(text: str) -> str:
    """
    Очищает реплику для TTS:
    - обрезает перевод после '—' или '-'
    - убирает круглые скобки и их содержимое (в т.ч. полноширинные)
    - удаляет латиницу и кириллицу (оставляя японский текст)
    - схлопывает пробелы
    """
    if not text:
        return ""
    t = text
    t = re.split(r"\s+—\s+|\s+-\s+", t, maxsplit=1)[0]
    # удалим ( ... ) и （ ... ）
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"（[^）]*）", "", t)
    t = re.sub(r"[A-Za-zА-Яа-яЁё]", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def synth_dialogue_to_mp3(dialogue: List[Dict[str, str]]) -> Path:
    """
    Для каждой реплики выбираем голос по метке спикера,
    но в TTS отправляем ТОЛЬКО японский текст без метки.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_AUDIO_DIR / f"jlpt_dialog_{ts}.mp3"

    merged = AudioSegment.silent(duration=0)
    silence = AudioSegment.silent(duration=PAUSE_MS)
    temp_files: List[str] = []

    try:
        for turn in dialogue:
            sp = (turn.get("speaker") or "").strip()
            jp = (turn.get("jp") or "").strip()
            if not jp:
                continue

            voice = _pick_voice_for_speaker(sp)
            clean_text = prepare_tts_text(jp)  # <- без "A:" / "B:" и без перевода/скобок
            if not clean_text:
                continue

            with NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                tmp_path = tf.name
            try:
                resp = client.audio.speech.create(
                    model=OPENAI_TTS_MODEL,
                    voice=voice,
                    input=clean_text
                )
                resp.write_to_file(tmp_path)
                temp_files.append(tmp_path)

                seg = AudioSegment.from_file(tmp_path, format="mp3")
                merged += seg + silence
            except Exception:
                with suppress(Exception):
                    os.remove(tmp_path)
                raise

        merged.export(out_path, format="mp3")
        return out_path
    finally:
        for p in temp_files:
            with suppress(Exception):
                os.remove(p)
