from pathlib import Path
from datetime import datetime
from openai_client import client
from config import OPENAI_TTS_MODEL, OPENAI_TTS_VOICE, OUT_AUDIO_DIR



def synth_dialogue_to_mp3(dialogue) -> Path:
    """
    Принимает массив реплик [{speaker,jp,...}], склеивает в единый текст и
    синтезирует аудио MP3. Возвращает путь к файлу.
    """
    # склеиваем «имя: фраза» (без перевода)
    lines = []
    for turn in dialogue:
        sp = turn.get("speaker") or ""
        jp = turn.get("jp") or ""
        line = f"{sp}: {jp}" if sp else jp
        lines.append(line)
    text = "\n".join(lines)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_AUDIO_DIR / f"jlpt_dialog_{ts}.mp3"

    # Streaming TTS (Python SDK)
    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text,
        response_format="mp3"  # mp3 удобно отдавать как audio в Telegram
    ) as stream:
        stream.stream_to_file(str(out_path))

    return out_path
