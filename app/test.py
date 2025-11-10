from pathlib import Path
from typing import Dict, Any
from openai_client import client
from config import PROMPT_PATH, OPENAI_TEXT_MODEL



JLPT_SCHEMA = {
    "name": "jlpt_audio_test",
    "schema": {
        "type": "object",
        "properties": {
            "level": {"type": "string", "description": "JLPT level like N5..N1"},
            "dialogue": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "jp": {"type": "string"},
                        "romaji": {"type": "string"},
                        "ru": {"type": "string"}
                    },
                    "required": ["speaker", "jp"]
                }
            },
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "choices": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 4
                        },
                        "answer_index": {"type": "integer"},
                        "explanation_ru": {"type": "string"}
                    },
                    "required": ["q", "choices", "answer_index"]
                }
            }
        },
        "required": ["level", "dialogue", "questions"]
    },
    "strict": True
}

def _read_system_prompt() -> str:
    return Path(PROMPT_PATH).read_text(encoding="utf-8")

def build_jlpt_test(level: str = "N5") -> Dict[str, Any]:
    """
    Генерирует JLPT-подобный тест с опорой на PDF-файлы.
    """
    system_prompt = _read_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Сгенерируй короткий японский диалог уровня {level} "
                "на бытовую тему (2-3 говорящих, 6-10 реплик), потом составь 3-5 вопросов "
                "как в JLPT (множественный выбор). "
                "В диалоге сохрани японский текст (jp), при возможности добавь romaji и перевод ru. "
                "Строго выведи JSON по заданной схеме."
            ),
        },
    ]

    # Responses API с File Search: прикрепляем vector_store
    resp = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        messages=messages,
        tools=[{"type": "file_search"}],
        response_format={
            "type": "json_schema",
            "json_schema": JLPT_SCHEMA
        }
    )

    # Вытаскиваем JSON из первого output_message
    for out in resp.output:
        if out.type == "message":
            for item in out.message.content:
                if item.type == "output_text":
                    import json
                    return json.loads(item.text)
    raise RuntimeError("Не удалось получить JSON теста")
