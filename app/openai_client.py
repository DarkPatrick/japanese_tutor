from openai import OpenAI
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import uuid
import json
import os
from dotenv import dotenv_values



secrets: dict = dotenv_values(".env")
OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
# OPENAI_TEXT_MODEL = secrets.get("OPENAI_TEXT_MODEL", "gpt-4o-mini")
OPENAI_TEXT_MODEL = secrets.get("OPENAI_TEXT_MODEL", "gpt-5")
STUDENTS_DIR = secrets.get("STUDENTS_DIR", "students")


client = OpenAI(api_key=OPENAI_API_KEY)


def _abs_students_dir() -> Path:
    # базируемся от корня проекта (где лежат твои .py)
    # если запускаешь из другого cwd — Path(__file__) защитит
    project_root = Path(__file__).resolve().parents[1]  # подними на уровень выше, если файлы в app/
    return (project_root / STUDENTS_DIR).resolve()


class ChatGPTAgent:
    def __init__(self, chats_path: str = "./chats.json"):
        # openai_api_key оставлен для совместимости интерфейса
        self.chats_path = chats_path
        self.chats = self._load_chats()
        self.client = client
        self.vector_store_id = secrets.get("VECTOR_STORE_ID")
        self.global_vector_store_id = secrets.get("VECTOR_STORE_ID")


    # --- Персональная векторка по юзеру ---

    def _student_dir(self, tg_user_id: Union[int, str]) -> Path:
        p = _abs_students_dir() / str(tg_user_id)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _get_or_create_user_vs(self, tg_user_id: Union[int, str]) -> str:
        uid = str(tg_user_id)
        meta = self.chats.setdefault("__users__", {})
        if uid in meta and meta[uid].get("vector_store_id"):
            return meta[uid]["vector_store_id"]

        vs = self.client.vector_stores.create(name=f"jp_teacher_student_{uid}")
        meta[uid] = {"vector_store_id": vs.id}
        self._save_chats()
        return vs.id

    def _upload_single_file_to_vs(self, vs_id: str, file_path: Path):
        # удалить старые файлы из VS
        try:
            files = self.client.vector_stores.files.list(vector_store_id=vs_id)
            for f in getattr(files, "data", []) or []:
                self.client.vector_stores.files.delete(vector_store_id=vs_id, file_id=f.id)
        except Exception:
            # если листинг недоступен — просто продолжим, загрузим свежий
            pass

        # загрузить новый файл
        if not file_path.exists():
            print(f"[VS] skip upload: file not found: {file_path}")
            return

        with open(file_path, "rb") as fh:
            up = self.client.files.create(file=fh, purpose="assistants")
        self.client.vector_stores.files.create(vector_store_id=vs_id, file_id=up.id)
        print(f"[VS] uploaded {file_path.name} to {vs_id} (file_id={up.id})")

    def sync_user_stats_to_vs(self, tg_user_id: Union[int, str]) -> str:
        """
        Перезаливает stats.json (или stats.json) студента в его персональный Vector Store.
        Возвращает user_vs_id.
        """
        user_vs_id = self._get_or_create_user_vs(tg_user_id)
        base = self._student_dir(tg_user_id)
        path = base / ("stats.json")
        if not path.exists():
            # fallback на другой формат, если нужного нет
            alt = base / ("stats.json")
            if alt.exists():
                path = alt
        if not path.exists():
            # нет файла — просто вернуть VS (без файлов)
            print(f"[VS] no stats file to upload for user {tg_user_id} (looked for {path})")
            return user_vs_id

        self._upload_single_file_to_vs(user_vs_id, path)
        return user_vs_id

    # ===== Работа с чатами =====

    def _load_chats(self) -> Dict[str, Dict]:
        if os.path.exists(self.chats_path):
            with open(self.chats_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def _save_chats(self):
        with open(self.chats_path, "w", encoding="utf-8") as f:
            json.dump(self.chats, f, ensure_ascii=False, indent=2)

    def create_chat(
        self,
        chat_id: Optional[str] = None,
        title: str = "",
        description: str = "",
        system_prompt: str = "",
        response_format: Optional[dict] = None,
        vector_store_id: Optional[str] = None
    ) -> str:
        """
        Создаёт новый чат. Единоразово задаются:
          - system_prompt (строка)
          - response_format (dict) — например JSON-схема для Responses API:
                {
                  "type": "json_schema",
                  "json_schema": {
                    "name": "my_schema",
                    "schema": {...},
                    "strict": True
                  }
                }
            Или можно передавать None для обычного текстового ответа.
        """
        if chat_id is None:
            chat_id = str(uuid.uuid4())
        if chat_id in self.chats:
            raise ValueError(f"Чат с id {chat_id} уже существует!")

        self.chats[chat_id] = {
            "title": title,
            "description": description,
            "system_prompt": system_prompt or "",
            "response_format": response_format or None,
            "history": [],
            "vector_store_id": self.vector_store_id
        }
        self._save_chats()
        print(f"Создан чат с id: {chat_id}, title: '{title}'")
        return chat_id

    def ensure_user_chat(
        self,
        telegram_user_id: Union[int, str],
        system_prompt: str,
        response_format: Optional[dict] = None,
        title: str = "",
        description: str = ""
    ) -> str:
        """
        Гарантирует существование чата для данного Telegram user id.
        Если чата нет — создаёт его с переданными system_prompt и response_format.
        Возвращает chat_id (строка с самим user_id).
        """
        chat_id = str(telegram_user_id)
        if chat_id not in self.chats:
            return self.create_chat(
                chat_id=chat_id,
                title=title or f"user:{chat_id}",
                description=description,
                system_prompt=system_prompt,
                response_format=response_format
            )
        return chat_id

    def delete_chat(self, chat_id: str):
        if chat_id in self.chats:
            del self.chats[chat_id]
            self._save_chats()
            print(f"Чат {chat_id} удалён.")
        else:
            print(f"Чат {chat_id} не найден.")

    def get_chat_history(self, chat_id: str) -> Optional[List[Dict[str, str]]]:
        chat = self.chats.get(chat_id)
        return chat["history"] if chat else None

    def list_chats(self) -> List[Dict[str, str]]:
        """Вернуть список чатов с их id, title и description."""
        return [
            {
                "chat_id": cid,
                "title": cdata.get("title", ""),
                "description": cdata.get("description", "")
            }
            for cid, cdata in self.chats.items()
        ]

    def search_chats(self, query: str) -> List[Dict[str, str]]:
        """Поиск по chat_id / title / description."""
        results = []
        query_lower = query.lower()
        for cid, cdata in self.chats.items():
            title = cdata.get("title", "")
            desc = cdata.get("description", "")
            if (query_lower in cid.lower() or
                query_lower in title.lower() or
                query_lower in desc.lower()):
                results.append({
                    "chat_id": cid,
                    "title": title,
                    "description": desc
                })
        return results

    def clear_chat_history(self, chat_id: str):
        """Очистить историю чата без удаления чата."""
        if chat_id in self.chats:
            self.chats[chat_id]["history"] = []
            self._save_chats()
            print(f"История чата {chat_id} очищена.")
        else:
            print(f"Чат {chat_id} не найден.")

    def export_chat_to_txt(self, chat_id: str, filename: Optional[str] = None):
        """Экспортировать чат в текстовый файл."""
        if chat_id not in self.chats:
            print(f"Чат {chat_id} не найден.")
            return

        chat = self.chats[chat_id]
        history = chat.get("history", [])
        system_prompt = chat.get("system_prompt", "")
        response_format = chat.get("response_format")

        if filename is None:
            filename = f"{chat_id}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Chat ID: {chat_id}\n")
            f.write(f"Title: {chat.get('title','')}\n")
            f.write(f"Description: {chat.get('description','')}\n\n")
            if system_prompt:
                f.write("=== SYSTEM PROMPT ===\n")
                f.write(system_prompt + "\n\n")
            if response_format:
                f.write("=== RESPONSE FORMAT ===\n")
                f.write(json.dumps(response_format, ensure_ascii=False, indent=2) + "\n\n")
            for msg in history:
                f.write(f"{msg['role'].upper()}: {msg['content']}\n\n")

        print(f"Чат {chat_id} экспортирован в {filename}")

    # ===== Взаимодействие с агентом =====
    def _collect_output_chunks(resp) -> str:
        chunks: List[str] = []
        if hasattr(resp, "output"):
            for out in getattr(resp, "output") or []:
                if getattr(out, "type", None) == "message":
                    for item in getattr(out.message, "content", []) or []:
                        if getattr(item, "type", None) == "output_text":
                            chunks.append(item.text)
        return "\n".join(chunks).strip()

    def _responses_api_call(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[dict] = None,
        # vector_store_id: Optional[str] = None,
        vector_store_ids: list[str] | None = None
    ) -> str:
        """
        Вызов OpenAI Responses API с кросс-совместимостью:
        1) Сначала пробуем tools=[{"type":"file_search","vector_store_ids":[VS]}]  ← то, что требует твой сервер (400: tools[0].vector_store_ids)
        2) Если не прошло — ретраем старым способом:
        tools=[{"type":"file_search"}] + attachments на последнем user-сообщении.
        JSON-схему (если есть) инжектим в отдельный system-инструктаж.
        """
        input_messages = list(messages)

        # Инжект схемы как system-инструкции (без response_format аргумента)
        if response_format and isinstance(response_format, dict):
            if response_format.get("type") == "json_schema" and "json_schema" in response_format:
                try:
                    schema_text = json.dumps(response_format["json_schema"], ensure_ascii=False)
                except Exception:
                    schema_text = str(response_format["json_schema"])
                schema_instruction = (
                    "You MUST return a single JSON object that VALIDATES against the following JSON Schema. "
                    "Return ONLY the raw JSON (no code fences, no extra text, no markdown):\n"
                    f"{schema_text}"
                )
                input_messages = [{"role": "system", "content": schema_instruction}] + input_messages

        # Попробуем VS из аргумента или из self/chats
        # vs_id = vector_store_id or getattr(self, "vector_store_id", None)
        vs_ids = vector_store_ids or []
        if not vs_ids:
            # без векторки — обычный вызов
            resp = client.responses.create(model=model, input=input_messages)
            return getattr(resp, "output_text", "").strip() or _collect_output_chunks(resp)

        # === Попытка A: современный формат tools с vector_store_ids на самом tool ===
        try:
            resp = client.responses.create(
                model=model,
                input=input_messages,
                tools=[{"type": "file_search", "vector_store_ids": vs_ids}],
            )
            return getattr(resp, "output_text", "").strip() or _collect_output_chunks(resp)
        except Exception as e_a:
            print("OLD WAY: ", e_a)
            # Если сервер старый/иной — fallback на attachments к последнему user
            # Подготовим копию сообщений с attachments
            im2 = list(input_messages)
            for i in range(len(im2) - 1, -1, -1):
                if im2[i].get("role") == "user":
                    msg = dict(im2[i])
                    # msg["attachments"] = [{"vector_store_id": vs_ids}]
                    msg["attachments"] = [{"vector_store_id": vs_id} for vs_id in vs_ids]
                    im2[i] = msg
                    break

            try:
                resp = client.responses.create(
                    model=model,
                    input=im2,
                    tools=[{"type": "file_search"}],
                )
                return getattr(resp, "output_text", "").strip() or _collect_output_chunks(resp)
            except Exception as e_b:
                # Оба пути не сработали — пробрасываем первую ошибку для дебага
                raise e_a


    def send_message(self, chat_id: str, user_message: str, tg_user_id: str | int | None = None) -> str:
        """
        Отправка сообщения агенту в рамках чата:
         - Берёт system_prompt и response_format из настроек чата
         - Отправляет history + новое сообщение
         - Возвращает ответ ассистента (строкой)
         - История чата обновляется
        """
        if chat_id not in self.chats:
            raise ValueError(f"Чат {chat_id} не существует. Создайте чат через create_chat().")

        chat_data = self.chats[chat_id]
        # vs_id = chat_data.get("vector_store_id") or self.vector_store_id
        history = chat_data["history"]
        system_prompt = chat_data.get("system_prompt", "")
        response_format = chat_data.get("response_format", None)

        # Собираем messages: system + история + новое сообщение user
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # История — как есть (user/assistant)
        messages.extend(history)

        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)

        user_vs_id = self._get_or_create_user_vs(tg_user_id)
        vs_ids = [user_vs_id] + ([self.global_vector_store_id] if self.global_vector_store_id else [])

        reply_content = self._responses_api_call(
            model=OPENAI_TEXT_MODEL,
            messages=messages,
            response_format=response_format,
            # vector_store_id=vs_id
            vector_store_ids=vs_ids
        )

        # Обновляем историю чата
        history.append(user_msg)
        history.append({"role": "assistant", "content": reply_content})
        self._save_chats()

        return reply_content

    # ===== Совместимость со старым методом =====

    def chat(self, chat_id: str, user_message: str, n_results: int = 3) -> str:
        """
        Backward-compatible обёртка поверх send_message().
        Параметр n_results оставлен для совместимости, не используется.
        """
        return self.send_message(chat_id, user_message)
