from openai import OpenAI
from typing import Optional, List, Dict, Any, Union
import uuid
import json
import os
from dotenv import dotenv_values



secrets: dict = dotenv_values(".env")
OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
OPENAI_TEXT_MODEL = secrets.get("OPENAI_TEXT_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)


class ChatGPTAgent:
    def __init__(self, chats_path: str = "./chats.json"):
        # openai_api_key оставлен для совместимости интерфейса
        self.chats_path = chats_path
        self.chats = self._load_chats()

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
        response_format: Optional[dict] = None
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
            "history": []
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

    def _responses_api_call(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[dict] = None
    ) -> str:
        """
        Вызов OpenAI Responses API (совместимо с SDK без параметра `response_format`).

        Если передан `response_format` с JSON-схемой, мы НЕ передаём его как аргумент,
        а добавляем отдельное системное сообщение-инструкцию со схемой.
        Это заставляет модель вернуть строгий JSON, валидный по схеме.

        Примечание: в Responses API используется `input`, а не `messages`.
        """
        input_messages: List[Dict[str, str]] = list(messages)

        # Инжектим схему в системное сообщение, если она есть
        if response_format and isinstance(response_format, dict):
            if response_format.get("type") == "json_schema" and "json_schema" in response_format:
                try:
                    schema_obj = response_format["json_schema"]
                    schema_text = json.dumps(schema_obj, ensure_ascii=False)
                except Exception:
                    schema_text = str(response_format["json_schema"])

                schema_instruction = (
                    "You MUST return a single JSON object that VALIDATES against the following JSON Schema. "
                    "Return ONLY the raw JSON (no code fences, no extra text, no markdown):\n"
                    f"{schema_text}"
                )
                # Вставляем дополнительный system-промт ПЕРЕД имеющимися сообщениями
                input_messages = [{"role": "system", "content": schema_instruction}] + input_messages

        # Создаём ответ через Responses API
        resp = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            input=input_messages
        )

        # Надёжный способ получить итоговый текст
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()

        # Фолбэк: собрать текст по частям
        chunks: List[str] = []
        if hasattr(resp, "output"):
            for out in resp.output:
                if getattr(out, "type", None) == "message":
                    for item in getattr(out.message, "content", []) or []:
                        if getattr(item, "type", None) == "output_text":
                            chunks.append(item.text)
        return "\n".join(chunks).strip()


    def send_message(self, chat_id: str, user_message: str) -> str:
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

        reply_content = self._responses_api_call(
            model=OPENAI_TEXT_MODEL,
            messages=messages,
            response_format=response_format
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
