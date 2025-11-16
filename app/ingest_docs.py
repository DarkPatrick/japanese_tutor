# scripts/ingest_docs.py
from pathlib import Path
from dotenv import dotenv_values
from openai import OpenAI
from contextlib import ExitStack


DOCS_DIR = Path("app/docs")


def main():
    secrets = dotenv_values(".env")
    client = OpenAI(api_key=secrets["OPENAI_API_KEY"])

    # 1) создаём векторное хранилище
    vs = client.vector_stores.create(name="JP Teacher CSVs")
    print("Vector store created:", vs.id)

    # 2) собираем все CSV
    files = list(DOCS_DIR.glob("*.csv"))
    if not files:
        print("No CSV files found in app/docs")
        return

    # 3) пробуем батч-загрузку (новый SDK)
    try:
        # ВАЖНО: держим файлы открытыми, пока идёт upload_and_poll
        with ExitStack() as stack:
            file_handles = [stack.enter_context(open(str(p), "rb")) for p in files]
            batch = client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vs.id,
                files=file_handles,
            )
        print(f"Uploaded {len(files)} files via file_batches to", vs.id)

    except Exception as e:
        # 4) fallback: по одному файлу
        print("file_batches unavailable, fallback to per-file upload:", e)
        for p in files:
            with open(str(p), "rb") as fh:
                up = client.files.create(
                    file=fh,
                    purpose="assistants"  # ← ОБЯЗАТЕЛЬНО для новых SDK
                )
            client.vector_stores.files.create(
                vector_store_id=vs.id,
                file_id=up.id
            )
            print("Uploaded:", p.name)

    # 5) подсказываем, что сохранить в .env
    print("\nSave this to your .env:")
    print(f"VECTOR_STORE_ID={vs.id}")

if __name__ == "__main__":
    main()
