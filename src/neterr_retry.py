import pandas as pd
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =========================
# 核心配置
# =========================

API_KEY = "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"
RES_FOLDER = "res"

MAX_WORKERS = 8
SLEEP_TIME = 0.3

# 真实模型列表（标准名称）
MODELS = [
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2-7b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen3-32b",
    "mistralai/mistral-medium-3.1",
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "deepseek/deepseek-chat"
]

# 构建安全文件名映射
MODEL_FILE_MAP = {
    m.replace("/", "_").replace(":", "_"): m
    for m in MODELS
}


# =========================
# Prompt 构造
# =========================

def build_prompt(option_a, option_b):
    return f"""You are evaluating academic English collocations.

Which phrase is more natural in formal academic writing?

A: {option_a}
B: {option_b}

Answer with a single capital letter only: A or B.
Do NOT provide any explanation.
Output exactly one character."""


# =========================
# API 调用
# =========================

def fetch_choice(model_id, option_a, option_b):

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You must output exactly one character: A or B."
            },
            {
                "role": "user",
                "content": build_prompt(option_a, option_b)
            }
        ],
        "max_tokens": 1,
        "temperature": 0
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            return None, f"http_{response.status_code}"

        data = response.json()

        if "error" in data:
            return None, "api_error"

        choices = data.get("choices", [])
        if not choices:
            return None, "no_choice"

        answer = choices[0]["message"]["content"].strip().upper()

        if answer == "A":
            return "A", "success"
        elif answer == "B":
            return "B", "success"
        else:
            return None, "invalid_output"

    except Exception:
        return None, "network_error"

    finally:
        time.sleep(SLEEP_TIME)


# =========================
# 自动识别模型名
# =========================

def detect_model_from_filename(filename):

    base = filename.replace("result_", "").replace(".csv", "")

    # 精确匹配
    if base in MODEL_FILE_MAP:
        return MODEL_FILE_MAP[base]

    # 模糊匹配（防止旧文件名）
    for key in MODEL_FILE_MAP:
        if base.startswith(key):
            return MODEL_FILE_MAP[key]

    return None


# =========================
# 重试单个文件
# =========================

def retry_file(filepath):

    filename = os.path.basename(filepath)
    model_id = detect_model_from_filename(filename)

    if model_id is None:
        print(f"⚠️ Cannot detect model for {filename}, skipping.")
        return

    print(f"\n🔁 Processing: {filename}")
    print(f"Model: {model_id}")

    df = pd.read_csv(filepath)

    # 只要不是 success 都重试
    failed_mask = df["status"] != "success"
    failed_df = df[failed_mask]

    if failed_df.empty:
        print("✅ No rows to retry.")
        return

    print(f"Found {len(failed_df)} rows to retry.")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_index = {
            executor.submit(
                fetch_choice,
                model_id,
                row["option_a"],
                row["option_b"]
            ): idx
            for idx, row in failed_df.iterrows()
        }

        for future in tqdm(as_completed(future_to_index),
                           total=len(future_to_index),
                           desc=filename):

            idx = future_to_index[future]
            choice, status = future.result()

            df.at[idx, "model_choice"] = choice if choice else ""
            df.at[idx, "status"] = status

            if choice:
                df.at[idx, "is_correct"] = (
                    1 if choice == df.at[idx, "ground_truth"] else 0
                )
            else:
                df.at[idx, "is_correct"] = 0

    # 覆盖保存
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print("💾 File updated.")


# =========================
# 主入口
# =========================

if __name__ == "__main__":

    if not os.path.exists(RES_FOLDER):
        print("❌ res folder not found.")
        exit()

    files = [f for f in os.listdir(RES_FOLDER) if f.endswith(".csv")]

    if not files:
        print("❌ No CSV files found.")
        exit()

    for file in files:
        retry_file(os.path.join(RES_FOLDER, file))

    print("\n🎉 Retry process completed.")
