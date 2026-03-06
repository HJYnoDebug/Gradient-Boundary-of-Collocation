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
INPUT_BENCHMARK = "academic_collocation_benchmark_v2.xlsx"

MAX_WORKERS = 5      # 不建议超过 3，避免 429
SLEEP_TIME = 0.3     # 每次请求后 sleep，进一步降低 429 风险

MODELS=["meta-llama/llama-3.1-405b-instruct"
        ]

MODELS1 = [
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-medium-3.1",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "deepseek/deepseek-chat"
]


# =========================
# 构造硬约束 Prompt
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
# API 调用（Forced Binary Choice）
# =========================

def fetch_choice(model_id, option_a, option_b):

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = build_prompt(option_a, option_b)

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You must output exactly one character: A or B."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1,
        "temperature": 0,
        "top_p": 1
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
# 主运行函数（支持断点续传）
# =========================

def run_model(model_id, df):

    print(f"\n🚀 Running model: {model_id}")

    model_safe = model_id.replace("/", "_").replace(":", "_")
    output_file = f"result_{model_safe}.csv"

    # -------------------------
    # 断点续传
    # -------------------------
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        done_indices = set(existing_df["original_index"].tolist())
        print(f"🔁 Resuming. Already done: {len(done_indices)}")
    else:
        done_indices = set()

    remaining_df = df[~df["original_index"].isin(done_indices)]

    if remaining_df.empty:
        print("✅ Already completed.")
        return

    # -------------------------
    # 多线程执行
    # -------------------------
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_row = {
            executor.submit(
                fetch_choice,
                model_id,
                row["option_a"],
                row["option_b"]
            ): row
            for _, row in remaining_df.iterrows()
        }

        for future in tqdm(
                as_completed(future_to_row),
                total=len(remaining_df),
                desc=model_id):

            row = future_to_row[future]
            choice, status = future.result()

            result_row = {
                "original_index": row["original_index"],
                "type": row.get("type", ""),
                "node": row.get("node", ""),
                "option_a": row.get("option_a", ""),
                "option_b": row.get("option_b", ""),
                "ground_truth": row.get("ground_truth", ""),
                "model_choice": choice if choice else "",
                "is_correct": 1 if choice == row["ground_truth"] else 0,
                "status": status
            }

            result_df = pd.DataFrame([result_row])

            # 动态写入，防止崩溃丢数据
            if not os.path.exists(output_file):
                result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            else:
                result_df.to_csv(
                    output_file,
                    mode="a",
                    header=False,
                    index=False,
                    encoding="utf-8-sig"
                )

    print(f"✅ Finished model: {model_id}")


# =========================
# 主入口
# =========================

if __name__ == "__main__":

    if not os.path.exists(INPUT_BENCHMARK):
        print("❌ Benchmark file not found.")
        exit()

    benchmark_df = pd.read_excel(INPUT_BENCHMARK)

    if "original_index" not in benchmark_df.columns:
        benchmark_df["original_index"] = benchmark_df.index

    for model in MODELS:
        run_model(model, benchmark_df)

    print("\n🎉 All models finished.")
