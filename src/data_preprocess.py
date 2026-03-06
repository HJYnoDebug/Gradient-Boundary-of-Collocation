import pandas as pd
import requests
import json
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- 配置 ---
API_KEY = "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"
MODEL_NAME = "openai/gpt-4o-mini"
INPUT_FILE = "../data/ColloCaidCollocationErrorsDB.xlsx"
OUTPUT_FILE = "llm_collocation_analysis_results.xlsx"
MAX_WORKERS = 10  # 线程数


def clean_text(text):
    """清理短语中的末尾标点"""
    if pd.isna(text): return ""
    return re.sub(r'[,\.]$', '', str(text)).strip()


def get_prob(logprob):
    """将对数概率转为线性概率"""
    return math.exp(logprob) if logprob is not None else 0


def fetch_logprobs(prompt):
    """调用 OpenRouter 并提取 A/B 概率"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "logprobs": True,
        "top_logprobs": 10,
        "max_tokens": 1,
        "temperature": 0
    }

    try:
        # 增加超时时间防止个别线程阻塞
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
        data = response.json()

        # 检查是否有错误返回
        if 'error' in data:
            return None

        top_lp = data['choices'][0]['logprobs']['content'][0]['top_logprobs']
        results = {"A": 0.0, "B": 0.0}
        for item in top_lp:
            token = item['token'].strip().upper()
            if token in ["A", "B"]:
                results[token] = get_prob(item['logprob'])
        return results
    except Exception:
        return None


def process_row(index, row):
    """单行处理逻辑，供线程池调用"""
    prob_phrase = clean_text(row['problem'])
    solu_phrase = clean_text(row['solution'])
    err_type = row['problem type']

    # 随机化 A/B 位置
    is_solu_a = random.choice([True, False])
    opt_a = solu_phrase if is_solu_a else prob_phrase
    opt_b = prob_phrase if is_solu_a else solu_phrase

    prompt = (
        f"In formal academic writing, which expression is more natural and appropriate?\n"
        f"A: {opt_a}\n"
        f"B: {opt_b}\n"
        f"Answer with only the letter A or B."
    )

    probs = fetch_logprobs(prompt)

    if probs:
        p_solu = probs["A"] if is_solu_a else probs["B"]
        p_prob = probs["B"] if is_solu_a else probs["A"]

        return {
            "index": index,  # 用于后续排序
            "type": err_type,
            "problem": prob_phrase,
            "solution": solu_phrase,
            "p_solution": p_solu,
            "p_problem": p_prob,
            "delta_p": p_solu - p_prob,
            "is_correct": 1 if p_solu > p_prob else 0
        }
    return None


# --- 主程序 ---
if __name__ == "__main__":
    df = pd.read_excel(INPUT_FILE)
    results_list = []

    print(f"🚀 开始并发处理 | 模型: {MODEL_NAME} | 线程数: {MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 创建任务列表
        futures = {executor.submit(process_row, i, row): i for i, row in df.iterrows()}

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                results_list.append(result)

    # 按照原始索引排序，保证数据整齐
    final_df = pd.DataFrame(results_list).sort_values(by="index").drop(columns=["index"])

    # 保存结果
    final_df.to_excel(OUTPUT_FILE, index=False)

    # 打印简单摘要
    avg_acc = final_df['is_correct'].mean()
    print(f"\n✅ 分析完成！")
    print(f"📊 平均识别准确率: {avg_acc:.2%}")
    print(f"💾 结果已保存至: {OUTPUT_FILE}")