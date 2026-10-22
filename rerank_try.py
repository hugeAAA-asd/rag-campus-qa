"""用 rerank 重排向量检索的结果（单题版）。

跑法：python rerank_try.py

它做三件事：
    阶段一  向量召回 10 个候选（快、糙）
    阶段二  把 10 个候选送去 rerank 精排（慢、准）
    对比    看正确答案在哪个阶段排第 1
"""
import json

import requests

from rag_core import EMBED_API_KEY, INDEX_FILE, cosine, embed_texts

# ---------------- 可调参数 ----------------
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
# 如果报 400/404（模型不存在），换成 "BAAI/bge-reranker-v2-m3" 试试
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"

QUESTION = "代替他人考试会受什么处分"   # ← 换问题改这里
RECALL_K = 10                          # 第一阶段召回几个候选
FINAL_K = 3                            # 第二阶段留几个


# ================= 阶段一：向量召回 =================
data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
chunks = data["chunks"]

q_vec = embed_texts([QUESTION], is_query=True)[0]
scored = [(cosine(q_vec, c["embedding"]), c["text"]) for c in chunks]
scored.sort(key=lambda x: x[0], reverse=True)

candidates = scored[:RECALL_K]     # [(向量分数, 文本), ...] 共 10 个

print(f"问题：{QUESTION}")
print(f"\n=== 阶段一：向量召回 top{RECALL_K} ===")
for i, (s, t) in enumerate(candidates, 1):
    print(f"  [{i:>2}] {s:.4f}  {t[:38].replace(chr(10), ' ')}")


# ================= 阶段二：送 rerank =================
resp = requests.post(
    RERANK_URL,
    headers={
        "Authorization": f"Bearer {EMBED_API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": RERANK_MODEL,
        "query": QUESTION,
        "documents": [t for _s, t in candidates],   # 只把文本传过去
        "top_n": FINAL_K,
        "return_documents": False,
    },
    timeout=60,
)

if resp.status_code != 200:
    raise SystemExit(f"rerank 失败 {resp.status_code}：{resp.text[:300]}")

result = resp.json()

# 第一次跑，先把原始响应打出来看清楚结构
print(f"\n=== rerank 返回的原始数据 ===")
for item in result["results"]:
    print("  ", item)


# ================= 映射回原文 =================
# rerank 返回的 index 是「在 documents 列表里的位置」，
# 也就是「在 candidates 列表里的位置」。所以 candidates[index] 就是那一对。
print(f"\n=== 阶段二：rerank 后的 top{FINAL_K} ===")
for rank, item in enumerate(result["results"], 1):
    idx = item["index"]                    # 在 candidates 里的位置
    score = item["relevance_score"]        # 相关性分数（越大越相关）
    text = candidates[idx][1]              # ← 关键：用 index 回去取原文
    print(f"  [{rank}] rerank={score:.4f}  {text[:38].replace(chr(10), ' ')}")