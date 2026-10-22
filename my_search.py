"""自己实现 search()。

写完运行：python my_search.py
如果输出三块、第 1 名是「奖学金」相关内容，就成功了。
"""
import json

from rag_core import INDEX_FILE, cosine, embed_texts


def search(question, top_k=3):
    """返回 [(相似度, 原文块), ...]，按相似度从高到低，最多 top_k 个。"""

    # ── 第 1 步：读 index.json ──────────────────────────────
    # 提示：INDEX_FILE.read_text(encoding="utf-8") 得到文字
    #       json.loads(文字) 得到字典
    data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))

    # ── 第 2 步：从字典里取出 chunks 列表 ──────────────────
    # 提示：字典["chunks"]
    chunks = data["chunks"]

    # ── 第 3 步：把 question 变成向量 ──────────────────────
    # 提示：embed_texts([question], is_query=True)[0]
    q_vec = embed_texts([question],is_query=True)[0]

    # ── 第 4 步：对每个块算相似度，收集成 (分数, 文本) ────
    # 提示：先 scored = []，再 for c in chunks: 里算 cosine 并 append
    scored = [(cosine(q_vec,c["embedding"]),c["text"])for c in chunks]
    # ── 第 5 步：从大到小排序 ──────────────────────────────
    # 提示：scored.sort(key=lambda x: x[0], reverse=True)
    scored.sort(key=lambda x: x[0], reverse=True)
    # ── 第 6 步：返回前 top_k 个 ───────────────────────────
    # 提示：return scored[:top_k]
    return scored[:top_k]


if __name__ == "__main__":
    hits = search("学校奖学金申请条件是什么")
    if not hits:
        print("search() 什么都没返回 —— 是不是还没写？")
    else:
        for i, (score, text) in enumerate(hits, 1):
            print(f"第{i}名  {score:.4f}  {text[:45]}")