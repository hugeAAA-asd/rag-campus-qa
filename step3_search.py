"""第 3 天：只做检索，不调大模型。

这一步是排错的关键——如果这里找不准，后面大模型再聪明也答不对。

运行：
  python step3_search.py 学校奖学金申请条件是什么
  python step3_search.py            (用默认问题)
"""
import sys

from rag_core import search

question = " ".join(sys.argv[1:]).strip() or "学校奖学金申请条件是什么"

print(f"问题：{question}\n")
hits = search(question)

for rank, (score, text) in enumerate(hits, 1):
    print(f"===== 第 {rank} 名   相似度 {score:.4f} =====")
    print(text.replace("\n", " ")[:300])
    print()

print("自检：")
print("  · 如果第 1 名就是正确答案所在的条款 —— 检索没问题")
print("  · 如果第 1 名跑偏了 —— 调 chunk_size / overlap，或把 top_k 调大")