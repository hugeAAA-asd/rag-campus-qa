"""第 4 天：检索 + 拼 prompt + 调 DeepSeek 生成回答。

运行：
  python step4_ask.py 学校奖学金申请条件是什么
  python step4_ask.py            (用默认问题)
"""
import sys

from rag_core import ask_llm, search

question = " ".join(sys.argv[1:]).strip() or "学校奖学金申请条件是什么"

print(f"问题：{question}\n")
print("① 检索中 ...")
hits = search(question)
for rank, (score, text) in enumerate(hits, 1):
    print(f"   [{rank}] 相似度 {score:.4f}  {text[:40].replace(chr(10), ' ')}...")

print("\n② 生成中 ...\n")
answer = ask_llm(question, [t for _, t in hits])

print("=" * 50)
print(answer)
print("=" * 50)