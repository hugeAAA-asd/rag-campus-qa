"""连续问答模式：模型只加载一次，之后每问一句都很快。

为什么需要这个文件：
    step4_ask.py 是"问一句就退出"，每次运行都要重新 import torch（约 9 秒）。
    这个版本把加载放在循环外面，那 9 秒一次会话只付一次。

运行：python step5_chat.py
"""
from rag_core import ask_llm, embed_texts, search

print("正在加载本地模型（约 10 秒，只这一次）...")
embed_texts(["预热"])
print("就绪。直接提问，输入 exit 退出。\n")

while True:
    try:
        question = input("你：" ).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not question:
        continue
    if question.lower() in ("exit", "quit", "q"):
        break

    hits = search(question)
    print("[检索] " + "  |  ".join(f"{s:.3f}" for s, _ in hits))
    print(f"[AI] {ask_llm(question, [t for _, t in hits])}\n")