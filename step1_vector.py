"""第 1 天：只玩向量，不碰 RAG。

目标：亲眼看到"意思相近"的句子向量距离更近，而不是字面相近。
运行：python step1_vector.py
"""
from rag_core import cosine, embed_texts

sentences = [
    "今天天气怎么样",
    "外面下雨了吗",
    "苹果手机多少钱",
    "这道数学题怎么解",
]

print("正在请求向量 ...\n")
vectors = embed_texts(sentences)

print(f"\n向量维度：{len(vectors[0])}")
print("每个向量就是这么多数字组成的一串，肉眼不可读，但机器能比较它们。\n")

print("两两相似度（1 = 完全一样，0 = 毫无关系）：")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = cosine(vectors[i], vectors[j])
        print(f"  {score:6.4f}   {sentences[i]}  vs  {sentences[j]}")

print("\n注意看：'天气'和'下雨'用词完全不同，但相似度很高。")
print("这就是 RAG 的地基——它比的是「意思」，不是「字面」。")