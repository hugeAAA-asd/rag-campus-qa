"""第 2 天：切块 + 向量化 + 存成 index.json。

运行：python step2_build_index.py
"""
import json

from rag_core import CORPUS_FILE, EMBED_MODEL, INDEX_FILE, embed_texts, split_text

CHUNK_SIZE = 400
OVERLAP = 80

corpus_file = CORPUS_FILE
if not corpus_file.exists():
    raise SystemExit(f"找不到语料文件：{corpus_file}")

text = corpus_file.read_text(encoding="utf-8")
print(f"语料共 {len(text)} 字\n")

chunks = split_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
print(f"切成 {len(chunks)} 块（chunk_size={CHUNK_SIZE}, overlap={OVERLAP}）\n")

for i, c in enumerate(chunks[:3], 1):
    print(f"--- 第 {i} 块（{len(c)} 字）---")
    print(c[:100].replace("\n", " ") + "...")
    print()

print("开始向量化 ...")
vectors = embed_texts(chunks)

index = {
    "model": EMBED_MODEL,
    "chunk_size": CHUNK_SIZE,
    "overlap": OVERLAP,
    "chunks": [{"text": c, "embedding": v} for c, v in zip(chunks, vectors)],
}
INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

print(f"\n索引已写入：{INDEX_FILE}")
print(f"文件大小：{INDEX_FILE.stat().st_size / 1024:.1f} KB")