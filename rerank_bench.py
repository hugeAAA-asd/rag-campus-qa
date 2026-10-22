"""检索质量评测：把"正确答案排第几名"量化出来。

它不测大模型，只测**检索环节**——这是 RAG 里最容易出问题的地方。

用法:
    python benchmark.py

输出四层:
    1. 每题里正确答案排第几名
    2. recall@k（正确答案落在前 k 块的比例）
    3. 排名分布
    4. 要达到 X% 覆盖率，TOP_K 至少要设多大
"""
import json
import os
from pathlib import Path

import requests
from collections import Counter

from rag_core import EMBED_API_KEY, INDEX_FILE, TOP_K, cosine, embed_texts
RERANK_LOCAL_MODEL = "BAAI/bge-reranker-base"


def _cached_model_dir(repo_id):
    """在 HF 缓存里找模型快照目录；找不到返回 None。

    直接传目录路径给 CrossEncoder，它就完全不会联网。
    """
    home = os.getenv("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    snaps = Path(home) / "hub" / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return None
    subs = [x for x in snaps.iterdir() if x.is_dir()]
    return max(subs, key=lambda x: x.stat().st_mtime) if subs else None


RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RECALL_K = 10        # 向量先召回几个候选
# (问题, 答案所在块里必然出现的独特短语)
TESTS = [
    ("本科最长学习年限是多少年", "最长学习年限为六年"),
    ("新生逾期多久不报到算放弃入学", "逾期两周未报到"),
    ("新生入学后多久进行资格复查", "三个月内进行资格复查"),
    ("暂缓注册最长可以多久", "暂缓期最长一个月"),
    ("休学最多累计几年", "累计不得超过两学年"),
    ("复学需要提交什么医院的材料", "二级甲等以上医院"),
    ("申请转专业对绩点有什么要求", "平均学分绩点不低于3.2"),
    ("每学期选课学分上限是多少", "选课学分上限为三十学分"),
    ("选课分为哪几个阶段", "预选和正选两个阶段"),
    ("开学后多久可以自由退课", "开学后两周内可以自由退课"),
    ("补考还不及格怎么办", "补考仍不及格的"),
    ("免修考试要考到多少分", "成绩达到八十五分以上"),
    ("跨院系选修最多能认定多少学分", "最多可认定六学分"),
    ("考试时可以带什么资料", "不得携带任何与考试内容相关的资料"),
    ("代替他人考试会受什么处分", "代替他人考试或者让他人代替自己考试的"),
    ("什么情况可以申请缓考", "直系亲属重大变故"),
    ("平时成绩占总评多少", "平时成绩一般占百分之三十"),
    ("九十分以上换算成多少绩点", "九十分以上为4.0"),
    ("成绩复核要在多久内申请", "十个工作日内申请成绩复核"),
    ("优秀学生奖学金分几个等级", "分为一等、二等、三等三个等级"),
    ("奖学金一等每年多少钱", "一等每人每年五千元"),
    ("国家助学金分几档", "分为两档"),
    ("助学贷款毕业后多久开始还本", "还本宽限期为三年"),
    ("勤工助学每月最多工作多少小时", "每月工作时长不得超过四十小时"),
    ("宿舍每人每月免费用多少电", "每人每月免费额度为十五度"),
    ("本科生最多能借几本书", "本科生可借图书二十册"),
    ("图书逾期费最多收多少", "单册逾期费上限为二十元"),
    ("图书馆占座多久会被清理", "离开超过三十分钟"),
    ("进实验室必须穿什么", "必须穿戴实验服和防护眼镜"),
    ("社团最少要多少人才能注册", "社团人数不得少于二十人"),
    ("举办大型活动要提前多久报批", "提前十个工作日提交活动方案"),
    ("什么情况会被开除学籍", "触犯国家法律构成刑事犯罪"),
]
_rerank_model = None


def call_rerank(question, documents):
    """本地 cross-encoder 重排。

    返回 [{"index": i, "relevance_score": s}, ...]，格式和云端 API 一致，
    所以外面那段 item["index"] 的逻辑不用改。
    """
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        target = _cached_model_dir(RERANK_LOCAL_MODEL) or RERANK_LOCAL_MODEL
        if target == RERANK_LOCAL_MODEL:
            print("  缓存里没找到，将尝试联网下载（需要能访问 hf-mirror）...")
        else:
            print("  命中本地缓存，离线加载（不联网）")
        _rerank_model = CrossEncoder(str(target))

    scores = _rerank_model.predict([(question, d) for d in documents])

    items = [{"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)]
    items.sort(key=lambda x: x["relevance_score"], reverse=True)
    return items

def main():
    if not INDEX_FILE.exists():
        raise SystemExit("还没有索引，请先运行 step2_build_index.py")

    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    chunks = index["chunks"]
    n = len(TESTS)

    print(f"索引: {len(chunks)} 块  chunk_size={index['chunk_size']}  overlap={index['overlap']}")
    print(f"当前配置: TOP_K={TOP_K}    测试题目: {n}")
    print("-" * 68)

    q_vecs = embed_texts([q for q, _ in TESTS], is_query=True)

    vector_ranks = []
    rerank_ranks = []
    for (question, keyword), qv in zip(TESTS, q_vecs):
        scored = sorted(
            ((cosine(qv, c["embedding"]), i) for i, c in enumerate(chunks)),
            key=lambda x: x[0],
            reverse=True,
        )
        candidates = scored[:RECALL_K]
        vector_rank = None
        for pos, (_score, idx) in enumerate(candidates, 1):
            if keyword in chunks[idx]["text"]:
                vector_rank = pos
                break
        vector_ranks.append(vector_rank)

        docs = [chunks[idx]["text"] for _s,idx in candidates]
        results = call_rerank(question, docs)

        rerank_rank = None
        for pos,item in enumerate(results,1):
            for pos, item in enumerate(results, 1):
                block_no = candidates[item["index"]][1]  # 块编号（int）
                text = chunks[block_no]["text"]  # 用编号取文本
                if keyword in text:
                    rerank_rank = pos
                    break
                break
        rerank_ranks.append(rerank_rank)

    print("-" * 68)
    print(f"  {'方案':<16}{'recall@1':>10}{'recall@3':>10}")
    for name, rs in (("纯向量", vector_ranks), ("向量+rerank", rerank_ranks)):
        r1 = sum(1 for r in rs if r and r <= 1)
        r3 = sum(1 for r in rs if r and r <= 3)
        print(f"  {name:<16}{r1:>4}/{n:<6}{r3:>4}/{n}")

    print("-" * 68)
    print("  未命中（前 10 里都没有正确答案）：")
    for name, rs in (("纯向量", vector_ranks), ("向量+rerank", rerank_ranks)):
        miss = sum(1 for r in rs if not r)
        print(f"    {name:<16} {miss} 题")


if __name__ == "__main__":
    main()