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
from collections import Counter

from rag_core import INDEX_FILE, TOP_K, cosine, embed_texts

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

    ranks = []
    for (question, keyword), qv in zip(TESTS, q_vecs):
        scored = sorted(
            ((cosine(qv, c["embedding"]), i) for i, c in enumerate(chunks)),
            key=lambda x: x[0],
            reverse=True,
        )
        rank = None
        for pos, (_score, idx) in enumerate(scored, 1):
            if keyword in chunks[idx]["text"]:
                rank = pos
                break
        ranks.append(rank)

        if rank is None:
            label = "未命中"
        else:
            label = f"第{rank}名"
            if rank > TOP_K:
                label += " ❌"
        print(f"  {label:<12} {question}")

    print("-" * 68)
    for k in (1, 2, 3, 4, 5, 10):
        hit = sum(1 for r in ranks if r and r <= k)
        bar = "#" * hit
        print(f"  recall@{k:<3} {hit:>2}/{n} = {hit / n:>5.0%}  {bar}")

    print("-" * 68)
    print("  排名分布:")
    dist = Counter(r if r else 999 for r in ranks)
    for rank in sorted(dist):
        name = "未命中" if rank == 999 else f"第{rank}名"
        print(f"    {name:<8} {dist[rank]:>2} 题  {'*' * dist[rank]}")

    print("-" * 68)
    max_rank = max((r for r in ranks if r), default=0)
    print("  要达到指定覆盖率，TOP_K 至少需要:")
    for target in (0.80, 0.90, 0.95, 1.00):
        need = None
        for k in range(1, max_rank + 1):
            if sum(1 for r in ranks if r and r <= k) / n >= target:
                need = k
                break
        if need:
            print(f"    {target:>4.0%} 覆盖  ->  TOP_K >= {need}")
        else:
            print(f"    {target:>4.0%} 覆盖  ->  即使给全部块也达不到")


if __name__ == "__main__":
    main()