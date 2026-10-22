# 📚 校园知识库智能问答系统（RAG Demo）

基于 RAG（检索增强生成）技术实现的校园校规问答系统。不依赖 LangChain，从零手写每一步，方便理解 RAG 原理。

## ✨ 功能特性

- 🔍 **本地 Embedding**：使用 `BAAI/bge-small-zh-v1.5` 本地模型，无需调用远程 Embedding API
- 📝 **智能切块**：按中文标点切分文本，带 overlap 重叠，避免答案卡在切口
- 🎯 **向量检索**：手写余弦相似度计算，返回 Top-K 最相关文档块
- 🤖 **大模型生成**：接入 DeepSeek API，基于检索到的资料回答，不编造
- 💬 **网页聊天界面**：Streamlit 搭建，支持多轮对话、对话历史持久化
- 📄 **参考资料溯源**：每条回答下方可展开查看检索到的原文片段和相似度分数
- 📊 **知识库统计**：侧边栏实时显示文档块数量、总字符数

## 🛠️ 技术栈

| 模块 | 技术选型 |
|---|---|
| Embedding 模型 | BAAI/bge-small-zh-v1.5（本地） |
| 向量检索 | 手写余弦相似度 + JSON 索引 |
| 大模型 | DeepSeek-Flash API |
| 网页框架 | Streamlit |
| 配置管理 | python-dotenv |

## 📁 项目结构

```
rag_demo/
├── app.py                  # Streamlit 网页入口
├── rag_core.py             # RAG 核心库：切块 / 向量化 / 检索 / 生成
├── benchmark.py            # 32 题检索基准，衡量 recall@k
├── step1_vector.py         # 练习 1：只看向量相似度
├── step2_build_index.py    # 练习 2：切块 + 建索引
├── step3_search.py         # 练习 3：只检索不调模型（排错用）
├── step4_ask.py            # 练习 4：检索 + 生成
├── step5_chat.py           # 练习 5：连续问答（模型只加载一次）
├── corpus_big.txt          # 语料：学生手册（约 3700 字，13 块）
├── index.json              # 向量索引（首次运行生成，不提交）
├── .env.example            # 配置模板
├── .env                    # 真实密钥（不提交）
├── requirements.txt        # 基础依赖
└── requirements-local.txt  # 本地模型额外依赖
```

## 🚀 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-local.txt  # 本地模型依赖（torch + sentence-transformers）
```

### 2. 配置密钥

复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=sk-你的key
```

### 3. 建索引（首次必须执行）

仓库**不包含** `index.json`——它由语料生成，体积大且可随时重建，所以被 `.gitignore` 排除了。

第一次运行前，必须先建索引：

```bash
python step2_build_index.py
```

看到「索引已写入」即成功。**以后只要改了 `corpus_big.txt`，都要重跑这一步。**

### 4. 启动网页版

```bash
streamlit run app.py
```

浏览器会自动打开，输入问题即可开始问答。

## 📊 检索效果

自建 32 题检索基准（`benchmark.py`，覆盖语料全部章节），衡量**检索环节**的 recall@k：

| 配置 | recall@1 | recall@3 |
|---|---|---|
| 本地 `bge-small-zh`（512 维） | 25/32 = 78% | 32/32 = 100% |
| 云端 `bge-m3`（1024 维） | 29/32 = 91% | 32/32 = 100% |
| 本地 + `bge-reranker-base` 重排 | **32/32 = 100%** | 32/32 = 100% |

**结论**

1. **瓶颈在 embedding 模型的分辨率，不在切块策略**——`chunk_size` 在
   300 / 400 之间调整时 `recall@1` 无变化（78% → 78%），只是把失败的
   题目换了一批
2. **加入 cross-encoder 两阶段重排**后，原本排第 2~3 名的 7 个问题
   全部被提到第 1 名
3. 该实验的候选池为 10 / 13 块（占语料 77%），高于真实场景的召回比例，
   因此提升幅度**不宜直接外推**

## 💡 示例提问

- 奖学金申请条件是什么？
- 宿舍晚上几点熄灯？
- 每学期最多能选多少学分？
- 考试作弊会有什么处分？
- 图书馆借书能借多久？

## 📖 RAG 原理

```
用户提问
   ↓
问题向量化（Embedding）
   ↓
与知识库所有文档块计算余弦相似度
   ↓
取 Top-K 最相关的文档块
   ↓
把文档块 + 问题一起塞进 Prompt
   ↓
大模型基于资料生成回答
```

## 📝 许可证

MIT
