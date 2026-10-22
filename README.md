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
├── app.py              # Streamlit 网页入口
├── rag_core.py         # RAG 核心库：切块/向量化/检索/生成
├── corpus_big.txt      # 完整学生手册语料
├── index.json          # 向量索引文件（自动生成）
├── .env                # 密钥配置（不提交 git）
└── requirements.txt    # 依赖
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

### 3. 启动网页版

```bash
streamlit run app.py
```

浏览器会自动打开，输入问题即可开始问答。

## 📊 检索效果

| 方案 | recall@1 | recall@3 |
|---|---|---|
| 纯向量检索 | 25/32 | 32/32 |
| 向量 + Rerank 重排序 | 32/32 | 32/32 |

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
