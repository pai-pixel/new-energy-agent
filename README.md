# ⚡ 新能源行业垂直智能体

基于 **DeepSeek API** 的新能源行业 AI 助手，通过 FastAPI + 静态 HTML 提供 Web Chat 界面。

## 功能

| 功能 | 说明 |
|------|------|
| 📊 上网电价查询 | 按省份 + 月份查询上网电价 |
| ⚡ 脱硫煤电价查询 | 按省份 + 月份查询脱硫煤标杆电价 |
| 🏭 工商业电价查询 | 按省份 + 月份查询工商业用电价格 |
| 🌤️ 天气查询 | 查询各城市实时天气（wttr.in） |
| 📚 新能源政策知识 | 联网搜索 + AI 综合回答 |
| 💬 多轮上下文 | 自动继承省份、电价类型，场景随意切换 |
| 🗄️ 智能缓存 | 首次查询联网搜索真实数据并入库，再次查询秒回（越查越快） |

## 架构

```
用户 → Web Chat UI → FastAPI (/api/chat) → Agent 引擎 → DeepSeek API
                              ├── 电价查询 (SQLite 缓存优先 + Web 搜索兜底)
                              ├── 天气查询 (wttr.in)
                              ├── Web 搜索 (DuckDuckGo)
                              └── 安全过滤器 (三层防御)
```

### 电价缓存策略 (hybrid)

1. 查询本地 SQLite 数据库 (`data/electricity_prices.db`) — 命中即秒回
2. 未命中 → Web 搜索真实数据 → LLM 提取价格 → **写入数据库缓存**
3. 下次同一省份 + 月份 + 类型直接命中缓存，毫秒级返回

## 快速开始

### 1. 获取 DeepSeek API Key

注册 [DeepSeek 平台](https://platform.deepseek.com)，获取 API Key。

### 2. 配置环境变量

```bash
# 方式一：设置环境变量
export DEEPSEEK_API_KEY=sk-your-api-key-here

# 方式二：创建 .env 文件
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 3. 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. 启动

```bash
python src/server.py
```

浏览器打开 `http://localhost:7860`。

> 首次查询某省电价会自动联网搜索真实数据并写入 `data/electricity_prices.db`，无需手动初始化数据库。

## 项目结构

```
new-energy-agent/
├── src/
│   ├── agent.py                  # Agent 主循环 (工具调用 + 上下文管理)
│   ├── config.py                 # 配置 + 环境变量读取 + 省份/电价类型映射
│   ├── model_engine.py           # DeepSeek API 调用封装
│   ├── safety_guard.py           # 三层安全过滤器
│   ├── server.py                 # FastAPI 后端 (/api/chat, /api/reset)
│   └── tools/
│       ├── electricity_query.py  # 电价查询 (DB缓存优先 + Web搜索兜底 + 入库)
│       ├── weather_query.py      # wttr.in 天气查询
│       └── web_search.py         # DuckDuckGo 搜索
├── ui/
│   └── index.html                # 聊天界面 (静态页, 深色主题)
├── data/
│   ├── db.py                     # SQLite 数据库操作
│   └── seed.py                   # (可选) 离线 Mock 数据脚本，默认不使用
├── requirements.txt
├── .env.example
└── README.md
```

## 安全机制

| 层级 | 防御 | 说明 |
|------|------|------|
| L1 | 关键词正则黑名单 | 推理前，零 LLM 开销 |
| L2 | System Prompt 安全指令 | 推理中 |
| L3 | 领域边界判断 | 推理后，非新能源领域拒答 |

## 上下文管理示例

| 轮次 | 用户输入 | 智能体理解 |
|------|---------|-----------|
| 1 | "上海的上网电价" | 查询上海上网电价 |
| 2 | "那江苏呢" | 继承"上网电价"，查江苏 |
| 3 | "工商业电价呢" | 继承"江苏"，查工商业电价 |
| 4 | "那天气呢" | 继承"江苏"，查天气 |

## 电价数据

采用 **Web 搜索真实数据 + 本地 SQLite 缓存** 方案：
- 数据库无数据 → Web 搜索真实电价 → LLM 提取 → 写入缓存
- 数据库有数据 → 毫秒级返回（越查越快）
- 数据来源为公开搜索结果的真实电价，非 Mock 数据

## 模型接口

模型层预留了可替换接口，当前对接 DeepSeek API (OpenAI 兼容)。

如需切换其他模型，修改 `config.py` 中的配置：
```python
LLM_BASE_URL = "https://your-api.com"
LLM_MODEL = "your-model-name"
```

或通过环境变量：
```bash
export DEEPSEEK_BASE_URL=https://your-api.com
export DEEPSEEK_MODEL=your-model-name
```

## 许可证

MIT
