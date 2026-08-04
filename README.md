# ⚡ 新能源行业垂直智能体

基于 **Qwen2.5-3B-Instruct (AWQ 4-bit)** 的新能源行业 AI 助手，部署在 **Google Colab Free (T4 GPU)**，通过 Gradio 提供公网 Web 访问。

## 功能

| 功能 | 说明 |
|------|------|
| 📊 上网电价查询 | 按省份 + 月份查询上网电价，含趋势图表 |
| ⚡ 脱硫煤电价查询 | 按省份 + 月份查询脱硫煤标杆电价 |
| 🏭 工商业电价查询 | 按省份 + 月份查询工商业用电价格 |
| 🌤️ 天气查询 | 查询各城市实时天气（wttr.in） |
| 📚 新能源政策知识 | 联网搜索 + AI 综合回答 |
| 💬 多轮上下文 | 自动继承省份、电价类型，场景随意切换 |

## 架构

```
用户 → Gradio Chat UI → Agent 引擎 → vLLM (Qwen2.5-3B)
                              ├── 电价实时查询 (Web Search → LLM提取 → SQLite缓存)
                              ├── 天气查询 (wttr.in)
                              ├── Web 搜索 (DuckDuckGo)
                              └── 安全过滤器 (三层防御)
```

## 快速开始（Colab 部署）

### 1. 配置 Colab Secrets

在 Colab 侧边栏 → 🔑 Secrets 面板中添加：

| Secret | 值 | 说明 |
|--------|-----|------|
| `HF_TOKEN` | `hf_xxx` | HuggingFace Token（可选） |
| `NGROK_TOKEN` | `xxx` | ngrok Token（可选） |

### 2. 运行 Notebook

打开 `colab_notebook.ipynb`，从上到下依次运行每个 Cell。

首次运行会从 HuggingFace Hub 下载模型（~2.5GB，约 2-5 分钟），后续运行缓存到 Google Drive。

### 3. 获取公网 URL

运行完成后，Cell 7 会输出：
- Gradio 公网 URL: `https://xxxx.gradio.live`
- ngrok URL（如已配置）: `https://xxxx.ngrok-free.app`

## 项目结构

```
new-energy-agent/
├── colab_notebook.ipynb          # Colab 一键部署 Notebook
├── src/
│   ├── agent.py                  # Agent 主循环 + Gradio UI
│   ├── config.py                 # 配置 + Secrets 读取 + 省份映射
│   ├── intent_router.py          # 意图分类 + 实体抽取
│   ├── context_manager.py        # 多轮对话状态管理
│   ├── safety_guard.py           # 三层安全过滤器
│   ├── prompt_templates.py       # System Prompt + Few-shot 模板
│   ├── tools/
│   │   ├── electricity_query.py  # 电价实时查询 (Search→Fetch→LLM→Cache)
│   │   ├── weather_query.py      # wttr.in 天气查询
│   │   └── web_search.py         # DuckDuckGo 搜索 + 网页抓取
│   └── visualization/
│       ├── chart_builder.py      # Plotly 趋势图/对比图
│       └── table_builder.py      # Dataframe 数据表
├── requirements.txt
└── .env.example                  # 本地开发参考（不含真实值）
```

## 安全机制

| 层级 | 防御 | 位置 |
|------|------|------|
| L1 | 关键词正则黑名单 | 推理前 |
| L2 | System Prompt 安全指令 | 推理中 |
| L3 | 领域边界判断 | 推理后 |

## 电价数据来源

采用 **实时 Web Search + LLM 提取 + SQLite 缓存** 方案：
- 首次查询某省电价时，自动搜索北极星电力网/发改委等公开页面
- LLM 从页面文本中提取电价数字
- 结果写入 SQLite 缓存，同省同月后续查询 < 50ms 返回
- 无需爬虫，无需定时任务，零维护

## 注意事项

- Colab 免费版运行约 4-12 小时后自动断开
- 模型推理速度约 40 tokens/s
- 电价数据来自公开网页，仅供参考
- AI 生成内容请以官方数据为准

## 许可证

MIT
